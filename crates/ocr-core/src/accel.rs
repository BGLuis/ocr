//! Execution-provider selection and the honest CUDA probe (report section 2.9).
//!
//! CPU is the baseline. CUDA is used only when it is actually usable: ORT must
//! be built with the provider **and** its shared-library dependencies must be
//! present. `--accel cuda` turns a failed probe into a hard error instead of a
//! silent CPU fallback (report section 4).

use std::path::{Path, PathBuf};

use oar_ocr::core::config::{OrtExecutionProvider, OrtSessionConfig};
pub use ocr_proto::Accel;

use crate::error::{Error, Result};

/// What the daemon should actually do about acceleration.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Providers in preference order, ready for `OrtSessionConfig`.
    pub providers: Vec<OrtExecutionProvider>,
    /// The provider that will really run the graph (first non-CPU entry, or
    /// CPU). Logged at startup and echoed in every response.
    pub effective: Effective,
    /// Human-readable reason for the decision, for the startup log.
    pub note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effective {
    Cpu,
    Cuda,
}

impl Effective {
    pub fn as_str(&self) -> &'static str {
        match self {
            Effective::Cpu => "cpu",
            Effective::Cuda => "cuda",
        }
    }
}

impl Plan {
    /// A plain CPU plan.
    fn cpu(note: impl Into<String>) -> Self {
        Plan {
            providers: vec![OrtExecutionProvider::CPU],
            effective: Effective::Cpu,
            note: note.into(),
        }
    }

    /// CUDA first, CPU fallback after it.
    fn cuda(device_id: i32, note: impl Into<String>) -> Self {
        Plan {
            providers: vec![
                OrtExecutionProvider::CUDA {
                    device_id: Some(device_id),
                    gpu_mem_limit: None,
                    arena_extend_strategy: None,
                    // `Default` (fixed heuristic, no per-shape benchmarking):
                    // OCR feeds variable-width tensors, so `Exhaustive` would
                    // re-search on almost every call. oar-ocr documents the
                    // same choice internally.
                    cudnn_conv_algo_search: Some("default".to_string()),
                    cudnn_conv_use_max_workspace: None,
                },
                OrtExecutionProvider::CPU,
            ],
            effective: Effective::Cuda,
            note: note.into(),
        }
    }

    /// Turn the plan into an `OrtSessionConfig` for the oar-ocr builders.
    pub fn session_config(&self, intra_threads: Option<usize>) -> OrtSessionConfig {
        let mut cfg = OrtSessionConfig::new().with_execution_providers(self.providers.clone());
        if let Some(n) = intra_threads {
            cfg = cfg.with_intra_threads(n);
        }
        cfg
    }
}

/// Resolve the requested accelerator into a concrete [`Plan`].
///
/// * `Cpu` — always succeeds.
/// * `Cuda` — succeeds only if the probe passes; otherwise [`Error::CudaRequired`].
/// * `Auto` — CUDA if the probe passes, else CPU with the reason in `note`.
pub fn plan(requested: Accel, device_id: i32) -> Result<Plan> {
    match requested {
        Accel::Cpu => Ok(Plan::cpu("forced by --accel cpu")),
        Accel::Cuda => match probe() {
            Ok(()) => Ok(Plan::cuda(
                device_id,
                "required by --accel cuda; CUDA execution provider initialized",
            )),
            Err(why) => Err(Error::CudaRequired(why)),
        },
        Accel::Auto => match probe() {
            Ok(()) => Ok(Plan::cuda(device_id, "--accel auto: CUDA probe passed")),
            Err(why) => Ok(Plan::cpu(format!(
                "--accel auto: CUDA unavailable ({why}); using CPU"
            ))),
        },
    }
}

/// Returns `Ok(())` if the CUDA execution provider looks genuinely usable.
///
/// This is a static probe (build features + shared libraries on disk), not a
/// full session build. It is conservative on purpose: a false negative costs
/// GPU speed, a false positive costs a silent CPU fallback that the report
/// explicitly calls out. F0 replaces this with a real one-shot session once
/// models and cuDNN are in place.
#[allow(clippy::needless_return)] // one of the two cfg blocks is always compiled out
pub fn probe() -> std::result::Result<(), String> {
    #[cfg(not(feature = "cuda"))]
    {
        return Err("binary was built without the `cuda` feature".to_string());
    }

    #[cfg(feature = "cuda")]
    {
        use ort::ep::{ExecutionProvider, CUDA};

        match CUDA::default().is_available() {
            Ok(true) => {}
            Ok(false) => return Err("ONNX Runtime was not built with CUDA support".to_string()),
            Err(e) => return Err(format!("ONNX Runtime CUDA query failed: {e}")),
        }

        // ORT rc.13 links CUDA 12 / cuDNN 9 (`libcudart.so.12`, `libcudnn.so.9`).
        // The report's machine has CUDA 13.x and no cuDNN installed — both are a
        // `pacman -S` away, but until then this must fail rather than fall back
        // in silence.
        if find_lib("libcudnn.so.9").is_none() {
            return Err("cuDNN 9 (libcudnn.so.9) not found; install `extra/cudnn`".to_string());
        }
        if find_lib("libcudart.so.12").is_none() {
            return Err(
                "libcudart.so.12 not found; ORT rc.13 links CUDA 12 while this host has CUDA 13.x \
                 (see report section 4)"
                    .to_string(),
            );
        }

        Ok(())
    }
}

/// Search the usual loader paths for a shared object by file name.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn find_lib(name: &str) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for var in ["LD_LIBRARY_PATH", "CUDA_PATH"] {
        if let Ok(val) = std::env::var(var) {
            for p in val.split(':').filter(|s| !s.is_empty()) {
                dirs.push(PathBuf::from(p));
                dirs.push(Path::new(p).join("lib64"));
                dirs.push(Path::new(p).join("lib"));
            }
        }
    }
    dirs.extend(
        [
            "/usr/lib",
            "/usr/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/lib",
            "/usr/local/lib64",
            "/lib",
            "/lib64",
            "/opt/cuda/lib64",
            "/opt/cuda/lib",
            "/usr/local/cuda/lib64",
        ]
        .iter()
        .map(PathBuf::from),
    );

    dirs.into_iter()
        .map(|d| d.join(name))
        .find(|p| p.exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_plan_is_always_cpu() {
        let p = plan(Accel::Cpu, 0).unwrap();
        assert_eq!(p.effective, Effective::Cpu);
        assert_eq!(p.providers.len(), 1);
        assert!(matches!(p.providers[0], OrtExecutionProvider::CPU));
    }

    #[test]
    fn auto_falls_back_to_cpu_when_probe_fails() {
        // In CI / on this host the probe fails (no cuDNN); auto must still
        // produce a usable CPU plan and record why.
        if probe().is_err() {
            let p = plan(Accel::Auto, 0).unwrap();
            assert_eq!(p.effective, Effective::Cpu);
            assert!(p.note.contains("CUDA unavailable"));
        }
    }

    #[test]
    fn cuda_is_fatal_when_probe_fails() {
        if let Err(why) = probe() {
            let err = plan(Accel::Cuda, 0).unwrap_err();
            assert!(matches!(err, Error::CudaRequired(_)));
            assert!(err.to_string().contains(&why) || err.to_string().contains("CUDA"));
        }
    }
}
