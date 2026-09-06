//! Execution-provider selection and honest accelerator probes (report §2.9).
//!
//! CPU is the baseline. A GPU provider is used only when it is actually usable:
//! ORT must be built with it **and** its runtime dependencies must be present.
//! `--accel <gpu>` turns a failed probe into a hard error instead of a silent
//! CPU fallback (report §4).
//!
//! Vendors:
//! * **NVIDIA** — `cuda` feature, `--accel cuda` (or `auto`).
//! * **Intel** (iGPU / Arc / NPU) — `openvino` feature, `--accel openvino`.
//! * **AMD on Linux** — `webgpu` feature, `--accel webgpu` (vendor-neutral via
//!   Vulkan; ROCm is not wired through the pipeline library).
//! * **Windows, any vendor** — `directml` feature, `--accel directml`.
//!
//! `auto` only ever tries CUDA then CPU — it never silently reaches for a vendor
//! backend. Everything else is an explicit opt-in.

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
    Openvino,
    Webgpu,
    Directml,
}

impl Effective {
    pub fn as_str(&self) -> &'static str {
        match self {
            Effective::Cpu => "cpu",
            Effective::Cuda => "cuda",
            Effective::Openvino => "openvino",
            Effective::Webgpu => "webgpu",
            Effective::Directml => "directml",
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

    /// `gpu` first, CPU fallback after it.
    fn accelerated(
        gpu: OrtExecutionProvider,
        effective: Effective,
        note: impl Into<String>,
    ) -> Self {
        Plan {
            providers: vec![gpu, OrtExecutionProvider::CPU],
            effective,
            note: note.into(),
        }
    }

    fn cuda(device_id: i32, note: impl Into<String>) -> Self {
        Plan::accelerated(
            OrtExecutionProvider::CUDA {
                device_id: Some(device_id),
                gpu_mem_limit: None,
                arena_extend_strategy: None,
                // `Default` (fixed heuristic, no per-shape benchmarking): OCR
                // feeds variable-width tensors, so `Exhaustive` would re-search
                // on almost every call. oar-ocr documents the same choice.
                cudnn_conv_algo_search: Some("default".to_string()),
                cudnn_conv_use_max_workspace: None,
            },
            Effective::Cuda,
            note,
        )
    }

    fn openvino(note: impl Into<String>) -> Self {
        Plan::accelerated(
            OrtExecutionProvider::OpenVINO {
                // The user asked for the graphics card; "GPU" targets the Intel
                // iGPU / Arc. Falls to CPU only via the second provider, not
                // silently inside OpenVINO.
                device_type: Some(
                    std::env::var("OCRD_OPENVINO_DEVICE").unwrap_or_else(|_| "GPU".to_string()),
                ),
                num_threads: None,
            },
            Effective::Openvino,
            note,
        )
    }

    fn webgpu(note: impl Into<String>) -> Self {
        Plan::accelerated(OrtExecutionProvider::WebGPU, Effective::Webgpu, note)
    }

    fn directml(device_id: i32, note: impl Into<String>) -> Self {
        Plan::accelerated(
            OrtExecutionProvider::DirectML {
                device_id: Some(device_id),
            },
            Effective::Directml,
            note,
        )
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
/// * `Cuda` / `Openvino` / `Webgpu` / `Directml` — succeed only if the matching
///   probe passes; otherwise [`Error::AccelUnavailable`] (fatal, no fallback).
/// * `Auto` — CUDA if its probe passes, else CPU with the reason in `note`.
pub fn plan(requested: Accel, device_id: i32) -> Result<Plan> {
    match requested {
        Accel::Cpu => Ok(Plan::cpu("forced by --accel cpu")),
        Accel::Cuda => require("cuda", probe_cuda(), |note| Plan::cuda(device_id, note)),
        Accel::Openvino => require("openvino", probe_openvino(), Plan::openvino),
        Accel::Webgpu => require("webgpu", probe_webgpu(), Plan::webgpu),
        Accel::Directml => require("directml", probe_directml(), |note| {
            Plan::directml(device_id, note)
        }),
        Accel::Auto => match probe_cuda() {
            Ok(()) => Ok(Plan::cuda(device_id, "--accel auto: CUDA probe passed")),
            Err(why) => Ok(Plan::cpu(format!(
                "--accel auto: CUDA unavailable ({why}); using CPU. \
                 Intel GPU: --accel openvino. AMD GPU: --accel webgpu."
            ))),
        },
    }
}

fn require(
    name: &'static str,
    probe: std::result::Result<(), String>,
    build: impl FnOnce(String) -> Plan,
) -> Result<Plan> {
    match probe {
        Ok(()) => Ok(build(format!(
            "required by --accel {name}; execution provider initialized"
        ))),
        Err(reason) => Err(Error::AccelUnavailable {
            provider: name,
            reason,
        }),
    }
}

/// Back-compat alias: the CUDA probe.
pub fn probe() -> std::result::Result<(), String> {
    probe_cuda()
}

/// `Ok(())` if the CUDA execution provider looks genuinely usable.
///
/// Static probe (build feature + shared libraries on disk), not a full session
/// build — conservative on purpose: a false negative costs GPU speed, a false
/// positive costs the silent CPU fallback the report calls out. F0 replaces
/// this with a real one-shot session once the stack is in place.
#[allow(clippy::needless_return)] // one cfg block is always compiled out
pub fn probe_cuda() -> std::result::Result<(), String> {
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

        // ORT rc.13 links CUDA 12 / cuDNN 9.
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

/// `Ok(())` if the OpenVINO execution provider (Intel GPU/NPU/CPU) is usable.
#[allow(clippy::needless_return)]
pub fn probe_openvino() -> std::result::Result<(), String> {
    #[cfg(not(feature = "openvino"))]
    {
        return Err("binary was built without the `openvino` feature".to_string());
    }

    #[cfg(feature = "openvino")]
    {
        use ort::ep::{ExecutionProvider, OpenVINO};

        match OpenVINO::default().is_available() {
            Ok(true) => {}
            Ok(false) => {
                return Err(
                    "ONNX Runtime was not built with OpenVINO support. The prebuilt binaries \
                     from `download-binaries` omit it — build ORT against the OpenVINO runtime \
                     and point `ORT_LIB_LOCATION` at it."
                        .to_string(),
                )
            }
            Err(e) => return Err(format!("OpenVINO query failed: {e}")),
        }

        if find_lib("libopenvino.so").is_none() && find_lib("libopenvino_c.so").is_none() {
            return Err(
                "OpenVINO runtime (libopenvino.so) not found; install the `openvino` / \
                 `intel-openvino` package and source its setvars.sh"
                    .to_string(),
            );
        }

        Ok(())
    }
}

/// WebGPU is the vendor-neutral GPU path (Vulkan on Linux — the practical route
/// for AMD, since ROCm is not wired through oar-ocr). It is **not reachable with
/// the prebuilt ONNX Runtime**: enabling `ort/webgpu` makes the binary depend on
/// `libwebgpu_dawn.so`, which `download-binaries` does not ship, so the process
/// fails to load before it can run at all. Until ONNX Runtime is built from
/// source with Dawn and linked via `ORT_STRATEGY=system`, this always errors.
pub fn probe_webgpu() -> std::result::Result<(), String> {
    Err(
        "WebGPU is not available in this build. The prebuilt ONNX Runtime does not ship \
         libwebgpu_dawn.so; build ONNX Runtime from source with WebGPU/Dawn and link it with \
         ORT_STRATEGY=system (https://ort.pyke.io/setup/linking)."
            .to_string(),
    )
}

/// `Ok(())` if the DirectML execution provider is usable. Windows only.
#[allow(clippy::needless_return)]
pub fn probe_directml() -> std::result::Result<(), String> {
    #[cfg(not(feature = "directml"))]
    {
        return Err("binary was built without the `directml` feature".to_string());
    }

    #[cfg(feature = "directml")]
    {
        #[cfg(not(target_os = "windows"))]
        {
            return Err("DirectML is a Windows-only execution provider".to_string());
        }

        #[cfg(target_os = "windows")]
        {
            use ort::ep::{DirectML, ExecutionProvider};

            match DirectML::default().is_available() {
                Ok(true) => Ok(()),
                Ok(false) => {
                    Err("ONNX Runtime was not built with DirectML support".to_string())
                }
                Err(e) => Err(format!("DirectML query failed: {e}")),
            }
        }
    }
}

/// Search the usual loader paths for a shared object by file name.
#[cfg_attr(
    not(any(feature = "cuda", feature = "openvino")),
    allow(dead_code)
)]
fn find_lib(name: &str) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for var in ["LD_LIBRARY_PATH", "CUDA_PATH", "INTEL_OPENVINO_DIR"] {
        if let Ok(val) = std::env::var(var) {
            for p in val.split(':').filter(|s| !s.is_empty()) {
                dirs.push(PathBuf::from(p));
                dirs.push(Path::new(p).join("lib64"));
                dirs.push(Path::new(p).join("lib"));
                dirs.push(Path::new(p).join("runtime/lib/intel64"));
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
            "/opt/intel/openvino/runtime/lib/intel64",
        ]
        .iter()
        .map(PathBuf::from),
    );

    dirs.into_iter().map(|d| d.join(name)).find(|p| p.exists())
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
    fn auto_falls_back_to_cpu_when_cuda_probe_fails() {
        if probe_cuda().is_err() {
            let p = plan(Accel::Auto, 0).unwrap();
            assert_eq!(p.effective, Effective::Cpu);
            assert!(p.note.contains("CUDA unavailable"));
            // The fallback note points Intel/AMD users at their flag.
            assert!(p.note.contains("openvino") && p.note.contains("webgpu"));
        }
    }

    #[test]
    fn explicit_gpu_flags_are_fatal_when_the_provider_is_absent() {
        for (accel, name) in [
            (Accel::Cuda, "cuda"),
            (Accel::Openvino, "openvino"),
            (Accel::Webgpu, "webgpu"),
            (Accel::Directml, "directml"),
        ] {
            let probe = match accel {
                Accel::Cuda => probe_cuda(),
                Accel::Openvino => probe_openvino(),
                Accel::Webgpu => probe_webgpu(),
                Accel::Directml => probe_directml(),
                _ => unreachable!(),
            };
            if probe.is_err() {
                let err = plan(accel, 0).unwrap_err();
                assert!(
                    matches!(err, Error::AccelUnavailable { provider, .. } if provider == name),
                    "{name}: unexpected {err:?}"
                );
            }
        }
    }
}
