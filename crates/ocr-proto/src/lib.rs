//! Wire protocol shared by `ocrd` (daemon) and `ocrd-client`.
//!
//! One request and one response per connection, each a single UTF-8 JSON object
//! terminated by `\n`. Image bytes travel base64-encoded inside the request so
//! the frame stays a plain line of text — a screenshot PNG is a few MB, well
//! within what a Unix socket and `serde_json` handle without streaming.
//!
//! This crate is deliberately tiny (serde only). The client links just this;
//! the heavy `oar-ocr`/`ort` stack lives behind `ocr-core` in the daemon
//! (report section 2.7: "cliente minúsculo").

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Default Unix socket path.
///
/// Prefers `$XDG_RUNTIME_DIR/ocrd.sock` — a per-user, mode-0700 directory on any
/// systemd/Wayland session, which is the normal case here. Without it, falls
/// back to a **per-user subdirectory** of the temp dir,
/// `<tmpdir>/ocrd-<user>/ocrd.sock`, never a bare file in a shared `/tmp`.
///
/// The daemon ([`ocrd`]) is what makes this safe: it creates the fallback
/// directory `0700`, refuses it if it already exists owned by another user, and
/// chmods the socket itself to `0600`. So the predictable name cannot be
/// squatted or connected to by other local users — which matters because the
/// socket carries screen captures and their recognized text.
pub fn default_socket_path() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("XDG_RUNTIME_DIR") {
        if !dir.is_empty() {
            return std::path::Path::new(&dir).join("ocrd.sock");
        }
    }
    let who = std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "user".to_string());
    std::env::temp_dir()
        .join(format!("ocrd-{who}"))
        .join("ocrd.sock")
}

/// Recognizer selection. Several human-friendly spellings collapse onto the
/// three PP-OCRv5 recognizer buckets (report section 2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Lang {
    /// Run the CJK recognizer, then re-try low-confidence boxes with the
    /// Korean and Latin recognizers and keep the best score.
    #[default]
    Auto,
    /// `pp-ocrv5_mobile_rec` — simplified/traditional Chinese, Japanese, English.
    Cjk,
    /// `korean_pp-ocrv5_mobile_rec`.
    Korean,
    /// `latin_pp-ocrv5_mobile_rec` — Latin scripts with diacritics
    /// (Portuguese, Spanish, French, German, ...).
    Latin,
}

impl Lang {
    pub fn as_str(&self) -> &'static str {
        match self {
            Lang::Auto => "auto",
            Lang::Cjk => "cjk",
            Lang::Korean => "korean",
            Lang::Latin => "latin",
        }
    }

    /// Every accepted spelling, for CLI help text.
    pub const ALIASES: &'static [&'static str] = &[
        "auto",
        "cjk",
        "ja",
        "japanese",
        "jpn",
        "zh",
        "chinese",
        "chi",
        "en",
        "english",
        "eng",
        "ko",
        "korean",
        "kor",
        "latin",
        "pt",
        "portuguese",
        "por",
        "es",
        "spanish",
        "fr",
        "french",
        "de",
        "german",
    ];
}

impl FromStr for Lang {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Ok(Lang::Auto),
            "cjk" | "ja" | "japanese" | "jpn" | "zh" | "chinese" | "chi" | "zh-hans"
            | "zh-hant" | "en" | "english" | "eng" => Ok(Lang::Cjk),
            "ko" | "korean" | "kor" => Ok(Lang::Korean),
            "latin" | "pt" | "portuguese" | "por" | "pt-br" | "es" | "spanish" | "spa" | "fr"
            | "french" | "fra" | "de" | "german" | "deu" | "it" | "italian" => Ok(Lang::Latin),
            other => Err(format!(
                "unknown language `{other}` (try one of: {})",
                Lang::ALIASES.join(", ")
            )),
        }
    }
}

impl fmt::Display for Lang {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Accelerator selection (report section 2.9).
///
/// `Auto` only ever tries CUDA (NVIDIA) then CPU — it never silently reaches for
/// a vendor GPU backend. Intel and AMD are explicit opt-ins:
/// `Openvino` for Intel iGPU/Arc, `Webgpu` for AMD on Linux (vendor-neutral via
/// Vulkan), `Directml` for any vendor on Windows. Each requires the daemon to be
/// built with the matching Cargo feature and fails loudly otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Accel {
    /// Try CUDA; fall back to CPU (with a logged reason) if it does not
    /// initialize.
    #[default]
    Auto,
    /// Force CPU.
    Cpu,
    /// Require the CUDA execution provider (NVIDIA).
    Cuda,
    /// Require the OpenVINO execution provider, targeting the GPU device
    /// (Intel iGPU / Arc; also covers Intel NPU and CPU fallback within OpenVINO).
    Openvino,
    /// Require the WebGPU execution provider — vendor-neutral, backed by
    /// Vulkan/D3D12/Metal. The practical GPU path for AMD on Linux (ROCm is not
    /// wired through the pipeline library).
    Webgpu,
    /// Require the DirectML execution provider (Windows only; AMD, Intel, NVIDIA).
    Directml,
}

impl Accel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Accel::Auto => "auto",
            Accel::Cpu => "cpu",
            Accel::Cuda => "cuda",
            Accel::Openvino => "openvino",
            Accel::Webgpu => "webgpu",
            Accel::Directml => "directml",
        }
    }

    /// Accepted `--accel` spellings, for CLI help.
    pub const CHOICES: &'static str = "auto|cpu|cuda|openvino|webgpu|directml";
}

impl FromStr for Accel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Ok(Accel::Auto),
            "cpu" => Ok(Accel::Cpu),
            "cuda" | "nvidia" => Ok(Accel::Cuda),
            "openvino" | "intel" | "vino" => Ok(Accel::Openvino),
            "webgpu" | "vulkan" | "wgpu" | "amd" => Ok(Accel::Webgpu),
            "directml" | "dml" => Ok(Accel::Directml),
            other => Err(format!(
                "unknown accelerator `{other}` ({})",
                Accel::CHOICES
            )),
        }
    }
}

impl fmt::Display for Accel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single client request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// Encoded image bytes (PNG/JPEG/WebP/BMP), standard base64.
    pub image_b64: String,
    #[serde(default)]
    pub lang: Lang,
    /// Boxes below this recognition confidence are treated as suspect and,
    /// under [`Lang::Auto`], trigger the Korean/Latin retry. `None` uses the
    /// daemon default.
    #[serde(default)]
    pub min_confidence: Option<f32>,
    /// Protocol version guard; bump on breaking changes.
    #[serde(default = "one")]
    pub v: u32,
}

fn one() -> u32 {
    1
}

/// The daemon's reply.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum Response {
    Ok(Ocr),
    Error { message: String },
}

/// A successful recognition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ocr {
    /// Recognized lines in reading order.
    pub lines: Vec<Line>,
    /// `lines` joined by `\n` — the plain-text payload a shortcut pipes to
    /// `wl-copy`.
    pub text: String,
    /// Which recognizer bucket answered (`cjk` / `korean` / `latin` / `mixed`).
    pub lang_used: String,
    /// Execution provider that actually ran (`cpu` / `cuda`).
    pub accel: String,
    pub elapsed_ms: u64,
}

/// One detected + recognized text line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Line {
    pub text: String,
    pub confidence: f32,
    /// Detection polygon, 4 points `[x, y]`, clockwise from the top-left.
    pub bbox: [[f32; 2]; 4],
    /// Recognizer bucket that produced this line.
    pub lang: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lang_aliases_map_to_buckets() {
        assert_eq!("japanese".parse::<Lang>().unwrap(), Lang::Cjk);
        assert_eq!("zh-Hans".parse::<Lang>().unwrap(), Lang::Cjk);
        assert_eq!("pt-BR".parse::<Lang>().unwrap(), Lang::Latin);
        assert_eq!("KOR".parse::<Lang>().unwrap(), Lang::Korean);
        assert_eq!("".parse::<Lang>().unwrap(), Lang::Auto);
        assert!("klingon".parse::<Lang>().is_err());
    }

    #[test]
    fn every_listed_alias_parses() {
        for a in Lang::ALIASES {
            assert!(a.parse::<Lang>().is_ok(), "alias {a} should parse");
        }
    }

    #[test]
    fn accel_aliases_map_to_vendors() {
        assert_eq!("cuda".parse::<Accel>().unwrap(), Accel::Cuda);
        assert_eq!("nvidia".parse::<Accel>().unwrap(), Accel::Cuda);
        assert_eq!("intel".parse::<Accel>().unwrap(), Accel::Openvino);
        assert_eq!("openvino".parse::<Accel>().unwrap(), Accel::Openvino);
        assert_eq!("amd".parse::<Accel>().unwrap(), Accel::Webgpu);
        assert_eq!("vulkan".parse::<Accel>().unwrap(), Accel::Webgpu);
        assert_eq!("dml".parse::<Accel>().unwrap(), Accel::Directml);
        assert_eq!("".parse::<Accel>().unwrap(), Accel::Auto);
        assert!("quantum".parse::<Accel>().is_err());
    }

    #[test]
    fn accel_roundtrips_through_str() {
        for a in [
            Accel::Auto,
            Accel::Cpu,
            Accel::Cuda,
            Accel::Openvino,
            Accel::Webgpu,
            Accel::Directml,
        ] {
            assert_eq!(a.as_str().parse::<Accel>().unwrap(), a);
        }
    }

    #[test]
    fn request_roundtrips() {
        let req = Request {
            image_b64: "aGk=".into(),
            lang: Lang::Latin,
            min_confidence: Some(0.6),
            v: 1,
        };
        let line = serde_json::to_string(&req).unwrap();
        let back: Request = serde_json::from_str(&line).unwrap();
        assert_eq!(back.lang, Lang::Latin);
        assert_eq!(back.min_confidence, Some(0.6));
    }
}
