//! `ocr-core` — the isolation boundary around `oar-ocr` and `ort`
//! (report sections 2.1–2.4, 4).
//!
//! Nothing outside this crate names `oar-ocr`, `ort`, or an ONNX detail. The
//! daemon builds an [`Engine`] once, keeps it resident, and calls
//! [`Engine::recognize`] / [`Engine::recognize_bytes`] per request.
//!
//! ```no_run
//! use ocr_core::{Engine, EngineConfig};
//! use ocr_proto::Lang;
//!
//! # fn main() -> anyhow::Result<()> {
//! let engine = Engine::load(&EngineConfig::default())?;
//! let png = std::fs::read("shot.png")?;
//! let out = engine.recognize_bytes(&png, Lang::Auto)?;
//! println!("{}", out.text());
//! # Ok(())
//! # }
//! ```

mod accel;
mod engine;
mod error;
mod lang;
mod models;

pub use accel::{plan as accel_plan, probe as accel_probe, Accel, Effective, Plan};
pub use engine::{prefetch, Engine, EngineConfig, Recognition, TextLine};
pub use error::{Error, Result};
pub use lang::DEFAULT_MIN_CONFIDENCE;
pub use models::{Recognizer, DET_MODEL};

// Re-exported so callers get the exact enum the engine expects without adding
// an `ocr-proto` dependency of their own.
pub use ocr_proto::Lang;
