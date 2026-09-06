//! Error type for `ocr-core`. Every `oar-ocr` / `ort` error is wrapped here so
//! callers never name those crates directly (report section 4: keep the
//! `ort`-rc churn behind one boundary).

use std::path::PathBuf;

/// Result alias used throughout the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `--accel cuda` was requested but the CUDA execution provider could not
    /// be initialized. Deliberately fatal (report section 2.9).
    #[error("CUDA was required but is not available: {0}")]
    CudaRequired(String),

    /// A model or dictionary file could not be resolved or downloaded.
    #[error("model asset `{name}` could not be obtained: {source}")]
    ModelAsset {
        name: String,
        #[source]
        source: anyhow::Error,
    },

    /// The input bytes were not a decodable image.
    #[error("could not decode input image: {0}")]
    ImageDecode(String),

    /// A cache/working directory could not be created.
    #[error("could not create directory `{path}`: {source}")]
    Dir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Detection or recognition failed inside `oar-ocr`.
    #[error("OCR pipeline error: {0}")]
    Pipeline(String),
}

impl Error {
    pub(crate) fn pipeline(e: impl std::fmt::Display) -> Self {
        Error::Pipeline(e.to_string())
    }
}
