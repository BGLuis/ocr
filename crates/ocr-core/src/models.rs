//! The model set (report section 2.4): one shared detector, three PP-OCRv5
//! recognizer buckets. Names are the ones registered in
//! `oar_ocr_core::core::download::registry` — with the `auto-download` feature
//! the oar-ocr builders accept them directly and fetch to `$OAR_HOME` on first
//! use.

use std::path::PathBuf;

use ocr_proto::Lang;

use crate::error::{Error, Result};

/// Language-agnostic text detector, shared by every recognizer (4.6 MiB).
pub const DET_MODEL: &str = "pp-ocrv5_mobile_det.onnx";

/// A recognizer bucket: the concrete PP-OCRv5 model plus its character
/// dictionary (`dict.txt` is the real spec of what the model can emit —
/// report section 2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Recognizer {
    /// Simplified/traditional Chinese, Japanese, English — 15.8 MiB.
    Cjk,
    /// Korean — 12.8 MiB.
    Korean,
    /// Latin scripts with diacritics — 7.7 MiB.
    Latin,
}

impl Recognizer {
    pub const ALL: [Recognizer; 3] = [Recognizer::Cjk, Recognizer::Korean, Recognizer::Latin];

    pub fn as_str(&self) -> &'static str {
        match self {
            Recognizer::Cjk => "cjk",
            Recognizer::Korean => "korean",
            Recognizer::Latin => "latin",
        }
    }

    /// Registered ONNX model name.
    pub fn model_name(&self) -> &'static str {
        match self {
            Recognizer::Cjk => "pp-ocrv5_mobile_rec.onnx",
            Recognizer::Korean => "korean_pp-ocrv5_mobile_rec.onnx",
            Recognizer::Latin => "latin_pp-ocrv5_mobile_rec.onnx",
        }
    }

    /// Registered character-dictionary name.
    pub fn dict_name(&self) -> &'static str {
        match self {
            Recognizer::Cjk => "ppocrv5_dict.txt",
            Recognizer::Korean => "ppocrv5_korean_dict.txt",
            Recognizer::Latin => "ppocrv5_latin_dict.txt",
        }
    }

    /// The recognizer a forced [`Lang`] maps to. `Auto` has no single
    /// recognizer; it starts from [`Recognizer::Cjk`] and retries (see
    /// [`crate::lang`]).
    pub fn for_lang(lang: Lang) -> Recognizer {
        match lang {
            Lang::Auto | Lang::Cjk => Recognizer::Cjk,
            Lang::Korean => Recognizer::Korean,
            Lang::Latin => Recognizer::Latin,
        }
    }
}

/// Every asset the engine needs, for a `warm` / prefetch step.
pub fn all_assets() -> Vec<&'static str> {
    let mut v = vec![DET_MODEL];
    for r in Recognizer::ALL {
        v.push(r.model_name());
        v.push(r.dict_name());
    }
    v
}

/// Resolve a registered asset name to a local path, downloading it if needed.
///
/// With `auto-download` off this only succeeds for names that are already real
/// paths — the daemon then relies on the builders' own lookup and this is used
/// just for nicer error messages.
pub fn ensure(name: &str) -> Result<PathBuf> {
    #[cfg(feature = "auto-download")]
    {
        oar_ocr::download::resolve_path(name).map_err(|e| Error::ModelAsset {
            name: name.to_string(),
            source: anyhow::anyhow!(e.to_string()),
        })
    }
    #[cfg(not(feature = "auto-download"))]
    {
        let p = PathBuf::from(name);
        if p.is_file() {
            Ok(p)
        } else {
            Err(Error::ModelAsset {
                name: name.to_string(),
                source: anyhow::anyhow!(
                    "built without `auto-download`; pass an existing file path"
                ),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_have_distinct_models_and_dicts() {
        let mut models: Vec<_> = Recognizer::ALL.iter().map(|r| r.model_name()).collect();
        models.sort();
        models.dedup();
        assert_eq!(models.len(), 3);

        let mut dicts: Vec<_> = Recognizer::ALL.iter().map(|r| r.dict_name()).collect();
        dicts.sort();
        dicts.dedup();
        assert_eq!(dicts.len(), 3);
    }

    #[test]
    fn asset_list_is_det_plus_three_pairs() {
        assert_eq!(all_assets().len(), 1 + 3 * 2);
        assert_eq!(all_assets()[0], DET_MODEL);
    }

    #[test]
    fn forced_lang_maps_to_expected_bucket() {
        assert_eq!(Recognizer::for_lang(Lang::Korean), Recognizer::Korean);
        assert_eq!(Recognizer::for_lang(Lang::Latin), Recognizer::Latin);
        assert_eq!(Recognizer::for_lang(Lang::Cjk), Recognizer::Cjk);
        assert_eq!(Recognizer::for_lang(Lang::Auto), Recognizer::Cjk);
    }
}
