//! The engine: one shared detector + three resident recognizers, wired to the
//! `oar-ocr` predictor API. Implements the report's section 2.4 flow —
//! *detect once, recognize, then retry only the weak boxes under `Auto`.*

use image::RgbImage;
use oar_ocr::domain::Detection;
use oar_ocr::predictors::{TextDetectionPredictor, TextRecognitionPredictor};
use oar_ocr::processors::BoundingBox;
use oar_ocr::utils::get_rotate_crop_image;
use ocr_proto::Lang;

use crate::accel::{self, Accel, Effective};
use crate::error::{Error, Result};
use crate::lang::{self, Candidate};
use crate::models::{self, Recognizer, DET_MODEL};

/// Tunables for [`Engine::load`].
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub accel: Accel,
    pub device_id: i32,
    /// ONNX Runtime intra-op threads; `None` lets ORT decide.
    pub intra_threads: Option<usize>,
    /// Recognition confidence below which a box is "suspect" and, under
    /// `Auto`, retried with the other recognizers.
    pub min_confidence: f32,
    /// DB detector post-processing knobs (oar-ocr defaults shown).
    pub det_score_threshold: f32,
    pub det_box_threshold: f32,
    pub det_unclip_ratio: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            accel: Accel::Auto,
            device_id: 0,
            intra_threads: None,
            min_confidence: lang::DEFAULT_MIN_CONFIDENCE,
            det_score_threshold: 0.3,
            det_box_threshold: 0.6,
            det_unclip_ratio: 1.5,
        }
    }
}

/// One detected + recognized line.
#[derive(Debug, Clone)]
pub struct TextLine {
    pub text: String,
    pub confidence: f32,
    /// Detection quad, `[top-left, top-right, bottom-right, bottom-left]`.
    pub points: [[f32; 2]; 4],
    pub recognizer: Recognizer,
}

/// Result of [`Engine::recognize`].
#[derive(Debug, Clone)]
pub struct Recognition {
    pub lines: Vec<TextLine>,
    /// `cjk` / `korean` / `latin` / `mixed`.
    pub label: String,
    pub accel: Effective,
}

impl Recognition {
    /// Lines joined by `\n` in reading order.
    pub fn text(&self) -> String {
        self.lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Loaded models, ready to serve. Cheap to keep resident: ~41 MiB of weights
/// (report section 2.4).
pub struct Engine {
    detector: TextDetectionPredictor,
    recognizers: Vec<(Recognizer, TextRecognitionPredictor)>,
    effective: Effective,
    min_confidence: f32,
}

impl Engine {
    /// Resolve the accelerator, then build the detector and all three
    /// recognizers. Models are fetched on first use when built with
    /// `auto-download`.
    pub fn load(cfg: &EngineConfig) -> Result<Engine> {
        let plan = accel::plan(cfg.accel, cfg.device_id)?;
        tracing::info!(
            accel = plan.effective.as_str(),
            note = %plan.note,
            "execution provider selected"
        );
        let sess = plan.session_config(cfg.intra_threads);

        let det_path = models::ensure(DET_MODEL)?;
        let detector = TextDetectionPredictor::builder()
            .score_threshold(cfg.det_score_threshold)
            .box_threshold(cfg.det_box_threshold)
            .unclip_ratio(cfg.det_unclip_ratio)
            .with_ort_config(sess.clone())
            .build(det_path)
            .map_err(Error::pipeline)?;

        let mut recognizers = Vec::with_capacity(Recognizer::ALL.len());
        for r in Recognizer::ALL {
            let model = models::ensure(r.model_name())?;
            let dict = models::ensure(r.dict_name())?;
            let pred = TextRecognitionPredictor::builder()
                .dict_path(dict)
                .with_ort_config(sess.clone())
                .build(model)
                .map_err(Error::pipeline)?;
            recognizers.push((r, pred));
            tracing::info!(recognizer = r.as_str(), "recognizer resident");
        }

        Ok(Engine {
            detector,
            recognizers,
            effective: plan.effective,
            min_confidence: cfg.min_confidence,
        })
    }

    pub fn accel(&self) -> Effective {
        self.effective
    }

    fn rec(&self, want: Recognizer) -> &TextRecognitionPredictor {
        &self
            .recognizers
            .iter()
            .find(|(r, _)| *r == want)
            .expect("all three recognizers are loaded in Engine::load")
            .1
    }

    /// Decode `bytes` (PNG/JPEG/WebP/BMP) and recognize.
    pub fn recognize_bytes(
        &self,
        bytes: &[u8],
        lang: Lang,
        min_confidence: Option<f32>,
    ) -> Result<Recognition> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| Error::ImageDecode(e.to_string()))?
            .to_rgb8();
        self.recognize(&img, lang, min_confidence)
    }

    /// Detect text boxes once, recognize them, and (under [`Lang::Auto`]) retry
    /// the low-confidence boxes with the other recognizers.
    pub fn recognize(
        &self,
        image: &RgbImage,
        lang: Lang,
        min_confidence: Option<f32>,
    ) -> Result<Recognition> {
        let min_confidence = min_confidence.unwrap_or(self.min_confidence);
        let empty = |label_recs: &[Recognizer]| Recognition {
            lines: Vec::new(),
            label: lang::result_label(lang, label_recs),
            accel: self.effective,
        };

        // 1. Detection — one pass, language-agnostic.
        let det = self
            .detector
            .predict(vec![image.clone()])
            .map_err(Error::pipeline)?;
        let boxes: Vec<Detection> = det.detections.into_iter().next().unwrap_or_default();
        if boxes.is_empty() {
            return Ok(empty(&[]));
        }

        // 2. Reading order: group into rows by y, then left-to-right.
        let order = reading_order(&boxes);

        // 3. Crop each box to an upright strip.
        let mut crops: Vec<RgbImage> = Vec::with_capacity(order.len());
        let mut quads: Vec<[[f32; 2]; 4]> = Vec::with_capacity(order.len());
        for &i in &order {
            let pts = quad_points(&boxes[i].bbox);
            match get_rotate_crop_image(image, &pts) {
                Ok(c) => {
                    crops.push(c);
                    quads.push(to_quad_array(&pts));
                }
                Err(e) => {
                    tracing::warn!(box_idx = i, err = %e, "crop failed; skipping box");
                }
            }
        }
        if crops.is_empty() {
            return Ok(empty(&[]));
        }

        // 4. Primary recognizer over every crop.
        let primary_rec = Recognizer::for_lang(lang);
        let primary_out = self
            .rec(primary_rec)
            .predict(crops.clone())
            .map_err(Error::pipeline)?;
        let mut chosen: Vec<Candidate> = primary_out
            .texts
            .into_iter()
            .zip(primary_out.scores)
            .map(|(t, s)| Candidate::new(primary_rec, t, s))
            .collect();

        // 5. Confidence fallback — only for Auto, only the weak boxes.
        if lang == Lang::Auto {
            let weak: Vec<usize> = chosen
                .iter()
                .enumerate()
                .filter(|(_, c)| lang::needs_retry(c, min_confidence))
                .map(|(i, _)| i)
                .collect();

            if !weak.is_empty() {
                let weak_crops: Vec<RgbImage> = weak.iter().map(|&i| crops[i].clone()).collect();
                let mut retries: Vec<Vec<Candidate>> = vec![Vec::new(); weak.len()];
                for r in [Recognizer::Korean, Recognizer::Latin] {
                    let out = self
                        .rec(r)
                        .predict(weak_crops.clone())
                        .map_err(Error::pipeline)?;
                    for (slot, (t, s)) in retries
                        .iter_mut()
                        .zip(out.texts.into_iter().zip(out.scores))
                    {
                        slot.push(Candidate::new(r, t, s));
                    }
                }
                for (wi, &pi) in weak.iter().enumerate() {
                    let winner = lang::pick(&chosen[pi], &retries[wi], min_confidence).clone();
                    chosen[pi] = winner;
                }
            }
        }

        // 6. Assemble, dropping empty reads.
        let mut lines = Vec::with_capacity(chosen.len());
        let mut winner_recs = Vec::with_capacity(chosen.len());
        for (c, q) in chosen.into_iter().zip(quads) {
            if c.text.trim().is_empty() {
                continue;
            }
            winner_recs.push(c.recognizer);
            lines.push(TextLine {
                text: c.text,
                confidence: c.confidence,
                points: q,
                recognizer: c.recognizer,
            });
        }

        Ok(Recognition {
            label: lang::result_label(lang, &winner_recs),
            lines,
            accel: self.effective,
        })
    }
}

/// Download (or locate) every model and dictionary the engine needs, without
/// building sessions. For an `ocrd --warm` / F0 prefetch step.
pub fn prefetch() -> Result<Vec<std::path::PathBuf>> {
    models::all_assets()
        .iter()
        .map(|n| models::ensure(n))
        .collect()
}

/// Sort box indices into reading order: rows top-to-bottom (bucketed by the
/// median box height), then left-to-right within a row.
fn reading_order(boxes: &[Detection]) -> Vec<usize> {
    let heights: Vec<f32> = boxes
        .iter()
        .map(|b| {
            let (_, y0, _, y1) = b.bbox.aabb();
            (y1 - y0).max(1.0)
        })
        .collect();
    let row_h = (median(&heights) * 0.7).max(1.0);

    let mut idx: Vec<usize> = (0..boxes.len()).collect();
    idx.sort_by(|&a, &b| {
        let ca = boxes[a].bbox.center();
        let cb = boxes[b].bbox.center();
        let ra = (ca.y / row_h).round() as i64;
        let rb = (cb.y / row_h).round() as i64;
        ra.cmp(&rb)
            .then(ca.x.partial_cmp(&cb.x).unwrap_or(std::cmp::Ordering::Equal))
    });
    idx
}

/// Four corner points (TL, TR, BR, BL) for any detection polygon.
fn quad_points(bbox: &BoundingBox) -> Vec<oar_ocr::processors::Point> {
    bbox.get_min_area_rect().get_box_points()
}

fn to_quad_array(pts: &[oar_ocr::processors::Point]) -> [[f32; 2]; 4] {
    let mut out = [[0.0f32; 2]; 4];
    for (slot, p) in out.iter_mut().zip(pts.iter()) {
        *slot = [p.x, p.y];
    }
    out
}

fn median(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_basics() {
        assert_eq!(median(&[]), 0.0);
        assert_eq!(median(&[5.0]), 5.0);
        assert_eq!(median(&[1.0, 3.0]), 2.0);
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn default_config_matches_report_threshold() {
        assert_eq!(EngineConfig::default().min_confidence, 0.6);
        assert!(matches!(EngineConfig::default().accel, Accel::Auto));
    }
}
