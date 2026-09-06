//! Confidence-driven recognizer fallback for [`Lang::Auto`] (report section 2.4).
//!
//! Pure decision logic, deliberately free of `oar-ocr` types so it can be
//! unit-tested without models: the engine runs the CJK recognizer on every
//! detected box, then for the boxes that came back weak it also runs Korean and
//! Latin and calls [`pick`] to choose the winner.

use ocr_proto::Lang;

use crate::models::Recognizer;

/// Boxes below this recognition confidence are treated as suspect. Matches the
/// report's `< 0.6` "suspeito" band (section 4).
pub const DEFAULT_MIN_CONFIDENCE: f32 = 0.6;

/// One recognizer's answer for one cropped box.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub recognizer: Recognizer,
    pub text: String,
    pub confidence: f32,
}

impl Candidate {
    pub fn new(recognizer: Recognizer, text: impl Into<String>, confidence: f32) -> Self {
        Candidate {
            recognizer,
            text: text.into(),
            confidence,
        }
    }

    fn is_usable(&self) -> bool {
        !self.text.trim().is_empty()
    }
}

/// Should the weak-box retry run for this primary confidence?
///
/// Also retries when the primary produced empty text regardless of the score —
/// a blank read is worse than a low-confidence one.
pub fn needs_retry(primary: &Candidate, min_confidence: f32) -> bool {
    !primary.is_usable() || primary.confidence < min_confidence
}

/// Choose the winning candidate for a box.
///
/// * If the primary is at or above `min_confidence` and non-empty, keep it —
///   the retries should not even have been computed.
/// * Otherwise take the highest-confidence **usable** (non-empty) candidate.
/// * If nothing is usable, fall back to the primary so the box still carries
///   its (low) score downstream.
///
/// Ties prefer the primary, then the earlier retry, for output stability.
pub fn pick<'a>(
    primary: &'a Candidate,
    retries: &'a [Candidate],
    min_confidence: f32,
) -> &'a Candidate {
    if primary.is_usable() && primary.confidence >= min_confidence {
        return primary;
    }

    std::iter::once(primary)
        .chain(retries.iter())
        .filter(|c| c.is_usable())
        .max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(primary)
}

/// Label for the whole result: the single recognizer that answered, or
/// `"mixed"` when winners span more than one bucket.
pub fn result_label(forced: Lang, winners: &[Recognizer]) -> String {
    if forced != Lang::Auto {
        return Recognizer::for_lang(forced).as_str().to_string();
    }
    match winners.first() {
        None => "cjk".to_string(),
        Some(first) if winners.iter().all(|r| r == first) => first.as_str().to_string(),
        _ => "mixed".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(r: Recognizer, t: &str, conf: f32) -> Candidate {
        Candidate::new(r, t, conf)
    }

    #[test]
    fn confident_primary_wins_without_retry() {
        let primary = c(Recognizer::Cjk, "吾輩は猫である", 0.94);
        assert!(!needs_retry(&primary, DEFAULT_MIN_CONFIDENCE));
        assert_eq!(pick(&primary, &[], DEFAULT_MIN_CONFIDENCE).text, "吾輩は猫である");
    }

    #[test]
    fn weak_primary_yields_to_stronger_retry() {
        let primary = c(Recognizer::Cjk, "CHASIMR", 0.30);
        let retries = [
            c(Recognizer::Korean, "안녕하세요 세계", 0.91),
            c(Recognizer::Latin, "annyeong", 0.40),
        ];
        assert!(needs_retry(&primary, DEFAULT_MIN_CONFIDENCE));
        let w = pick(&primary, &retries, DEFAULT_MIN_CONFIDENCE);
        assert_eq!(w.recognizer, Recognizer::Korean);
        assert_eq!(w.text, "안녕하세요 세계");
    }

    #[test]
    fn empty_primary_triggers_retry_even_if_score_high() {
        let primary = c(Recognizer::Cjk, "   ", 0.99);
        assert!(needs_retry(&primary, DEFAULT_MIN_CONFIDENCE));
        let retries = [c(Recognizer::Latin, "ação", 0.72)];
        assert_eq!(pick(&primary, &retries, DEFAULT_MIN_CONFIDENCE).text, "ação");
    }

    #[test]
    fn all_empty_falls_back_to_primary() {
        let primary = c(Recognizer::Cjk, "", 0.10);
        let retries = [c(Recognizer::Korean, "", 0.05)];
        let w = pick(&primary, &retries, DEFAULT_MIN_CONFIDENCE);
        assert_eq!(w.recognizer, Recognizer::Cjk);
    }

    #[test]
    fn label_is_mixed_only_when_buckets_differ() {
        assert_eq!(result_label(Lang::Auto, &[Recognizer::Cjk, Recognizer::Cjk]), "cjk");
        assert_eq!(
            result_label(Lang::Auto, &[Recognizer::Cjk, Recognizer::Latin]),
            "mixed"
        );
        assert_eq!(result_label(Lang::Korean, &[Recognizer::Cjk]), "korean");
    }
}
