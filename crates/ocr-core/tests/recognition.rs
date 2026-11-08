//! Integration tests for the report's section-5 "automatable" checklist.
//!
//! They need the PP-OCRv5 models. When those are not present (offline CI, first
//! checkout) each test skips with a message instead of failing — unless
//! `OCR_REQUIRE_MODELS=1` is set, which turns a load failure into a hard error.
//!
//! Fetch the models once with:  `cargo run -p ocrd -- --warm`

use ocr_core::{Accel, Engine, EngineConfig, Lang};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data");

fn engine() -> Option<Engine> {
    let cfg = EngineConfig {
        accel: Accel::Cpu,
        ..EngineConfig::default()
    };
    match Engine::load(&cfg) {
        Ok(e) => Some(e),
        Err(e) => {
            if std::env::var_os("OCR_REQUIRE_MODELS").is_some() {
                panic!("OCR_REQUIRE_MODELS is set but the engine failed to load: {e}");
            }
            eprintln!("skipping: engine unavailable ({e}). Run `cargo run -p ocrd -- --warm`.");
            None
        }
    }
}

fn read(name: &str) -> Vec<u8> {
    std::fs::read(format!("{DATA}/{name}")).unwrap_or_else(|e| panic!("reading {name}: {e}"))
}

fn joined(engine: &Engine, name: &str, lang: Lang) -> String {
    engine
        .recognize_bytes(&read(name), lang, None)
        .unwrap_or_else(|e| panic!("recognize {name}: {e}"))
        .text()
}

/// The premise of the whole report: swapping the engine fixes CJK. The same
/// `ja_h.png` returned pure noise from Tesseract (report 1.4).
#[test]
fn japanese_horizontal_is_read_correctly() {
    let Some(engine) = engine() else { return };
    let out = engine
        .recognize_bytes(&read("ja_h.png"), Lang::Cjk, None)
        .unwrap_or_else(|e| panic!("recognize ja_h.png: {e}"));
    assert_eq!(out.text().trim(), "吾輩は猫である。名前はまだ無い。");
    assert_eq!(out.lines.len(), 2);
}

#[test]
fn japanese_horizontal_auto_has_clean_lines_and_cjk_label() {
    let Some(engine) = engine() else { return };
    let out = engine
        .recognize_bytes(&read("ja_h.png"), Lang::Auto, None)
        .unwrap_or_else(|e| panic!("recognize ja_h.png: {e}"));
    assert_eq!(out.label, "cjk");
    let line_texts: Vec<&str> = out.lines.iter().map(|l| l.text.trim()).collect();
    assert_eq!(line_texts, vec!["吾輩は猫である。", "名前はまだ無い。"]);
}

#[test]
fn chinese_is_read_correctly() {
    let Some(engine) = engine() else { return };
    let text = joined(&engine, "zh.png", Lang::Cjk);
    assert!(text.contains("人工智能技术发展"), "got: {text:?}");
}

/// Model selection has a real effect: the Korean recognizer reads `ko.png`,
/// the CJK one does not produce the same string.
#[test]
fn korean_model_selection_matters() {
    let Some(engine) = engine() else { return };
    let with_korean = joined(&engine, "ko.png", Lang::Korean);
    assert!(
        with_korean.contains("안녕하세요"),
        "korean model: {with_korean:?}"
    );

    let with_cjk = joined(&engine, "ko.png", Lang::Cjk);
    assert_ne!(
        with_korean.trim(),
        with_cjk.trim(),
        "korean and cjk recognizers must not agree on ko.png"
    );
}

/// The domestic case, easiest to forget: Portuguese diacritics survive the
/// Latin recognizer intact.
#[test]
fn portuguese_diacritics_survive_latin() {
    let Some(engine) = engine() else { return };
    let text = joined(&engine, "pt_diacriticos.png", Lang::Latin);
    for word in ["ação", "você", "pêssego", "coração"] {
        assert!(text.contains(word), "missing {word:?} in {text:?}");
    }
}

#[test]
fn portuguese_diacritics_survive_auto() {
    let Some(engine) = engine() else { return };
    let text = joined(&engine, "pt_diacriticos.png", Lang::Auto);
    for word in ["ação", "você", "pêssego", "coração"] {
        assert!(text.contains(word), "missing {word:?} in {text:?}");
    }
}

/// A pure-noise image must come back with low confidence so the `Auto`
/// fallback has a signal to act on (report 2.4, 5).
#[test]
fn noise_image_has_low_confidence() {
    let Some(engine) = engine() else { return };
    // 64x64 of random-ish bytes, encoded as PNG on the fly.
    let mut img = image::RgbImage::new(96, 48);
    for (i, p) in img.pixels_mut().enumerate() {
        let v = ((i * 2654435761usize) >> 13) as u8;
        *p = image::Rgb([v, v ^ 0x5a, v.wrapping_mul(3)]);
    }
    let out = engine
        .recognize(&img, Lang::Auto, None)
        .expect("recognize noise");
    assert!(
        out.lines.iter().all(|l| l.confidence < 0.9),
        "noise should not yield a high-confidence line: {:?}",
        out.lines
    );
}

#[test]
fn min_confidence_triggers_retry() {
    let Some(engine) = engine() else { return };
    let out = engine
        .recognize_bytes(&read("pt_diacriticos.png"), Lang::Auto, Some(0.9))
        .unwrap_or_else(|e| panic!("recognize pt_diacriticos: {e}"));
    let text = out.text();
    for word in ["ação", "coração"] {
        assert!(text.contains(word), "missing {word:?} in {text:?}");
    }
}

#[test]
fn english_beats_the_tesseract_floor() {
    let Some(engine) = engine() else { return };
    // Tesseract inserted a spurious leading quote here (report 1.4); PP-OCRv5
    // should return the string clean.
    let text = joined(&engine, "en.png", Lang::Auto);
    assert_eq!(text.trim(), "The quick brown fox 2026", "got: {text:?}");
}
