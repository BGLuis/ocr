# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Local, offline, multilingual OCR for daily desktop use: select a screen region, get its
text on the clipboard. A resident daemon (`ocrd`) keeps PP-OCRv5 models hot; a tiny client
(`ocrd-client`) ships an image over a Unix socket and prints the text. Latin
(pt/es/fr/de), Japanese, Chinese, Korean. No Python at runtime, no cloud.

The design rationale, the rejected alternatives, and the phase plan (F0–F7) live in
`docs/reports/OCR-MULTILINGUE-ANALISE-DE-VIABILIDADE.md` (Portuguese). Code comments cite
its section numbers ("report §2.4"). Currently through F3; F4 (script auto-detect tuning,
box merging) is partial. Commit messages in this repo have been written in Portuguese.

## Commands

```sh
cargo build                       # workspace debug build
cargo build --release             # LTO, stripped
cargo test                        # unit + logic tests, NO models needed
cargo clippy --workspace          # lint (code carries targeted #[allow]s; keep it clean)
cargo fmt

# Run one test
cargo test -p ocr-proto lang_aliases_map_to_buckets
cargo test -p ocr-core --test recognition korean_model_selection_matters

# Recognition tests against the real models. They SKIP when models are absent
# unless OCR_REQUIRE_MODELS=1 forces a hard failure.
cargo run -p ocrd -- --warm                                  # fetch ~41 MiB to ~/.oar (OAR_HOME)
OCR_REQUIRE_MODELS=1 cargo test -p ocr-core --test recognition

# Run the daemon / client from source
cargo run -p ocrd -- --accel auto            # add --once to serve a single connection
cargo run -p ocrd-client -- --file tests/data/en.png --lang auto --json

# GPU builds (opt-in features, forwarded ocrd -> ocr-core; runtime-probed before trusted)
cargo build -p ocrd --features cuda          # NVIDIA: works with prebuilt ORT + cuDNN 9
cargo build -p ocrd --features openvino      # Intel: needs a system ORT (ORT_STRATEGY=system)
cargo build -p ocrd --features directml      # Windows only
# No webgpu feature: ort/webgpu makes the prebuilt binary fail to load.
```

Toolchain: Rust edition 2021, `rust-version = 1.95` (workspace uses newer std APIs such as
`is_multiple_of`; the machine has 1.97).

## Architecture

Four-crate workspace with a strict dependency direction:

```
ocr-proto  ── wire types, serde only. Both ocrd and ocrd-client link just this.
ocr-core   ── the ONLY crate that names `oar-ocr` or `ort`. Engine, models, accel probe,
              language fallback. Every oar-ocr/ort error is wrapped in ocr_core::Error so
              callers never see those crate names.
ocrd       ── loads ocr-core once, serves one request per connection on a Unix socket.
ocrd-client── reads an image, prints the text. Links ocr-proto only, stays ~1 MiB.
```

Why the isolation: `ort` has no stable release and its API changes between rc builds. It is
pinned exactly (`=2.0.0-rc.13`), `oar-ocr` is pinned exactly (`=0.9.2`), and all use is
confined to `ocr-core`. Do not add `ort`/`oar-ocr` calls to any other crate. ONNX Runtime
is statically linked via `ort`'s `download-binaries` feature — installed binaries need no
sibling `.so`.

### The recognition flow (`ocr-core/src/engine.rs`)

One language-agnostic detector + three resident PP-OCRv5 recognizers (CJK, Korean, Latin),
~41 MiB total, all kept in memory. Per request:

1. Detect text boxes once.
2. Sort boxes into reading order (rows bucketed by median height, then left-to-right).
3. Crop each box to an upright strip.
4. Run the primary recognizer over every crop. For a forced `--lang`, that is the only pass.
5. **Under `Lang::Auto` only**: boxes that came back below `min_confidence` (default 0.6) or
   empty are re-run through the Korean and Latin recognizers; `lang::pick` keeps the
   highest-confidence non-empty candidate.
6. Assemble lines, dropping empty reads. `label` is the single winning bucket, or `mixed`.

`ocr-core/src/lang.rs` holds this decision logic as pure functions with no `oar-ocr` types,
so it is unit-tested without models. `ocr-core/src/models.rs` maps recognizer buckets to
registered ONNX model + dict names; `oar-ocr`'s `auto-download` feature fetches them by name
to `$OAR_HOME` (`~/.oar`) on first use. `ocrd --warm` / `ocr_core::prefetch()` pre-fetches.

### Accelerator selection (`ocr-core/src/accel.rs`)

CPU is the baseline. `--accel auto` tries CUDA then CPU and **never** silently picks a
vendor GPU backend. Explicit `--accel cuda|openvino|webgpu|directml` is **fatal** if the
provider is unavailable — no silent CPU fallback (that is the whole point per report §4).
Each provider has an honest probe (build feature present + runtime libraries on disk).
`webgpu` always errors: the prebuilt ORT does not ship `libwebgpu_dawn.so`.

### Wire protocol & socket (`ocr-proto/src/lib.rs`, `ocrd/src/main.rs`)

Newline-delimited JSON, one `Request` and one `Response` per connection. Image bytes travel
base64-encoded inside the request. Serial, single-connection (no concurrency yet).

Socket path: `$XDG_RUNTIME_DIR/ocrd.sock`, else a per-user temp subdir
(`<tmp>/ocrd-<user>/ocrd.sock`) — never a bare file in shared `/tmp`. The daemon enforces
owner-only access: parent dir 0700 and refused if owned by another uid, stale socket removed
only if we own it, socket chmod'd 0600 after bind. `SocketGuard` removes it on drop; there
is no signal-handler cleanup yet (F3 caveat).

`Lang` and `Accel` (in `ocr-proto`) accept many human spellings (`ja`, `zh-Hans`, `pt-BR`,
`nvidia`, `intel`, `amd`, …) that collapse onto the internal buckets — see their `FromStr`.

## Deploy artifacts

`contrib/ocrd.service` (systemd --user unit), `contrib/hyprland.conf.example` (keybinds),
`contrib/ocr` (a no-systemd wrapper that starts `ocrd` on first call). Env overrides:
`OCRD_SOCKET`, `OCRD_ACCEL`, `OCRD_BIN`, `OCRD_CLIENT`, `OAR_HOME`, `OCRD_OPENVINO_DEVICE`.

## Test images

`tests/data/{en,ja_h,zh,ko,pt_diacriticos}.png` — the synthetic images from report §1.4,
used by `ocr-core/tests/recognition.rs` as the acceptance checklist.
