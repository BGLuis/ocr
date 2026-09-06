# ocr — multilingual OCR on a keystroke

Local, offline OCR for daily use: select a region, get its text on the clipboard.
Latin (Portuguese/Spanish/French/German), Japanese, Chinese and Korean, from one
resident daemon. No Python at runtime, no cloud.

Design rationale and the alternatives that were rejected:
[`docs/reports/OCR-MULTILINGUE-ANALISE-DE-VIABILIDADE.md`](docs/reports/OCR-MULTILINGUE-ANALISE-DE-VIABILIDADE.md).

## Status

| Phase | What | State |
|---|---|---|
| F0 | Spike: fetch models, run on the four test images, compare to Tesseract | ✅ done — see *Verification* below |
| F1 | `ocr-core`: `recognize(image, lang) -> Vec<TextLine>`, 1 detector + 3 recognizers, confidence fallback, `--accel` | ✅ done |
| F2 | `ocrd-client`: image on stdin → text on stdout; Hyprland bind | ✅ done |
| F3 | `ocrd`: Unix socket, resident models, systemd unit | ✅ done (single-connection serial; no signal cleanup yet) |
| F4 | Script auto-detection tuning, box merging, forced-language shortcuts | ⬜ partial (forced language works; box merging pending) |
| F5 | `manga-ocr` autoregressive decode | ⬜ not started |
| F6 | Video: ffmpeg → SSIM dedup → OCR → `.srt` | ⬜ not started |
| F7 | GUI | ⬜ not started |

## Architecture

```
grim -g "$(slurp)" -  │  ocrd-client  ──JSON over $XDG_RUNTIME_DIR/ocrd.sock──▶  ocrd
    (PNG on stdin)     │  (tiny, serde  │                                          │
                       │   only)        │                       ┌──────────────────┴─────────────┐
                       │                │                       │ ocr-core (isolates oar-ocr/ort)│
   text on stdout  ◀───┤  ◀──JSON───────┘                       │  pp-ocrv5_mobile_det (shared)  │
    │                                                           │  ├ cjk   rec  (ja/zh/en)        │
   wl-copy                                                      │  ├ korean rec                  │
                                                                │  └ latin rec (pt/es/fr/de)     │
                                                                └────────────────────────────────┘
```

- **`ocr-proto`** — the wire types. Depends on `serde` only, so the client stays small.
- **`ocr-core`** — the only crate that names `oar-ocr` or `ort` (report §4). One
  language-agnostic detector, three PP-OCRv5 recognizers, all resident (~41 MiB).
  Under `Lang::Auto` it runs the CJK recognizer on every box, then re-runs only
  the low-confidence boxes through Korean and Latin and keeps the best (report §2.4).
- **`ocrd`** — loads `ocr-core` once, answers one request per connection.
- **`ocrd-client`** — reads an image, prints the text.

## Install

```sh
cargo install --path crates/ocrd          # -> ~/.cargo/bin/ocrd        (self-contained, ~29 MiB)
cargo install --path crates/ocrd-client   # -> ~/.cargo/bin/ocrd-client (~1 MiB)
install -Dm755 contrib/ocr ~/.local/bin/ocr   # optional auto-start wrapper (see "Use in scripts")
ocrd --warm                               # fetch models to ~/.oar (~41 MiB, one time)
```

ONNX Runtime is statically linked via the `ort` `download-binaries` feature at
build time — the installed binaries need no sibling `.so` and run from anywhere.
CPU is the default and needs nothing else.

### GPU (optional)

`--accel` accepts `auto | cpu | cuda | openvino | webgpu | directml`. `auto`
tries CUDA then CPU and **never** silently picks a vendor backend; the rest
*require* their provider and fail loudly if it is missing (report §4). Every one
is probed before it is trusted.

For a single screenshot the CPU path is already fast enough (~30–130 ms warm
here); GPU only pays off for batch flows (video, PDF — F6). So none of this is on
the default path.

| Vendor | Flag | Build | Reality with the prebuilt ONNX Runtime |
|---|---|---|---|
| **NVIDIA** | `--accel cuda` | `--features cuda` | Works, given cuDNN 9 + `libcudart.so.12` on the host. This machine has CUDA 13.x and no cuDNN, so the probe refuses and `auto` stays on CPU until `pacman -S cudnn` (and a CUDA-12 runtime). |
| **Intel** iGPU / Arc / NPU | `--accel openvino` | `--features openvino` | **Does not link** with the downloaded ORT — pyke's prebuilt binaries omit OpenVINO. Needs a system ONNX Runtime built with it: `ORT_STRATEGY=system ORT_LIB_LOCATION=/path/to/onnxruntime cargo build --features openvino`. Then `--accel openvino` targets `GPU` (override with `OCRD_OPENVINO_DEVICE`). |
| **AMD** on Linux | `--accel webgpu` | *(no feature flag)* | ROCm is not exposed by the pipeline library, and `ort/webgpu` makes the binary depend on `libwebgpu_dawn.so`, which the prebuilt ORT does not ship — enabling it breaks the daemon at load. Needs ONNX Runtime built from source with WebGPU/Dawn, linked via `ORT_STRATEGY=system`. Until then `--accel webgpu` returns a clean fatal error. |
| **Windows**, any GPU | `--accel directml` | `--features directml` | Same as OpenVINO: needs a system ORT with the DirectML EP. Windows-only; on Linux the probe rejects it. |

See <https://ort.pyke.io/setup/linking> for building/linking a system ONNX
Runtime. The daemon and client code already speak every one of these flags — the
only gap is the ORT binary.

## Run

```sh
# terminal / systemd
ocrd --accel auto        # auto|cpu|cuda|openvino|webgpu|directml

# one-shot
grim -g "$(slurp)" - | ocrd-client --lang auto | wl-copy
ocrd-client --file shot.png --lang latin --json
```

Install the daemon and the Hyprland binds:

```sh
cp contrib/ocrd.service ~/.config/systemd/user/ && systemctl --user enable --now ocrd
cat contrib/hyprland.conf.example >> ~/.config/hypr/hyprland.conf
```

## Use in scripts

Two ways to get `ocr` on your PATH for other projects:

**A. systemd user service (recommended for daily use).** The daemon stays warm;
scripts just call `ocrd-client`:

```sh
cp contrib/ocrd.service ~/.config/systemd/user/
systemctl --user enable --now ocrd

# then, anywhere:
ocrd-client --file page.png --lang latin
grim -g "$(slurp)" - | ocrd-client --lang auto | wl-copy
```

**B. the `ocr` wrapper (no systemd needed).** `contrib/ocr` starts `ocrd` on the
first call and reuses it after — good for portable scripts and non-Hyprland
machines:

```sh
install -Dm755 contrib/ocr ~/.local/bin/ocr

ocr --file screenshot.png                       # text on stdout
ocr --file invoice.png --json | jq -r '.lines[].text'
xclip -selection clipboard -t image/png -o | ocr --lang latin
```

Overridable via env: `OCRD_SOCKET`, `OCRD_ACCEL` (`auto|cpu|cuda|openvino|webgpu|directml`), `OCRD_BIN`,
`OCRD_CLIENT`.

`ocrd-client --ping` exits 0 iff the daemon is reachable — useful for your own
health checks. First call after a boot pays the ~150–200 ms model load once;
every call after that is ~30–130 ms warm.

### `ocrd-client` options

| Flag | Meaning |
|---|---|
| `--lang` | `auto` (default), `cjk` / `ja` / `zh` / `en`, `korean` / `ko`, `latin` / `pt` / `es` / `fr` / `de` |
| `--file PATH` | read image from a file instead of stdin |
| `--json` | full response (lines, per-line confidence, bboxes, timing) |
| `--verbose` | per-line confidence to stderr, text to stdout |
| `--socket PATH` | override `$XDG_RUNTIME_DIR/ocrd.sock` |
| `--ping` | exit 0 if ocrd is reachable, non-zero otherwise; reads no image |

## Verification

Unit + logic tests (no models needed):

```sh
cargo test
```

Recognition tests against the real models (report §5 checklist). They skip if the
models are absent unless `OCR_REQUIRE_MODELS=1`:

```sh
cargo run -p ocrd -- --warm
OCR_REQUIRE_MODELS=1 cargo test -p ocr-core --test recognition
```

F0 measured on this machine (CPU, warm daemon), against the Tesseract baseline in
report §1.4:

| Image | Input | `ocrd --lang auto` | confidence | Tesseract (`-l eng`) |
|---|---|---|---|---|
| `en.png` | `The quick brown fox 2026` | `The quick brown fox 2026` | 96 % | `‘The quick brown fox 2026` |
| `ja_h.png` | `吾輩は猫である。名前はまだ無い。` | `吾輩は猫である。` / `名前はまだ無い。` | 99 % | `EEIIETH D, BEITEIEEL,` |
| `zh.png` | `人工智能技术发展` | `人工智能技术发展` | 99 % | `ANIEERALE` |
| `ko.png` | `안녕하세요 세계` | `안녕하세요 세계` (korean, via fallback) | 98 % | `CHASIMR MIA` |
| `pt_diacriticos.png` | `ação, você, pêssego, coração` | intact with `--lang latin` | 98 % | — |

Known rough edges (F4):
- `ja_h.png` is split into two boxes at the `。` plus a spurious separator glyph —
  correct text, wrong line count. Needs same-row box merging.
- `--lang auto` on Portuguese can pick the CJK recognizer at ~87 % confidence and
  mangle diacritics (`ǎ` for `ã`). Above the 0.6 threshold, so no fallback fires.
  Force `--lang latin`, or add a script heuristic. This is report Risk #4.

## Layout

```
crates/ocr-proto     wire types (serde only)
crates/ocr-core      oar-ocr/ort boundary: engine, models, accel probe, lang fallback
crates/ocrd          daemon
crates/ocrd-client   client
tests/data           the five synthetic images from report §1.4 / §7
contrib              systemd unit, Hyprland binds
```
