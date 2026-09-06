<div align="center">

<!-- GitHub status badges -->

![GitHub Stars](https://www.shieldcn.dev/github/stars/bgluis/ocr.svg?variant=secondary&size=sm)
![GitHub Forks](https://www.shieldcn.dev/github/forks/bgluis/ocr.svg?variant=secondary&size=sm)
![Watchers](https://www.shieldcn.dev/github/watchers/bgluis/ocr.svg?variant=secondary&size=sm)
![Contributors](https://www.shieldcn.dev/github/contributors/bgluis/ocr.svg?theme=emerald&size=sm)
![License](https://www.shieldcn.dev/github/license/bgluis/ocr.svg?variant=ghost&size=sm)

<br/>

<!-- Technology badges -->

![Rust](https://www.shieldcn.dev/badge/Rust-1.95%2B-000000.svg?logo=rust&variant=branded&size=sm)
![ONNX Runtime](https://www.shieldcn.dev/badge/ONNX_Runtime-static-005CED.svg?logo=onnx&variant=branded&size=sm)
![Linux](https://www.shieldcn.dev/badge/Linux-Wayland-FCC624.svg?logo=linux&variant=branded&size=sm)

  <h3>ocr</h3>
  Local, offline, multilingual OCR: select a screen region and its text lands on your clipboard.
</div>

<p align="center">
  <a href="README.md">Português (Brasil)</a> &nbsp;·&nbsp; <b>English</b>
</p>

# 📖 About

`ocr` is an optical-character-recognition tool for daily use on the Linux/Wayland
desktop. A **resident daemon** (`ocrd`) keeps the models hot in memory and a
**tiny client** (`ocrd-client`) ships an image over a Unix socket and prints the
recognized text. The core use case is a keybind: select a screen region, the
text lands on your clipboard.

Under the hood:

- **Pure Rust at runtime** — no Python, no network calls. It is a Cargo
  workspace of four crates: `ocr-proto` (wire types), `ocr-core` (the only
  boundary that touches `oar-ocr`/`ort`), `ocrd` (daemon) and `ocrd-client`
  (client).
- **PP-OCRv5 models** served through
  [`oar-ocr`](https://crates.io/crates/oar-ocr) on the
  [ONNX Runtime](https://onnxruntime.ai/), which is **statically linked** — the
  installed binaries run from anywhere with no sibling `.so`.
- **One detector + three recognizers** kept resident (~41 MiB): CJK (Japanese,
  Chinese, English), Korean, and Latin (Portuguese, Spanish, French, German).
- **Automatic language with a confidence fallback** — under `--lang auto` the
  CJK recognizer runs on every box; low-confidence boxes (< 0.6) are re-tried
  through the Korean and Latin recognizers and the best result wins.
- **CPU by default.** GPU acceleration is an explicit opt-in via
  `--accel auto|cpu|cuda|openvino|webgpu|directml`, always probed before use and
  fatal when unavailable (no silent fallback).
- **Wayland integration**: `grim` + `slurp` + `wl-clipboard`, Hyprland binds and
  a `systemd --user` unit ready in `contrib/`.

The design rationale and the rejected alternatives are in
[`docs/reports/OCR-MULTILINGUE-ANALISE-DE-VIABILIDADE.md`](docs/reports/OCR-MULTILINGUE-ANALISE-DE-VIABILIDADE.md)
(Portuguese).

# 📋 Motivation

I built this to clear a personal blocker: not being able to have — or make — a
local OCR that was actually any good. And, alongside that, to rebuild a feature I
liked and used a lot in PowerToys, now on my own Linux/Wayland setup.

# 💻 Getting started

### Requirements

- [Rust](https://www.rust-lang.org/tools/install) 1.95+ (edition 2021; the
  reference machine runs 1.97).
- Internet access **on the first build** (`ort` downloads the ONNX Runtime) and
  on the first `ocrd --warm` (~41 MiB of PP-OCRv5 models, written to `~/.oar`).
- For the "screenshot → OCR → clipboard" flow on Wayland:
  [grim](https://sr.ht/~emersion/grim/),
  [slurp](https://github.com/emersion/slurp),
  [wl-clipboard](https://github.com/bugaevc/wl-clipboard).
- Optional — GPU: NVIDIA with [cuDNN 9](https://developer.nvidia.com/cudnn) for
  `--features cuda`. Intel (`openvino`) and Windows (`directml`) need a system
  ONNX Runtime — see the acceleration table at the end of this section.

### Installation

1. Clone the project repository:

```sh
git clone https://github.com/bgluis/ocr.git
```

2. Move into the project directory:

```sh
cd ocr
```

3. Pick **one** of the methods below.

#### Method 1 — `cargo install` (recommended)

Installs both binaries into `~/.cargo/bin` and fetches the models once:

```sh
cargo install --path crates/ocrd          # -> ~/.cargo/bin/ocrd        (~29 MiB)
cargo install --path crates/ocrd-client   # -> ~/.cargo/bin/ocrd-client (~1 MiB)
ocrd --warm                               # downloads the models to ~/.oar

# optional: a wrapper that starts ocrd on demand, no systemd
install -Dm755 contrib/ocr ~/.local/bin/ocr
```

#### Method 2 — build from source

```sh
cargo build --release                     # LTO, binaries in ./target/release
cargo run -p ocrd -- --warm               # downloads the models
./target/release/ocrd --accel auto        # start the daemon
```

For GPU, add the matching feature when building `ocrd`:
`cargo build --release -p ocrd --features cuda` (or `openvino` / `directml`).

### Usage

```sh
# 1. Keep the daemon running (terminal or systemd)
ocrd --accel auto

# 2. Anywhere: crop the screen, recognize, copy
grim -g "$(slurp)" - | ocrd-client --lang auto | wl-copy

# Or from a file, with the full JSON response
ocrd-client --file shot.png --lang latin --json
```

Install the service and the Hyprland binds:

```sh
cp contrib/ocrd.service ~/.config/systemd/user/ && systemctl --user enable --now ocrd
cat contrib/hyprland.conf.example >> ~/.config/hypr/hyprland.conf
```

Main `ocrd-client` options:

| Flag          | Meaning                                                                                            |
| ------------- | -------------------------------------------------------------------------------------------------- |
| `--lang`      | `auto` (default), `cjk` / `ja` / `zh` / `en`, `korean` / `ko`, `latin` / `pt` / `es` / `fr` / `de` |
| `--file PATH` | read the image from a file instead of stdin                                                        |
| `--json`      | full response (lines, per-line confidence, bboxes, timing)                                         |
| `--verbose`   | per-line confidence to stderr, text to stdout                                                      |
| `--ping`      | exit 0 if `ocrd` is reachable; reads no image                                                      |

### Environment configuration

The project has no `.env`; behavior is tuned through these variables:

| Variable                   | Default                      | Purpose                                                                      |
| -------------------------- | ---------------------------- | ---------------------------------------------------------------------------- |
| `OCRD_SOCKET`              | `$XDG_RUNTIME_DIR/ocrd.sock` | Unix socket path                                                             |
| `OCRD_ACCEL`               | `auto`                       | `auto\|cpu\|cuda\|openvino\|webgpu\|directml`                                |
| `OAR_HOME`                 | `~/.oar`                     | ONNX model cache                                                             |
| `OCRD_BIN` / `OCRD_CLIENT` | found on `PATH`              | Binaries used by the `contrib/ocr` wrapper                                   |
| `OCRD_OPENVINO_DEVICE`     | `GPU`                        | Target OpenVINO device                                                       |
| `RUST_LOG`                 | `ocrd=info,ocr_core=info`    | Log verbosity                                                                |
| `OCR_REQUIRE_MODELS`       | —                            | If set, recognition tests fail (instead of skipping) when models are missing |

### Tests

```sh
cargo test                                        # unit + logic, no models

cargo run -p ocrd -- --warm                       # download the models
OCR_REQUIRE_MODELS=1 cargo test -p ocr-core --test recognition
```

### GPU acceleration

| Vendor                       | Flag               | Build                 | Reality with the prebuilt ONNX Runtime                                                                          |
| ---------------------------- | ------------------ | --------------------- | --------------------------------------------------------------------------------------------------------------- |
| **NVIDIA**                   | `--accel cuda`     | `--features cuda`     | Works, given cuDNN 9 + `libcudart.so.12` on the host.                                                           |
| **Intel** (iGPU / Arc / NPU) | `--accel openvino` | `--features openvino` | Does not link with the downloaded ORT; needs a system ONNX Runtime built with OpenVINO (`ORT_STRATEGY=system`). |
| **AMD** on Linux             | `--accel webgpu`   | _(no flag)_           | Needs an ONNX Runtime built from source with WebGPU/Dawn; until then, a clean fatal error.                      |
| **Windows**, any GPU         | `--accel directml` | `--features directml` | Like OpenVINO: needs a system ORT with the DirectML EP. Windows only.                                           |

`--accel auto` only ever tries CUDA and falls back to CPU (with the reason
logged); it never silently picks a vendor backend.

# 🤝 Contributors

 <a href="https://github.com/bgluis/ocr/graphs/contributors">
   <img src="https://contrib.rocks/image?repo=bgluis/ocr"/>
 </a>
