//! `ocrd` — the resident half (report sections 2.7, F3).
//!
//! Loads the detector and all three recognizers once, then serves one request
//! per connection on a Unix socket. Keeping the models hot is the whole point:
//! for a single screenshot the cost is dominated by startup, not inference
//! (report section 2.7).

use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use base64::prelude::{Engine as _, BASE64_STANDARD};
use clap::Parser;
use ocr_core::{Engine, EngineConfig};
use ocr_proto::{default_socket_path, Accel, Line, Ocr, Request, Response};

/// Resident OCR daemon.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Accelerator. auto = try CUDA then CPU. cpu = force. The rest require that
    /// provider (fatal if absent, no fallback) and the matching build feature:
    /// cuda (NVIDIA), openvino (Intel GPU), webgpu (AMD on Linux), directml (Windows).
    #[arg(long, default_value = "auto")]
    accel: String,

    /// GPU device index, for cuda / directml.
    #[arg(long, default_value_t = 0)]
    device_id: i32,

    /// ONNX Runtime intra-op threads. 0 = let ORT decide.
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Boxes below this recognition confidence are "suspect" and, under
    /// Lang::Auto, retried with the other recognizers.
    #[arg(long, default_value_t = ocr_core::DEFAULT_MIN_CONFIDENCE)]
    min_confidence: f32,

    /// Unix socket to bind. Defaults to $XDG_RUNTIME_DIR/ocrd.sock.
    #[arg(long)]
    socket: Option<PathBuf>,

    /// Download/locate every model and dictionary, print the paths, and exit
    /// (F0 prefetch — no socket, no session build).
    #[arg(long)]
    warm: bool,

    /// Serve exactly one connection, then exit (for tests / scripting).
    #[arg(long)]
    once: bool,
}

fn main() -> std::process::ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ocrd=info,ocr_core=info".into()),
        )
        .with_target(false)
        .init();

    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("{e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    if args.warm {
        let paths = ocr_core::prefetch().context("prefetching models")?;
        for p in paths {
            println!("{}", p.display());
        }
        return Ok(());
    }

    let accel: Accel = args.accel.parse().map_err(|e: String| anyhow::anyhow!(e))?;
    let cfg = EngineConfig {
        accel,
        device_id: args.device_id,
        intra_threads: (args.threads > 0).then_some(args.threads),
        min_confidence: args.min_confidence,
        ..EngineConfig::default()
    };

    let t0 = Instant::now();
    let engine = Engine::load(&cfg).context("loading OCR engine")?;
    tracing::info!(
        accel = engine.accel().as_str(),
        startup_ms = t0.elapsed().as_millis() as u64,
        "engine ready"
    );

    let sock_path = args.socket.unwrap_or_else(default_socket_path);
    let listener = bind(&sock_path)?;
    let _guard = SocketGuard(sock_path.clone());
    tracing::info!(socket = %sock_path.display(), "listening");

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                if let Err(e) = handle(&engine, stream) {
                    tracing::warn!("connection error: {e:#}");
                }
                if args.once {
                    break;
                }
            }
            Err(e) => {
                tracing::warn!("accept failed: {e}");
            }
        }
    }
    Ok(())
}

/// Bind the socket with owner-only access. The socket carries screen captures
/// and their recognized text, so no other local user may reach it:
///
/// * the parent directory must be private (0700, owned by us) — enforced for
///   the temp-dir fallback, trusted for `$XDG_RUNTIME_DIR`;
/// * a stale socket is only removed if we own it;
/// * the socket file itself is chmod'd to 0600 right after bind.
fn bind(path: &Path) -> Result<UnixListener> {
    if let Some(parent) = path.parent() {
        prepare_socket_dir(parent)?;
    }

    if path.exists() {
        match UnixStream::connect(path) {
            Ok(_) => anyhow::bail!("another ocrd is already listening on {}", path.display()),
            Err(_) => {
                if !owned_by_us(path) {
                    anyhow::bail!(
                        "stale socket {} is not owned by this user; refusing to remove it",
                        path.display()
                    );
                }
                std::fs::remove_file(path)
                    .with_context(|| format!("removing stale socket {}", path.display()))?;
            }
        }
    }

    let listener =
        UnixListener::bind(path).with_context(|| format!("binding {}", path.display()))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("restricting {} to owner-only", path.display()))?;
    Ok(listener)
}

/// Ensure the socket's parent directory cannot be used by other users.
/// `$XDG_RUNTIME_DIR` is already a per-user 0700 directory shared by many apps'
/// sockets, so it is trusted as-is; any other parent (the temp-dir fallback, or
/// an explicit `--socket` path) is created 0700 and rejected if it already
/// exists owned by someone else or reachable by group/other.
fn prepare_socket_dir(dir: &Path) -> Result<()> {
    let is_xdg = std::env::var_os("XDG_RUNTIME_DIR")
        .map(|x| Path::new(&x) == dir)
        .unwrap_or(false);

    match std::fs::metadata(dir) {
        Ok(md) if md.is_dir() => {
            if !is_xdg {
                if !same_uid(md.uid()) {
                    anyhow::bail!(
                        "{} is owned by uid {}, not this user; refusing to use it",
                        dir.display(),
                        md.uid()
                    );
                }
                if md.permissions().mode() & 0o077 != 0 {
                    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
                        .with_context(|| format!("tightening {} to 0700", dir.display()))?;
                }
            }
            Ok(())
        }
        Ok(_) => anyhow::bail!("{} exists and is not a directory", dir.display()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
                .with_context(|| format!("restricting {} to 0700", dir.display()))?;
            Ok(())
        }
        Err(e) => Err(e).with_context(|| format!("stat {}", dir.display())),
    }
}

/// Real uid of this process, via `/proc/self` (Linux). `u32::MAX` on failure,
/// which [`same_uid`] treats as "no match".
fn own_uid() -> u32 {
    std::fs::metadata("/proc/self")
        .map(|m| m.uid())
        .unwrap_or(u32::MAX)
}

fn same_uid(uid: u32) -> bool {
    let mine = own_uid();
    mine != u32::MAX && mine == uid
}

fn owned_by_us(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|m| same_uid(m.uid()))
        .unwrap_or(false)
}

struct SocketGuard(PathBuf);
impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn handle(engine: &Engine, mut stream: UnixStream) -> Result<()> {
    let mut req_line = String::new();
    {
        let mut reader = BufReader::new(&mut stream);
        reader
            .read_line(&mut req_line)
            .context("reading request line")?;
    }
    if req_line.trim().is_empty() {
        return Ok(());
    }

    let response = match process(engine, &req_line) {
        Ok(ocr) => Response::Ok(ocr),
        Err(e) => Response::Error {
            message: format!("{e:#}"),
        },
    };

    let mut out = serde_json::to_string(&response).context("encoding response")?;
    out.push('\n');
    stream
        .write_all(out.as_bytes())
        .context("writing response")?;
    stream.flush().ok();
    Ok(())
}

fn process(engine: &Engine, req_line: &str) -> Result<Ocr> {
    let req: Request = serde_json::from_str(req_line.trim()).context("decoding request")?;
    let bytes = BASE64_STANDARD
        .decode(req.image_b64.as_bytes())
        .context("decoding base64 image")?;

    let t0 = Instant::now();
    let out = engine
        .recognize_bytes(&bytes, req.lang)
        .context("recognition failed")?;
    let elapsed_ms = t0.elapsed().as_millis() as u64;

    let lines = out
        .lines
        .iter()
        .map(|l| Line {
            text: l.text.clone(),
            confidence: l.confidence,
            bbox: l.points,
            lang: l.recognizer.as_str().to_string(),
        })
        .collect();

    tracing::info!(
        lang = %req.lang,
        lang_used = %out.label,
        lines = out.lines.len(),
        elapsed_ms,
        accel = out.accel.as_str(),
        "served"
    );

    Ok(Ocr {
        text: out.text(),
        lines,
        lang_used: out.label,
        accel: out.accel.as_str().to_string(),
        elapsed_ms,
    })
}
