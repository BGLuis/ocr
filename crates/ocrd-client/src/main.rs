//! `ocrd-client` — the thin half of the pair (report section 2.7).
//!
//! Reads an image from stdin (or `--file`), hands it to the resident `ocrd`
//! over its Unix socket, and prints the recognized text. Designed for a
//! Hyprland bind:
//!
//! ```text
//! bind = SUPER SHIFT, T, exec, grim -g "$(slurp)" - | ocrd-client --lang auto | wl-copy
//! ```

use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use base64::prelude::{Engine as _, BASE64_STANDARD};
use clap::Parser;
use ocr_proto::{default_socket_path, Lang, Request, Response};

/// Send an image to ocrd and print the text it reads back.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Language / script. One of: auto, cjk (ja/zh/en), korean, latin (pt/es/fr/de).
    #[arg(short, long, default_value = "auto")]
    lang: String,

    /// Read the image from this file instead of stdin.
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Path to the ocrd Unix socket. Defaults to $XDG_RUNTIME_DIR/ocrd.sock.
    #[arg(long)]
    socket: Option<PathBuf>,

    /// Print the full JSON response instead of just the text.
    #[arg(long)]
    json: bool,

    /// Also print per-line confidence to stderr.
    #[arg(long)]
    verbose: bool,

    /// Override the daemon's suspect-confidence threshold for this request.
    #[arg(long)]
    min_confidence: Option<f32>,

    /// Exit 0 if ocrd is listening on the socket, non-zero otherwise. No image
    /// is read. For scripts that (re)start the daemon on demand.
    #[arg(long)]
    ping: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ocrd-client: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    let sock = args.socket.unwrap_or_else(default_socket_path);

    if args.ping {
        return UnixStream::connect(&sock)
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!("ocrd not reachable at {} ({e})", sock.display()));
    }

    let lang: Lang = args.lang.parse().map_err(|e: String| anyhow::anyhow!(e))?;

    let image = match &args.file {
        Some(p) => std::fs::read(p).with_context(|| format!("reading {}", p.display()))?,
        None => {
            let mut buf = Vec::new();
            std::io::stdin()
                .read_to_end(&mut buf)
                .context("reading image from stdin")?;
            buf
        }
    };
    if image.is_empty() {
        bail!("no image data (stdin was empty and --file was not given)");
    }

    let req = Request {
        image_b64: BASE64_STANDARD.encode(&image),
        lang,
        min_confidence: args.min_confidence,
        v: 1,
    };

    let mut stream = UnixStream::connect(&sock).map_err(|e| {
        anyhow::anyhow!(
            "cannot reach ocrd at {} ({e}).\n       Start it with:  systemctl --user start ocrd   (or run `ocrd` directly)",
            sock.display()
        )
    })?;

    let mut line = serde_json::to_string(&req).context("encoding request")?;
    line.push('\n');
    stream
        .write_all(line.as_bytes())
        .context("sending request to ocrd")?;
    stream.flush().ok();

    let mut resp_line = String::new();
    BufReader::new(&mut stream)
        .read_line(&mut resp_line)
        .context("reading response from ocrd")?;
    if resp_line.trim().is_empty() {
        bail!("ocrd closed the connection without replying");
    }

    let resp: Response = serde_json::from_str(resp_line.trim()).context("decoding response")?;

    match resp {
        Response::Error { message } => bail!("ocrd error: {message}"),
        Response::Ok(ocr) => {
            if args.json {
                println!("{}", serde_json::to_string_pretty(&ocr)?);
                return Ok(());
            }
            if args.verbose {
                for l in &ocr.lines {
                    eprintln!("  [{:>5.1}%  {}]  {}", l.confidence * 100.0, l.lang, l.text);
                }
                eprintln!(
                    "  -- {} line(s), {}, {} ms, {}",
                    ocr.lines.len(),
                    ocr.lang_used,
                    ocr.elapsed_ms,
                    ocr.accel
                );
            }
            print!("{}", ocr.text);
            if !ocr.text.ends_with('\n') {
                println!();
            }
            Ok(())
        }
    }
}
