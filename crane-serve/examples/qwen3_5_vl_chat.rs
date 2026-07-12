//! Qwen 3.5 vision-language chat-completions server example.
//!
//! Boots crane-serve pointed at a local Qwen 3.5 multimodal checkpoint, then
//! waits for `/health` and POSTs a sample image to `/v1/chat/completions`.
//! Prints the model's description. End-to-end proof that the image
//! preprocessing + vision tower + text-decoder splice + chat-template
//! wiring all line up against a real OpenAI-compatible client.
//!
//! Usage:
//!
//! ```text
//! CRANE_QWEN35_VL_DIR=/path/to/Qwen3.5-0.8B \
//!     cargo run --release --features cuda --example qwen3_5_vl_chat -- \
//!     --port 8600 --image /tmp/test.png
//! ```
//!
//! The example:
//! 1. Starts crane-serve on `--port` with `--model-type qwen3_5_vl`.
//! 2. Base64-encodes the image at `--image` (default `/tmp/test_image.png`)
//!    and POSTs an OpenAI-style multimodal request to `/v1/chat/completions`.
//! 3. Decodes the JSON response and prints the assistant's message content.
//!
//! For interactive use, leave the server running (omit `--once`) and curl
//! from another shell — see `crates/qwen3_5_vl_chat/README.md` (in the
//! repo) for example payloads.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

const DEFAULT_MODEL_DIR: &str = "/home/hahihula/mywork/ai/additional_models/Qwen3.5-0.8B";
const DEFAULT_IMAGE: &str = "/tmp/test_image.png";

struct Args {
    port: u16,
    model_dir: PathBuf,
    image: PathBuf,
    max_tokens: usize,
    prompt: String,
    once: bool,
}

fn parse_args() -> Result<Args> {
    let mut port: u16 = 8600;
    let mut model_dir = PathBuf::from(DEFAULT_MODEL_DIR);
    let mut image = PathBuf::from(DEFAULT_IMAGE);
    let mut max_tokens: usize = 96;
    let mut prompt = "Briefly: what is in this image?".to_string();
    let mut once = true;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--port" => {
                port = it.next().context("--port needs a value")?.parse()?;
            }
            "--model-dir" => {
                model_dir = PathBuf::from(it.next().context("--model-dir needs a value")?);
            }
            "--image" => {
                image = PathBuf::from(it.next().context("--image needs a value")?);
            }
            "--max-tokens" => {
                max_tokens = it.next().context("--max-tokens needs a value")?.parse()?;
            }
            "--prompt" => {
                prompt = it.next().context("--prompt needs a value")?;
            }
            "--keep-running" => once = false,
            "-h" | "--help" => {
                println!(
                    "Usage: qwen3_5_vl_chat [--port N] [--model-dir DIR] [--image FILE] \
                     [--max-tokens N] [--prompt \"...\"] [--keep-running]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("Unknown argument: {other}"),
        }
    }

    Ok(Args {
        port,
        model_dir,
        image,
        max_tokens,
        prompt,
        once,
    })
}

fn spawn_server(model_dir: &PathBuf, port: u16) -> Result<Child> {
    let status = Command::new("cargo")
        .args([
            "run",
            "-q",
            "-p",
            "crane-serve",
            "--release",
            "--features",
            "cuda",
            "--",
            "--model-path",
        ])
        .arg(model_dir)
        .args(["--model-type", "qwen3_5_vl", "--port", &port.to_string()])
        .args(["--max-concurrent", "1"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn crane-serve (is `cargo` on PATH?)")?;
    Ok(status)
}

fn wait_for_health(port: u16, timeout: Duration) -> Result<()> {
    let client = reqwest::blocking::Client::new();
    let url = format!("http://127.0.0.1:{port}/health");
    let deadline = Instant::now() + timeout;
    loop {
        if client.get(&url).send().map(|r| r.status().is_success()).unwrap_or(false) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            anyhow::bail!("crane-serve did not become healthy within {timeout:?}");
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn base64_encode(path: &PathBuf) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    use base64::Engine;
    Ok(base64::engine::general_purpose::STANDARD.encode(&bytes))
}

fn send_chat_request(port: u16, image_path: &PathBuf, prompt: &str, max_tokens: usize) -> Result<String> {
    let b64 = base64_encode(image_path)?;
    let payload = serde_json::json!({
        "model": "qwen3.5",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{b64}")}}
            ]
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    });

    let client = reqwest::blocking::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let resp = client
        .post(&url)
        .json(&payload)
        .send()
        .context("POST /v1/chat/completions failed")?;
    let status = resp.status();
    let body = resp.text()?;
    if !status.is_success() {
        anyhow::bail!("chat completions returned HTTP {status}: {body}");
    }
    let json: serde_json::Value =
        serde_json::from_str(&body).context("decode response JSON")?;
    let text = json["choices"][0]["message"]["content"]
        .as_str()
        .context("response had no message.content")?;
    Ok(text.to_string())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    eprintln!(
        "[qwen3_5_vl_chat] model_dir={} image={} port={} max_tokens={}",
        args.model_dir.display(),
        args.image.display(),
        args.port,
        args.max_tokens
    );

    if !args.image.exists() {
        anyhow::bail!(
            "image not found at {} (pass --image PATH or set CRANE_QWEN35_VL_IMAGE)",
            args.image.display()
        );
    }

    eprintln!("[qwen3_5_vl_chat] spawning crane-serve (--model-type qwen3_5_vl)...");
    let mut child = spawn_server(&args.model_dir, args.port)?;

    // Always kill the child on exit, even if the chat request errors.
    let kill_guard = scopeguard_lite(&mut child);

    wait_for_health(args.port, Duration::from_secs(120))?;
    eprintln!("[qwen3_5_vl_chat] server healthy on port {}", args.port);

    let text = send_chat_request(args.port, &args.image, &args.prompt, args.max_tokens)?;
    println!("[qwen3_5_vl_chat] model said:");
    println!("---");
    println!("{text}");
    println!("---");

    if args.once {
        drop(kill_guard);
        let _ = child.kill();
        let _ = child.wait();
        eprintln!("[qwen3_5_vl_chat] done; server killed.");
    } else {
        eprintln!(
            "[qwen3_5_vl_chat] --keep-running: server stays up on port {}. \
             Hit Ctrl-C to stop.",
            args.port
        );
        // Don't drop the guard; the OS reaps the child on exit.
        std::mem::forget(kill_guard);
        let _ = child.wait();
    }

    Ok(())
}

/// Tiny RAII guard that kills the child if dropped. Avoids the `scopeguard`
/// dep — we only need one operation.
fn scopeguard_lite(child: &mut Child) -> KillGuard<'_> {
    KillGuard { child }
}

struct KillGuard<'a> {
    child: &'a mut Child,
}

impl Drop for KillGuard<'_> {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::io::stdout().flush();
    }
}