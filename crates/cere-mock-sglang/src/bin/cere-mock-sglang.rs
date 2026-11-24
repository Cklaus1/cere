//! CI-only fake of the probed SGLang HTTP subset (m1 design note K1). Not M1 acceptance evidence.
//!
//! Usage: `cere-mock-sglang [--port N] [--server-info FILE] [--token-latency-ms N] [--max-running-requests N]
//! [--natural-output-tokens N] [--startup-delay-ms N]`.
//! Prints the bound base URL on stdout and serves until SIGINT or SIGTERM.
//!
//! - `--port N`: port on 127.0.0.1 (default 0 = ephemeral).
//! - `--server-info FILE`: a JSON `/get_server_info` body (e.g. probed from a real server) served instead of the
//!   default one, every key kept; its `served_model_name` is the `/v1/models` entry and its `context_length`,
//!   `enable_cache_report`, `disable_radix_cache` and `max_running_requests` drive the mock.
//! - `--token-latency-ms N`: delay before each output token (default 0).
//! - `--max-running-requests N` (N >= 1): the scheduler cap reported in `/get_server_info` and applied to
//!   `/get_load`; it wins over the file's value (default: the file's value, else 4).
//! - `--natural-output-tokens N`: output tokens before the natural EOS (`finish_reason:"stop"`), unless
//!   `max_tokens` ends the stream first (`length`) (default 8). Set it above `max_tokens` so the budget governs.
//! - `--startup-delay-ms N` (0..=600000): sleep before binding the listener, so a supervising agent can observe
//!   `loading` (m36 design note Q14) (default 0).
//!
//! `startup_time` is always the mock's own launch time (Unix seconds, a JSON number), whatever the file says,
//! so with the file's `random_seed` each launch gets a new readiness `instance_id`. A bad flag or value, or an
//! unreadable or invalid file, exits 2 with the usage line on stderr.

use cere_mock_sglang::{MockConfig, MockSglang, ServerInfo};
use std::io::Write;
use std::process::ExitCode;
use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const USAGE: &str = "usage: cere-mock-sglang [--port N] [--server-info FILE] [--token-latency-ms N] [--max-running-requests N] [--natural-output-tokens N] [--startup-delay-ms N]";

/// Upper bound of `--startup-delay-ms` (m36 design note Q14).
const MAX_STARTUP_DELAY_MS: u64 = 600_000;

#[derive(Default)]
struct Args {
    port: Option<u16>,
    server_info: Option<String>,
    token_latency_ms: Option<u64>,
    max_running_requests: Option<u32>,
    natural_output_tokens: Option<u32>,
    startup_delay_ms: Option<u64>,
}

fn set<T>(slot: &mut Option<T>, flag: &str, value: T) -> Result<(), String> {
    match slot.replace(value) {
        Some(_) => Err(format!("{flag} given twice")),
        None => Ok(()),
    }
}

fn number<T: FromStr>(flag: &str, value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid {flag} value {value:?}"))
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut rest = args.iter();
    while let Some(flag) = rest.next() {
        let flag = flag.as_str();
        if !matches!(
            flag,
            "--port"
                | "--server-info"
                | "--token-latency-ms"
                | "--max-running-requests"
                | "--natural-output-tokens"
                | "--startup-delay-ms"
        ) {
            return Err(format!("unknown argument {flag:?}"));
        }
        let value = rest.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag {
            "--port" => set(&mut parsed.port, flag, number(flag, value)?)?,
            "--server-info" => set(&mut parsed.server_info, flag, value.clone())?,
            "--token-latency-ms" => set(&mut parsed.token_latency_ms, flag, number(flag, value)?)?,
            "--natural-output-tokens" => set(
                &mut parsed.natural_output_tokens,
                flag,
                number(flag, value)?,
            )?,
            "--startup-delay-ms" => {
                let delay: u64 = number(flag, value)?;
                if delay > MAX_STARTUP_DELAY_MS {
                    return Err(format!("{flag} must be at most {MAX_STARTUP_DELAY_MS}"));
                }
                set(&mut parsed.startup_delay_ms, flag, delay)?;
            }
            _ => {
                let cap: u32 = number(flag, value)?;
                if cap == 0 {
                    return Err(format!("{flag} must be at least 1"));
                }
                set(&mut parsed.max_running_requests, flag, cap)?;
            }
        }
    }
    Ok(parsed)
}

fn read_server_info(path: &str) -> Result<ServerInfo, String> {
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("--server-info {path}: {err}"))?;
    serde_json::from_str(&text).map_err(|err| format!("--server-info {path}: {err}"))
}

/// The served configuration; `launched` (Unix seconds) becomes `startup_time`.
fn mock_config(args: &Args, launched: f64) -> Result<MockConfig, String> {
    let mut server_info = match &args.server_info {
        Some(path) => read_server_info(path)?,
        None => ServerInfo::default(),
    };
    if let Some(cap) = args.max_running_requests {
        server_info.max_running_requests = cap;
    }
    server_info.startup_time = Some(serde_json::json!(launched));
    Ok(MockConfig {
        port: args.port.unwrap_or(0),
        server_info,
        token_latency: Duration::from_millis(args.token_latency_ms.unwrap_or(0)),
        natural_output_tokens: args
            .natural_output_tokens
            .unwrap_or(MockConfig::default().natural_output_tokens),
        ..MockConfig::default()
    })
}

#[tokio::main]
async fn main() -> ExitCode {
    let launched = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (config, startup_delay) = match parse_args(&args).and_then(|args| {
        Ok((
            mock_config(&args, launched)?,
            args.startup_delay_ms.unwrap_or(0),
        ))
    }) {
        Ok(parsed) => parsed,
        Err(message) => {
            eprintln!("cere-mock-sglang: {message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    // The delay keeps the inherited signal dispositions (a supervisor test runs the mock under `trap '' TERM`).
    tokio::time::sleep(Duration::from_millis(startup_delay)).await;
    // Handlers go in before the listener binds and the address is printed: a SIGTERM sent once the caller sees the
    // address must take the graceful path, never the default action.
    let mut stop = match StopSignal::install() {
        Ok(stop) => stop,
        Err(err) => {
            eprintln!("cere-mock-sglang: signal handler: {err}");
            return ExitCode::FAILURE;
        }
    };
    let port = config.port;
    let mock = match MockSglang::start(config).await {
        Ok(mock) => mock,
        Err(err) => {
            eprintln!("cere-mock-sglang: bind 127.0.0.1:{port}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let mut stdout = std::io::stdout();
    if writeln!(stdout, "cere-mock-sglang listening on {}", mock.base_url())
        .and_then(|()| stdout.flush())
        .is_err()
    {
        return ExitCode::FAILURE;
    }
    stop.recv().await;
    match mock.shutdown().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("cere-mock-sglang: shutdown: {err}");
            ExitCode::FAILURE
        }
    }
}

/// SIGTERM and SIGINT streams, registered up front.
struct StopSignal {
    term: tokio::signal::unix::Signal,
    int: tokio::signal::unix::Signal,
}

impl StopSignal {
    fn install() -> std::io::Result<Self> {
        use tokio::signal::unix::{signal, SignalKind};
        Ok(Self {
            term: signal(SignalKind::terminate())?,
            int: signal(SignalKind::interrupt())?,
        })
    }

    async fn recv(&mut self) {
        tokio::select! {
            _ = self.term.recv() => {}
            _ = self.int.recv() => {}
        }
    }
}
