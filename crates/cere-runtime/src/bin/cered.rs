//! `cered`: the CERE front end for one pinned, backend-managed SGLang server (m1 design note Wave C).
//!
//! `cered --config <toml>` loads the config, resolves the auth tokens (a missing or empty token env var is a
//! startup failure), opens the ledger (new epoch; prior-epoch requests fail `worker_restart` and are aborted
//! on the backend), starts readiness, binds `server.bind` and prints one stdout line
//! `cered listening on http://<addr> epoch=<n>`. With `[federation]` (M3.5) it first loads the coordinator's
//! PKI (unreadable or invalid: exit 2), serves the worker registry on `federation.listen` and prints one stderr
//! line `cered federation listening on <addr> cluster=<cluster_id> term=<epoch>`. SIGTERM/SIGINT run the
//! graceful shutdown (L8); the exit status is 0 only if every in-flight request reached its terminal state.
use std::io::Write;
use std::process::ExitCode;

use cere_runtime::app::{App, Tuning};
use cere_runtime::config::RuntimeConfig;
use cere_runtime::server;

const USAGE: &str = "cered [--help | --version | --check-config <path> | --config <path>]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [arg] if arg == "--help" => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        [arg] if arg == "--version" => {
            println!(
                "cered {}; package v{}",
                env!("CARGO_PKG_VERSION"),
                include_str!("../../../../PACKAGE_VERSION").trim()
            );
            ExitCode::SUCCESS
        }
        [arg, path] if arg == "--check-config" => match RuntimeConfig::load(path) {
            Ok(config) => {
                println!("config ok {}", config.config_hash());
                // One line per entry with a pinned deployment, in config order (M5 D10).
                for entry in &config.deployments {
                    if let (Some(name), Some(d)) = (&entry.served_model_name, &entry.lock) {
                        println!(
                            "deployment served_model_name={name} recipe_id={} lock_sha256={} profile_id={}",
                            d.recipe.recipe_id, d.lock_sha256, d.profile.profile_id
                        );
                    }
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::from(2)
            }
        },
        [arg, path] if arg == "--config" => serve(path),
        _ => {
            eprintln!("usage: {USAGE}");
            ExitCode::from(2)
        }
    }
}

fn serve(path: &str) -> ExitCode {
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("cered: cannot start the async runtime: {e}");
            return ExitCode::from(2);
        }
    };
    match runtime.block_on(run(path)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            eprintln!("cered: shutdown forced; not every request reached its terminal state");
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("cered: {e}");
            ExitCode::from(2)
        }
    }
}

async fn run(path: &str) -> Result<bool, String> {
    let config = RuntimeConfig::load(path).map_err(|e| e.to_string())?;
    let credentials = config.resolve_auth().map_err(|e| e.to_string())?;
    let operator_token = config.resolve_operator_token().map_err(|e| e.to_string())?;
    let bind = config.server.bind.clone();
    let cluster = config.federation.as_ref().map(|f| f.cluster_id.clone());
    let app = App::open(config, &credentials, operator_token, Tuning::default())?;
    let settings = app.settings();
    eprintln!(
        "cered config generation={} config_hash={}",
        settings.generation, settings.config_hash
    );
    if let (Some(addr), Some(cluster)) = (app.federation_addr(), cluster) {
        eprintln!(
            "cered federation listening on {addr} cluster={cluster} term={}",
            app.epoch()
        );
    }
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .map_err(|e| format!("cannot bind {bind}: {e}"))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "cered listening on http://{addr} epoch={}",
        app.epoch()
    )
    .and_then(|()| stdout.flush())
    .map_err(|e| format!("cannot write the startup line: {e}"))?;
    drop(stdout);
    server::serve(app, listener, stop_signal())
        .await
        .map_err(|e| format!("server error: {e}"))
}

/// Resolves on the first SIGTERM or SIGINT.
async fn stop_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let (mut term, mut int) = match (
        signal(SignalKind::terminate()),
        signal(SignalKind::interrupt()),
    ) {
        (Ok(term), Ok(int)) => (term, int),
        _ => {
            eprintln!("cered: cannot install signal handlers; stopping");
            return;
        }
    };
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
    }
}
