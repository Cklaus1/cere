//! `cere-worker`: the M3.5 node agent (m35 design note Q3, row P5) for one worker over one loopback backend,
//! served to its coordinator over mTLS gRPC.
//!
//! `cere-worker --config <toml>` starts the agent (state-dir lock, epoch +1 persisted before anything is served or
//! registered, PKI, in managed mode the reconciliation of a recorded backend group and the supervisor, listener,
//! link, backend probe) and prints one stdout line
//! `cere-worker listening on <addr> worker=<id>@<epoch> generation=<g>`; events go to stderr as
//! `cere-worker event=<kind> ...`. SIGTERM/SIGINT stop it (exit 0; a managed backend group is stopped first).
//! `--check-config <toml>` validates the config shape without opening a PEM (exit 0, or 2 with the message on
//! stderr). A recorded backend group that survives start reconciliation exits 1 (design-m36-agent.md §2.3); every
//! other startup error exits 2. `owner-only on|off --config <toml>` sets the agent's local owner-only switch
//! (net-teleport design note 4.4, C19: a marker in its `state_dir`; the running agent refuses other scopes' new
//! work while it is on and reports it in its heartbeat); it needs the config's `owner` (exit 2 otherwise).
use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use cere_runtime::agent::config::AgentConfig;
use cere_runtime::agent::{Agent, AgentError};

const USAGE: &str = "cere-worker [--help | --version | --check-config <path> | --config <path> | owner-only on|off --config <path>]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [arg] if arg == "--help" => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        [arg] if arg == "--version" => {
            println!(
                "cere-worker {}",
                include_str!("../../../../PACKAGE_VERSION").trim()
            );
            ExitCode::SUCCESS
        }
        [arg, path] if arg == "--check-config" => match AgentConfig::load(Path::new(path)) {
            Ok(config) => {
                println!("config ok worker={}", config.worker_id);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::from(2)
            }
        },
        [arg, path] if arg == "--config" => serve(Path::new(path)),
        [cmd, state, flag, path] if cmd == "owner-only" && flag == "--config" => {
            owner_only(state, Path::new(path))
        }
        _ => {
            eprintln!("usage: {USAGE}");
            ExitCode::from(2)
        }
    }
}

/// `cere-worker owner-only on|off --config <path>`.
fn owner_only(state: &str, path: &Path) -> ExitCode {
    let on = match state {
        "on" => true,
        "off" => false,
        _ => {
            eprintln!("usage: {USAGE}");
            return ExitCode::from(2);
        }
    };
    let config = match AgentConfig::load(path) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    if config.owner.is_none() {
        eprintln!("cere-worker: owner-only needs the config's owner");
        return ExitCode::from(2);
    }
    match cere_runtime::pool::set_owner_only_switch(&config.state_dir, on) {
        Ok(()) => {
            println!("owner-only {state} worker={}", config.worker_id);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("cere-worker: cannot set the owner-only switch: {e}");
            ExitCode::from(2)
        }
    }
}

fn serve(path: &Path) -> ExitCode {
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("cere-worker: cannot start the async runtime: {e}");
            return ExitCode::from(2);
        }
    };
    match runtime.block_on(run(path)) {
        Ok(()) => ExitCode::SUCCESS,
        Err((code, e)) => {
            eprintln!("cere-worker: {e}");
            ExitCode::from(code)
        }
    }
}

/// `Err((exit code, message))`.
async fn run(path: &Path) -> Result<(), (u8, String)> {
    use tokio::signal::unix::{signal, SignalKind};
    let fail = |e: String| (2, e);
    // Installed before the startup line, so a stop right after it is never the default (fatal) action.
    let mut term =
        signal(SignalKind::terminate()).map_err(|e| fail(format!("signal handler: {e}")))?;
    let mut int =
        signal(SignalKind::interrupt()).map_err(|e| fail(format!("signal handler: {e}")))?;
    let config = AgentConfig::load(path).map_err(|e| fail(e.to_string()))?;
    let worker_id = config.worker_id.clone();
    let agent = Agent::start(config).await.map_err(|e| match e {
        AgentError::Reconcile(_) => (1, e.to_string()),
        e => fail(e.to_string()),
    })?;
    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "cere-worker listening on {} worker={worker_id}@{} generation={}",
        agent.local_addr(),
        agent.epoch(),
        agent.generation()
    )
    .and_then(|()| stdout.flush())
    .map_err(|e| fail(format!("cannot write the startup line: {e}")))?;
    drop(stdout);
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
    }
    agent.shutdown().await;
    Ok(())
}
