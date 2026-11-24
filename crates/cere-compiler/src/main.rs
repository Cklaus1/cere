//! `cerec`: the M2.1 prepared-artifact tool (m21 design note R1-R5). Every rejection exits non-zero with a first
//! stderr line `<error_class>: <message>` and nothing on stdout.
use cere_compiler::artifact::{Cache, Class, Mode, ReadMode};
use cere_compiler::{
    canon_json, convert, inventory, prepare, probe, verify, Error, Limits, EXIT_REJECT,
};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

const USAGE: &str = "\
cerec convert SRC --out DIR --converter-git-sha SHA [--format s2|s3] [--threads N] [--limits JSON]
cerec inventory SRC --families F [--out FILE] [--limits JSON]
cerec prepare SRC [--token-map PATH] --name NAME [--lock LOCK] --families F --profile P --out ROOT
              [--no-wait] [--threads N] [--min-mem-available-gib G] [--limits JSON]
cerec verify DIR --expected-manifest-sha256 H --mode full|stamp [--cache cold] [--limits JSON]
              [--min-mem-available-gib G]
cerec probe-load DIR --expected-manifest-sha256 H --class device|host|all --cache cold|warm --hash on|off
              [--baseline] [--threads N] [--buffer-mib M] [--read buffered|direct|auto] [--min-mem-available-gib G]
              [--tensor-digests on|off (a per-tensor CRC-32 summary of the delivered bytes, key tensor_digests)]
cerec evict DIR
cerec --help | --version";

/// Parsed `--flag value` options, boolean switches and positionals.
struct Args {
    positional: Vec<String>,
    options: BTreeMap<String, String>,
    switches: Vec<String>,
}

fn usage(msg: impl Into<String>) -> Error {
    Error::reject("usage", format!("{}\n{USAGE}", msg.into()))
}

impl Args {
    /// `valued` lists the options that take a value; `switches` the boolean ones.
    fn parse(raw: &[String], valued: &[&str], switches: &[&str]) -> Result<Args, Error> {
        let mut out = Args {
            positional: Vec::new(),
            options: BTreeMap::new(),
            switches: Vec::new(),
        };
        let mut it = raw.iter();
        while let Some(a) = it.next() {
            if let Some(name) = a.strip_prefix("--") {
                if valued.contains(&name) {
                    let v = it
                        .next()
                        .ok_or_else(|| usage(format!("--{name} needs a value")))?;
                    if out.options.insert(name.to_string(), v.clone()).is_some() {
                        return Err(usage(format!("--{name} given twice")));
                    }
                } else if switches.contains(&name) {
                    out.switches.push(name.to_string());
                } else {
                    return Err(usage(format!("unknown option --{name}")));
                }
            } else {
                out.positional.push(a.clone());
            }
        }
        Ok(out)
    }

    fn one_positional(&self, what: &str) -> Result<PathBuf, Error> {
        match self.positional.as_slice() {
            [p] => Ok(PathBuf::from(p)),
            _ => Err(usage(format!("expected exactly one {what}"))),
        }
    }

    fn required(&self, name: &str) -> Result<&str, Error> {
        self.options
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| usage(format!("--{name} is required")))
    }

    fn limits(&self) -> Result<Limits, Error> {
        match self.options.get("limits") {
            None => Ok(Limits::default()),
            Some(text) => {
                Limits::from_json(text.as_bytes()).map_err(|e| usage(format!("--limits: {e}")))
            }
        }
    }
}

fn read_input(path: &str) -> Result<Vec<u8>, Error> {
    std::fs::read(path).map_err(|e| Error::reject("input", format!("{path}: {e}")))
}

fn emit(bytes: &[u8], out: Option<&String>) -> Result<(), Error> {
    match out {
        Some(path) => std::fs::write(path, bytes).map_err(|e| Error::io(path, e)),
        None => std::io::stdout()
            .write_all(bytes)
            .map_err(|e| Error::io("stdout", e)),
    }
}

fn cmd_inventory(raw: &[String]) -> Result<(), Error> {
    let a = Args::parse(raw, &["families", "out", "limits"], &[])?;
    let src = a.one_positional("SRC")?;
    let families = read_input(a.required("families")?)?;
    let bytes = inventory::run(&src, &families, &a.limits()?)?;
    emit(&bytes, a.options.get("out"))
}

fn cmd_convert(raw: &[String]) -> Result<(), Error> {
    let a = Args::parse(
        raw,
        &["out", "converter-git-sha", "format", "threads", "limits"],
        &[],
    )?;
    let threads = number(&a, "threads", 8)?;
    if !(1..=256).contains(&threads) {
        return Err(usage("--threads must be 1..=256"));
    }
    let format = match a.options.get("format") {
        None => convert::Format::S2,
        Some(f) => convert::Format::parse(f).ok_or_else(|| usage("--format must be s2 or s3"))?,
    };
    let opts = convert::Options {
        src: a.one_positional("SRC")?,
        out: PathBuf::from(a.required("out")?),
        converter_git_sha: a.required("converter-git-sha")?.to_string(),
        threads: threads as usize,
        limits: a.limits()?,
        format,
    };
    print_report(&convert::run(&opts)?)
}

fn number(a: &Args, name: &str, default: u64) -> Result<u64, Error> {
    match a.options.get(name) {
        None => Ok(default),
        Some(v) => v
            .parse()
            .map_err(|_| usage(format!("--{name} must be a non-negative integer"))),
    }
}

fn print_report(v: &serde_json::Value) -> Result<(), Error> {
    let bytes = canon_json::to_vec(v).map_err(|e| Error::io("encoding report", e))?;
    emit(&bytes, None)
}

fn cmd_prepare(raw: &[String]) -> Result<(), Error> {
    let a = Args::parse(
        raw,
        &[
            "token-map",
            "name",
            "lock",
            "families",
            "profile",
            "out",
            "threads",
            "min-mem-available-gib",
            "limits",
        ],
        &["no-wait"],
    )?;
    let threads = number(&a, "threads", 8)?;
    if !(1..=256).contains(&threads) {
        return Err(usage("--threads must be 1..=256"));
    }
    let opts = prepare::Options {
        src: a.one_positional("SRC")?,
        token_map: a.options.get("token-map").map(PathBuf::from),
        name: a.required("name")?.to_string(),
        lock: a.options.get("lock").map(|p| read_input(p)).transpose()?,
        families: read_input(a.required("families")?)?,
        profile: read_input(a.required("profile")?)?,
        out: PathBuf::from(a.required("out")?),
        no_wait: a.switches.iter().any(|s| s == "no-wait"),
        threads: threads as usize,
        min_mem_available_gib: number(&a, "min-mem-available-gib", 16)?,
        limits: a.limits()?,
    };
    print_report(&prepare::run(&opts)?)
}

/// `--name value` restricted to `choices`.
fn choice<T>(
    a: &Args,
    name: &str,
    parse: impl Fn(&str) -> Option<T>,
    choices: &str,
) -> Result<T, Error> {
    parse(a.required(name)?).ok_or_else(|| usage(format!("--{name} must be {choices}")))
}

fn cmd_verify(raw: &[String]) -> Result<(), Error> {
    let a = Args::parse(
        raw,
        &[
            "expected-manifest-sha256",
            "mode",
            "cache",
            "min-mem-available-gib",
            "limits",
        ],
        &[],
    )?;
    let mode = choice(
        &a,
        "mode",
        |s| Mode::parse(s).filter(|m| *m != Mode::Headers),
        "full or stamp",
    )?;
    let cold = match a.options.get("cache").map(String::as_str) {
        None => false,
        Some("cold") => true,
        Some(_) => return Err(usage("--cache must be cold")),
    };
    let opts = verify::Options {
        dir: a.one_positional("DIR")?,
        expected_manifest_sha256: a.required("expected-manifest-sha256")?.to_string(),
        mode,
        cold,
        limits: a.limits()?,
        min_mem_available_gib: number(&a, "min-mem-available-gib", 16)?,
    };
    print_report(&verify::run(&opts)?)
}

fn cmd_probe_load(raw: &[String]) -> Result<(), Error> {
    let a = Args::parse(
        raw,
        &[
            "expected-manifest-sha256",
            "class",
            "cache",
            "hash",
            "threads",
            "buffer-mib",
            "min-mem-available-gib",
            "read",
            "tensor-digests",
        ],
        &["baseline"],
    )?;
    let threads = number(&a, "threads", 8)?;
    if !(1..=256).contains(&threads) {
        return Err(usage("--threads must be 1..=256"));
    }
    let buffer_mib = number(&a, "buffer-mib", 64)?;
    if !(1..=1024).contains(&buffer_mib) {
        return Err(usage("--buffer-mib must be 1..=1024"));
    }
    let on_off = |s: &str| match s {
        "on" => Some(true),
        "off" => Some(false),
        _ => None,
    };
    let hash = choice(&a, "hash", on_off, "on or off")?;
    let tensor_digests = match a.options.get("tensor-digests") {
        None => false,
        Some(_) => choice(&a, "tensor-digests", on_off, "on or off")?,
    };
    let opts = probe::Options {
        dir: a.one_positional("DIR")?,
        expected_manifest_sha256: a.required("expected-manifest-sha256")?.to_string(),
        class: choice(&a, "class", Class::parse, "device, host or all")?,
        cache: choice(&a, "cache", Cache::parse, "cold or warm")?,
        hash,
        baseline: a.switches.iter().any(|s| s == "baseline"),
        threads: threads as usize,
        buffer_bytes: (buffer_mib as usize) << 20,
        read: match a.options.get("read") {
            None => ReadMode::Buffered,
            Some(_) => choice(&a, "read", ReadMode::parse, "buffered, direct or auto")?,
        },
        min_mem_available_gib: number(&a, "min-mem-available-gib", 16)?,
        tensor_digests,
    };
    print_report(&probe::run(&opts)?)
}

fn cmd_evict(raw: &[String]) -> Result<(), Error> {
    let a = Args::parse(raw, &[], &[])?;
    print_report(&verify::evict(
        &a.one_positional("DIR")?,
        &Limits::default(),
    )?)
}

fn run(args: &[String]) -> Result<(), Error> {
    match args.split_first() {
        None => Err(usage("no command")),
        Some((cmd, rest)) => match (cmd.as_str(), rest) {
            ("--help", []) => {
                println!("{USAGE}");
                Ok(())
            }
            ("--version", []) => {
                println!(
                    "cerec {}; package v{}",
                    env!("CARGO_PKG_VERSION"),
                    include_str!("../../../PACKAGE_VERSION").trim()
                );
                Ok(())
            }
            ("convert", rest) => cmd_convert(rest),
            ("inventory", rest) => cmd_inventory(rest),
            ("prepare", rest) => cmd_prepare(rest),
            ("verify", rest) => cmd_verify(rest),
            ("probe-load", rest) => cmd_probe_load(rest),
            ("evict", rest) => cmd_evict(rest),
            (other, _) => Err(usage(format!("unknown command {other}"))),
        },
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("{e}");
        std::process::exit(if e.exit == 0 { EXIT_REJECT } else { e.exit });
    }
}
