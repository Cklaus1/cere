//! Compiles `proto/cere_worker.proto` with protoc (a build requirement; CI installs `protobuf-compiler`).
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?).join("../../proto");
    let file = root.join("cere_worker.proto");
    println!("cargo:rerun-if-changed={}", file.display());
    let out = PathBuf::from(std::env::var("OUT_DIR")?);
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .file_descriptor_set_path(out.join("cere_worker_descriptor.bin"))
        .compile_protos(&[file], &[root])?;
    Ok(())
}
