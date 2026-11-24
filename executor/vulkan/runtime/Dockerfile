# Vulkan first-light runtime image (vulkan-amd design note D3, design 5.3): a Mesa-only user space, no ROCm. cered (built in
# cere-ci, glibc 2.36) and libcere_executor_vulkan.so run here against Mesa's RADV; the tokenizer sidecar
# (scripts/native/tokenizer_sidecar.py) needs transformers + tokenizers (text role; no xgrammar / sglang: the AMD profiles
# declare text output only). The sidecar packages are the ones executor/hip/runtime/Dockerfile installs (it pins no version),
# pinned here to the versions that image holds today (checked in cere-amd:rocm7.2.4): transformers 5.22.1, tokenizers 0.23.3,
# Jinja2 3.1.6, orjson 3.13.0, jsonschema 4.26.0; Python 3.12.3 (Ubuntu 24.04).
# Mesa: `apt-cache policy mesa-vulkan-drivers` in ubuntu:24.04 (2026-11-15) gives candidate 25.2.8-0ubuntu0.24.04.4
# (noble-updates), the version the kernel tests ran on (llama-amd:rocm7.2.4 holds the same). The archive moves: the build
# does not pin the apt version; record `vulkaninfo --summary` / the dpkg line of the built image as evidence.
# libvulkan1 here is 1.3.275.0-1build1. vulkan-tools is added so `vulkaninfo --summary` exists in the image.
# Build (NOT built yet; needs network). Expected size, an UNVERIFIED estimate from pieces measured in other images: ubuntu:24.04
# 119 MB + Mesa drivers with libLLVM 20 (~0.2-0.3 GB) + the sidecar venv (274 MB in cere-amd:rocm7.2.4), i.e. about 0.7-0.9 GB
# on disk (cere-amd:rocm7.2.4 is 14.2 GB):
#   docker build -t cere-amd-vk:mesa executor/vulkan/runtime
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv mesa-vulkan-drivers libvulkan1 vulkan-tools \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/sidecar && /opt/sidecar/bin/pip install --no-cache-dir \
    transformers==5.22.1 tokenizers==0.23.3 jinja2==3.1.6 orjson==3.13.0 jsonschema==4.26.0
ENV PATH=/opt/sidecar/bin:$PATH
