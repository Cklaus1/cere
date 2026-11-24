# AmdAmd PAR-6 first light: the AMD native runtime image. cered (built in cere-ci, glibc 2.36) and
# libcere_executor_hip.so (built in this base) run here with the ROCm 7.2.4 user space; the tokenizer sidecar
# (scripts/native/tokenizer_sidecar.py) needs transformers + tokenizers (text role; the grammar role's xgrammar /
# sglang are not installed: the AMD profile declares text output only). Build: docker build -t cere-amd:rocm7.2.4 .
FROM llama-amd:rocm7.2.4
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/sidecar && /opt/sidecar/bin/pip install --no-cache-dir transformers tokenizers jinja2 orjson jsonschema
ENV PATH=/opt/sidecar/bin:$PATH
