.PHONY: validate test test-python test-c test-executor-host test-rust test-proto check fmt clippy deny ci \
  toy-executors executor-cuda

validate:
	python3 scripts/check_third_party.py && python3 scripts/pin_configs.py --check

# Everything that runs on a CPU-only machine (no GPU, no model weights, no network).
test: validate test-python test-c test-executor-host test-rust test-proto

ci: test fmt clippy

# `-t .` imports the `tests` package, which keeps host state (production markers, CARGO_TARGET_DIR) out of the tests
# (tests/__init__.py).
test-python:
	python3 -m unittest discover -s tests -t . -p 'test_*.py'

test-c:
	@set -eu; d=$$(mktemp -d); trap 'rm -rf "$$d"' EXIT; \
	$${CC:-cc} -std=c11 -Wall -Wextra -Werror -Iexecutor/include tests/test_executor_abi.c -o "$$d/c-check"; \
	"$$d/c-check" > "$$d/c-layout.json"; \
	$${CXX:-c++} -x c++ -std=c++17 -Wall -Wextra -Werror -Iexecutor/include tests/test_executor_abi.c -o "$$d/cpp-check"; \
	"$$d/cpp-check" > "$$d/cpp-layout.json"; \
	cmp "$$d/c-layout.json" tests/fixtures/executor_abi_v5_layout.json; \
	cmp "$$d/cpp-layout.json" tests/fixtures/executor_abi_v5_layout.json; \
	echo "C11 and C++17 ABI v5 layout: PASS (equals tests/fixtures/executor_abi_v5_layout.json)"

# Host layer + CPU toy device (executor/cuda/src/{host,toy}, and the host-only research/*.cpp shared with the CUDA
# library; no CUDA, never research/*.cu): the toy libraries the Rust tests dlopen, and
# the executor/tests/host/*.cpp programs (linked with the same sources, default toy build).
HOST_CXX ?= c++
HOST_BUILD := build/host
HOST_CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Werror -pthread -Iexecutor/include -Iexecutor/cuda/src \
  -I$(HOST_BUILD)/generated
HOST_SOURCES := $(sort $(wildcard executor/cuda/src/host/*.cpp executor/cuda/src/toy/*.cpp \
  executor/cuda/src/research/*.cpp))
# The model's dimension constants, generated from its descriptor as executor/CMakeLists.txt does.
HOST_MODEL_HEADER := $(HOST_BUILD)/generated/cere_model_constants.h
HOST_HEADERS := executor/include/cere_executor.h $(wildcard executor/cuda/src/host/*.h executor/cuda/src/toy/*.h \
  executor/cuda/src/research/*.h) $(HOST_MODEL_HEADER)
HOST_TEST_HEADERS := $(HOST_HEADERS) executor/cuda/src/kernels/k4_nvfp4.h executor/cuda/src/model/mxfp8_plan.h \
  executor/cuda/src/model/plan.h executor/cuda/src/model/prefill_signature.h executor/cuda/src/model/segment.h \
  executor/cuda/src/p3b/p3b_tactic.h
HOST_EXPORTS := executor/cuda/src/host/exports.map
HOST_SOFLAGS := -fPIC -fvisibility=hidden -shared -Wl,--version-script=$(HOST_EXPORTS)
HOST_TESTS := $(patsubst executor/tests/host/%.cpp,$(HOST_BUILD)/tests/%,$(sort $(wildcard executor/tests/host/*.cpp)))
TOY_EXECUTOR := $(abspath $(HOST_BUILD)/libcere_executor_toy.so)
TOY_EXECUTOR_NOFORWARD := $(abspath $(HOST_BUILD)/libcere_executor_toy_noforward.so)
TOY_EXECUTOR_PRESTAGING := $(abspath $(HOST_BUILD)/libcere_executor_toy_prestaging.so)
# The same host sources built for the DICE-8B descriptor (family sdar), the library executor/tests/host/test_family.cpp
# loads to prove every refusal of an SDAR library by name.
SDAR_BUILD := $(HOST_BUILD)/sdar
SDAR_MODEL_HEADER := $(SDAR_BUILD)/generated/cere_model_constants.h
TOY_EXECUTOR_SDAR := $(abspath $(SDAR_BUILD)/libcere_executor_toy_sdar.so)
# The same for executor/models/inkling-2-mini.json (family inkling2_loop), whose refusals test_family.cpp proves too.
INKLING_BUILD := $(HOST_BUILD)/inkling
INKLING_MODEL_HEADER := $(INKLING_BUILD)/generated/cere_model_constants.h
TOY_EXECUTOR_INKLING := $(abspath $(INKLING_BUILD)/libcere_executor_toy_inkling.so)
# Host tests of executor/tests/host/inkling/: compiled against the Inkling header with the host sources they test, for
# code whose constants differ per family (host/mrope.h's spatial merge size).
INKLING_HOST_TESTS := $(patsubst executor/tests/host/inkling/%.cpp,$(INKLING_BUILD)/tests/%,\
  $(sort $(wildcard executor/tests/host/inkling/*.cpp)))
TOY_ENV := CERE_TOY_EXECUTOR=$(TOY_EXECUTOR) CERE_TOY_EXECUTOR_NOFORWARD=$(TOY_EXECUTOR_NOFORWARD) \
  CERE_TOY_EXECUTOR_PRESTAGING=$(TOY_EXECUTOR_PRESTAGING) CERE_TOY_EXECUTOR_SDAR=$(TOY_EXECUTOR_SDAR) \
  CERE_TOY_EXECUTOR_INKLING=$(TOY_EXECUTOR_INKLING)
TOY_LIBS := $(TOY_EXECUTOR) $(TOY_EXECUTOR_NOFORWARD) $(TOY_EXECUTOR_PRESTAGING) $(TOY_EXECUTOR_SDAR) \
  $(TOY_EXECUTOR_INKLING)

$(HOST_MODEL_HEADER): executor/tools/gen_model_header.py executor/models/qwen38-flash-next.json
	python3 executor/tools/gen_model_header.py executor/models/qwen38-flash-next.json $@

$(SDAR_MODEL_HEADER): executor/tools/gen_model_header.py executor/models/dice-8b.json
	python3 executor/tools/gen_model_header.py executor/models/dice-8b.json $@

$(TOY_EXECUTOR_SDAR): $(HOST_SOURCES) $(HOST_HEADERS) $(SDAR_MODEL_HEADER) $(HOST_EXPORTS)
	@mkdir -p $(@D)
	$(HOST_CXX) $(subst -I$(HOST_BUILD)/generated,-I$(SDAR_BUILD)/generated,$(HOST_CXXFLAGS)) $(HOST_SOFLAGS) \
	  $(HOST_SOURCES) -o $@

$(INKLING_MODEL_HEADER): executor/tools/gen_model_header.py executor/models/inkling-2-mini.json
	python3 executor/tools/gen_model_header.py executor/models/inkling-2-mini.json $@

$(TOY_EXECUTOR_INKLING): $(HOST_SOURCES) $(HOST_HEADERS) $(INKLING_MODEL_HEADER) $(HOST_EXPORTS)
	@mkdir -p $(@D)
	$(HOST_CXX) $(subst -I$(HOST_BUILD)/generated,-I$(INKLING_BUILD)/generated,$(HOST_CXXFLAGS)) $(HOST_SOFLAGS) \
	  $(HOST_SOURCES) -o $@

$(INKLING_BUILD)/tests/%: executor/tests/host/inkling/%.cpp executor/tests/host/check.h executor/cuda/src/host/mrope.cpp \
  executor/cuda/src/host/mrope.h executor/include/cere_executor.h $(INKLING_MODEL_HEADER)
	@mkdir -p $(@D)
	$(HOST_CXX) $(subst -I$(HOST_BUILD)/generated,-I$(INKLING_BUILD)/generated,$(HOST_CXXFLAGS)) $< \
	  executor/cuda/src/host/mrope.cpp -o $@

$(TOY_EXECUTOR): $(HOST_SOURCES) $(HOST_HEADERS) $(HOST_EXPORTS)
	@mkdir -p $(@D)
	$(HOST_CXX) $(HOST_CXXFLAGS) $(HOST_SOFLAGS) $(HOST_SOURCES) -o $@

$(TOY_EXECUTOR_NOFORWARD): $(HOST_SOURCES) $(HOST_HEADERS) $(HOST_EXPORTS)
	@mkdir -p $(@D)
	$(HOST_CXX) $(HOST_CXXFLAGS) -DCERE_TOY_NO_FORWARD $(HOST_SOFLAGS) $(HOST_SOURCES) -o $@

# A library from before the optional exports (put staging, decode rounds, prefix counters, prefill slice, constrained
# drafts, LoRA, decode share, research hooks): the runtime's fallback and refusal tests load it.
$(TOY_EXECUTOR_PRESTAGING): $(HOST_SOURCES) $(HOST_HEADERS) $(HOST_EXPORTS)
	@mkdir -p $(@D)
	$(HOST_CXX) $(HOST_CXXFLAGS) -DCERE_TOY_PRE_STAGING $(HOST_SOFLAGS) $(HOST_SOURCES) -o $@

$(HOST_BUILD)/tests/%: executor/tests/host/%.cpp executor/tests/host/check.h $(HOST_SOURCES) $(HOST_TEST_HEADERS)
	@mkdir -p $(@D)
	$(HOST_CXX) $(HOST_CXXFLAGS) -DCERE_HEADER_PATH='"$(abspath executor/include/cere_executor.h)"' \
	  -DCERE_LAYOUT_FIXTURE='"$(abspath tests/fixtures/executor_abi_v5_layout.json)"' $< $(HOST_SOURCES) -ldl -o $@

toy-executors: $(TOY_LIBS)

# Each library exports exactly the ABI v5 functions of tests/fixtures/executor_abi_v5_layout.json (the pre-staging
# toy: all but the optional exports listed above).
test-executor-host: $(TOY_LIBS) $(HOST_TESTS) $(INKLING_HOST_TESTS)
	@set -eu; for lib in $(TOY_LIBS); do \
	  syms=$$(nm -D --defined-only "$$lib" | awk '{print $$3}' | sort); skip=$$([ "$$lib" = $(TOY_EXECUTOR_PRESTAGING) ] && echo 1 || echo 0); \
	  want=$$(python3 -c 'import json,sys; f=json.load(open(sys.argv[1]))["functions"]; f=[s for s in f if sys.argv[2]=="0" or not (s.endswith("register_staging") or s in ("cere_executor_set_decode_rounds", "cere_model_prefix_stats", "cere_model_prefix_capture_stats", "cere_executor_set_prefill_slice", "cere_executor_round_drafts", "cere_executor_constrain_drafts", "cere_executor_set_decode_share") or "_lora" in s or "_research" in s)]; print("\n".join(sorted(f)))' \
	    tests/fixtures/executor_abi_v5_layout.json $$skip); \
	  if [ "$$syms" != "$$want" ]; then echo "$$lib exports differ from the ABI v5 functions:"; echo "$$syms"; exit 1; fi; \
	done; echo "toy libraries export exactly the ABI v5 functions: PASS"; \
	for t in $(HOST_TESTS) $(INKLING_HOST_TESTS); do $(TOY_ENV) "$$t"; done

# The Rust tests load the toy libraries through the CERE_TOY_EXECUTOR* variables and fail when they are absent.
test-rust: toy-executors
	$(TOY_ENV) cargo test --workspace --locked

check:
	cargo check --workspace

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

# License and source policy for the Rust dependency tree (deny.toml); needs `cargo install cargo-deny`.
deny:
	cargo deny check licenses bans sources

# cere-proto's build.rs runs protoc through tonic-build, so the generated bindings are compiled and their wire tests
# run here too.
test-proto:
	@set -eu; command -v protoc >/dev/null; d=$$(mktemp -d); trap 'rm -rf "$$d"' EXIT; \
	protoc --proto_path=proto --descriptor_set_out="$$d/cere-worker.pb" proto/cere_worker.proto; \
	echo "Protobuf descriptor compilation: PASS"
	cargo test -p cere-proto --locked

# The CUDA executor (sm_120a: RTX PRO 6000 Blackwell, RTX 5090). CERE_MODEL picks the model descriptor
# (executor/models/<id>.json; default qwen38-flash-next). The build records its provenance in build.json:
# CERE_IMAGE_DIGEST names the container image the build ran in (all zeros = a host build). See docs/BUILDING.md.
CERE_MODEL ?= qwen38-flash-next
CERE_IMAGE_DIGEST ?= sha256:0000000000000000000000000000000000000000000000000000000000000000
executor-cuda:
	@set -eu; head=$$(git rev-parse HEAD); dirty=$$(test -z "$$(git status --porcelain)" && echo 0 || echo 1); \
	cmake -S executor -B build/native-$(CERE_MODEL) -G Ninja -DCMAKE_BUILD_TYPE=Release -DCERE_MODEL=$(CERE_MODEL) \
	  -DCERE_IMAGE_DIGEST=$(CERE_IMAGE_DIGEST) -DCERE_GIT_HEAD=$$head -DCERE_GIT_DIRTY=$$dirty; \
	cmake --build build/native-$(CERE_MODEL)
