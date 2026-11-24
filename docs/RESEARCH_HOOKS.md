# Research hooks (activation capture, interventions, steering)

The CUDA executor and `cered` carry opt-in research hooks (internally "WeightLab"): they record the model's
activations at named points and can edit them in place while a request runs. They are **off by default**: no shipped
production profile enables them, and without the config table the routes do not exist and every request runs the
normal code path.

## What they do

- **Capture.** A plan names capture points (for example the residual state entering a layer's attention or MoE block,
  a block's output, the attention/GDN projections, the MoE router logits and the chosen experts with their gate
  weights, the final mix and the logits), the layers, and the positions (prompt ranges and/or generated tokens).
  After the request finishes its captures are listed under `GET /v1/cere/research/captures` and downloaded once as a
  safetensors file (tensor names `<point>/L<layer>`, one row per captured position), then freed.
- **Interventions.** A plan may also edit activations at the same points: `add` (steering: `y + alpha * v` for an
  uploaded vector `v`), `ablate_zero`, `ablate_mean`, `replace` (rows from an uploaded tensor), and on the MoE
  routing `expert_bias`, `expert_mask` and `expert_force`. Vectors are uploaded as safetensors through
  `POST /v1/cere/research/vectors` and live until `cered` restarts.
- **Binding.** A plan is bound to a client scope (`[[auth.scopes]]`); every request of that scope runs under it and
  its reply carries `X-Cere-Research-Plan: <plan_sha256>`. Requests of other scopes are unaffected.

Under speculation only accepted positions are captured or edited; draft tokens never are. Planned rows run eagerly
(no decode graph), their sessions are never snapshotted or preempted, and an intervening plan gets its own
prefix-cache namespace so no cached state crosses between a plan and the base model.

## Supported models

Qwen3.8-Flash-Next on the CUDA executor only. Libraries built for other model families (Inkling 2 Mini, DICE-8B) and
the Vulkan and HIP executors refuse a deployment with the research table at load time; the config loader also refuses
it together with `replicas`.

## Enabling them

Add the table to a native deployment in the runtime config (all five keys are required; every key needs a restart):

```toml
[deployments.native.research]
device_staging_bytes = 67108864           # device capture staging, reserved at model load (counted against VRAM)
host_buffer_bytes = 1073741824            # pinned host ring shared by all undrained captures
max_capture_bytes_per_request = 268435456 # plans whose worst-case capture exceeds this are refused
max_plans = 8                             # registered plans
max_vectors = 64                          # uploaded vectors
```

`configs/runtime/native-g0-spec-research.toml` is a complete example. With the table present `cered` additionally
requires an operator token (`[auth] operator_token_env`): every `/v1/cere/research/*` route takes the operator bearer
only. It also refuses to start on the production port constant (`18080`) or on the GPU named by the production marker
file (`CERE_GPU1_PRODUCTION_MARKER`), so a research profile cannot be started in place of a production one by
accident.

The full route reference (plans, bindings, captures, vectors, every refusal and its status code) is
[API.md "Research (WeightLab)"](API.md#research-weightlab); the config keys and their ranges are in
[CONFIGURATION.md](CONFIGURATION.md).

## Cost when unused

- **Without the table** nothing is reserved and the routes are not registered (`404`); the executor is called with a
  null research binding, so the forward pass is the normal one.
- **With the table but no plan bound to a request's scope**, that request is bit-identical to the same request on the
  profile without the table: in the project's validation every shipped CUDA profile tested unbound gave bit-identical
  outputs, and capture-only and no-op plans gave the same replies as unbound requests. The measured throughput
  cost of having the hooks compiled in and unused was within 1 % (worst cell -0.68 %).
- `alpha = 0` and an empty expert mask leave every bit unchanged.

The executor-side tests are `executor/cuda/src/ktest/research_ktest.cpp` and
`executor/tests/gpu/test_kernels_research_patch.py` (GPU); the runtime's route and refusal tests
(`crates/cere-runtime/tests/server_native_research.rs`) run on CPU against the toy executor.
