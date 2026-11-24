from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from cere_contracts import (ContractError,load_json,validate,require_capability,
                            require_physical_capacity,safe_relative,canonical_digest,
                            parse_loopback_base_url,validate_deployment,JSON_SCHEMA_KEYWORDS)
import pin_configs

def fixture(name):return load_json(ROOT/'examples'/f'{name}.json')

class Contracts(unittest.TestCase):
    def test_all_positive_fixtures(self):
        for p,n in load_json(ROOT/'examples/fixture_map.json').items():
            with self.subTest(p=p):validate(n,load_json(ROOT/'examples'/p))
    def test_metaschema_covers_every_schema(self):
        from jsonschema import Draft202012Validator
        names={p.name for p in (ROOT/'schemas').glob('*.schema.json')}
        for n in ('execution_artifact','artifact_families','artifact_profile','artifact_pin','artifact_inventory'):
            self.assertIn(f'{n}.schema.json',names)
        for n in sorted(names):
            with self.subTest(schema=n):
                s=load_json(ROOT/'schemas'/n);Draft202012Validator.check_schema(s)
                self.assertEqual(s['$schema'],'https://json-schema.org/draft/2020-12/schema')
    def test_artifact_configs_and_fixture_validate(self):
        for name,path in (('artifact_families','configs/artifacts/qwen4_exp-nvfp4.families.json'),
                          ('artifact_profile','configs/artifacts/profiles/rtxpro6000bw-plehost-text-v1.json'),
                          ('artifact_families','tests/fixtures/model_src.families.json'),
                          ('artifact_profile','tests/fixtures/model_src.profile.json'),
                          ('artifact_inventory','tests/fixtures/model_src.inventory.json'),
                          ('artifact_families','configs/artifacts/qwen4_exp-nvfp4-plenvfp4.families.json'),
                          ('artifact_profile','configs/artifacts/profiles/rtxpro6000bw-plehost-nvfp4-text-v1.json'),
                          ('artifact_inventory','configs/artifacts/qwen38-flash-next-fp4local-cere.inventory.json'),
                          ('artifact_families','configs/artifacts/qwen4_exp-nvfp4-plenvfp4-mxfp8.families.json'),
                          ('artifact_profile','configs/artifacts/profiles/rtxpro6000bw-plehost-nvfp4-mxfp8-text-v1.json'),
                          ('artifact_inventory','configs/artifacts/qwen38-flash-next-fp4local-cere-s3.inventory.json')):
            with self.subTest(path=path):
                x=load_json(ROOT/path);validate(name,x)
                x['typo']=True
                with self.assertRaises(ContractError):validate(name,x)
    def test_execution_artifact_v1_rejected(self):
        with self.assertRaises(ContractError):
            validate('execution_artifact',{'schema_version':'1','artifact_version':'1','files':[],'tensors':[]})
    def test_unknown_fields(self):
        for n in ('work_unit','state_object','resource_graph','execution_capabilities','worker_node','cluster_config'):
            with self.subTest(schema=n):
                x=fixture(n);x['typo']=True
                with self.assertRaises(ContractError):validate(n,x)
    def test_negative_sizes(self):
        x=fixture('state_object');x['size_bytes']=-1
        with self.assertRaises(ContractError):validate('state_object',x)
    def test_bad_hash(self):
        x=fixture('state_object');x['model_fingerprint']='latest'
        with self.assertRaises(ContractError):validate('state_object',x)
    def test_nonfinite(self):
        for v in (float('nan'),float('inf'),-float('inf')):
            x=fixture('state_object');x['reuse_probability']=v
            with self.assertRaises(ContractError):validate('state_object',x)
    def test_resource_duplicate(self):
        x=fixture('resource_graph');x['nodes'].append(copy.deepcopy(x['nodes'][0]))
        with self.assertRaises(ContractError):validate('resource_graph',x)
    def test_resource_endpoint(self):
        x=fixture('resource_graph');x['links'][0]['to']='missing'
        with self.assertRaises(ContractError):validate('resource_graph',x)
    def test_unknown_probe_cannot_admit(self):
        with self.assertRaises(ContractError):require_physical_capacity(fixture('resource_graph'),'gpu0')
    def test_physical_probe_capacity(self):
        x=fixture('resource_graph');x['evidence']='physical_probe';x['nodes'][0]['capacity_bytes']=1000
        self.assertEqual(require_physical_capacity(x,'gpu0'),1000)
    def test_missing_physical_capacity(self):
        x=fixture('resource_graph');x['evidence']='physical_probe'
        with self.assertRaises(ContractError):require_physical_capacity(x,'gpu0')
    def test_bad_batch_range(self):
        x=fixture('execution_capabilities');x['decode_batch_min']=32
        with self.assertRaises(ContractError):validate('execution_capabilities',x)
    def test_serial_not_native(self):
        x=fixture('execution_capabilities');x['execution_mode']='serial'
        with self.assertRaises(ContractError):validate('execution_capabilities',x)
    def test_unknown_capabilities(self):
        x=fixture('execution_capabilities');x['evidence']='unknown'
        with self.assertRaises(ContractError):require_capability(x,rows=1,context=128,operation='decode')
    def test_synthetic_not_real(self):
        with self.assertRaises(ContractError):require_capability(fixture('execution_capabilities'),rows=1,context=128,operation='decode')
    def test_synthetic_optin(self):
        self.assertEqual(require_capability(fixture('execution_capabilities'),rows=8,context=128,operation='decode',allow_synthetic=True),'native')
    def test_unsupported_shape_or_features(self):
        x=fixture('execution_capabilities')
        for kw in ({'rows':32},{'context':8192},{'modality':'image'},{'output':'tool_calls'},{'operation':'verify'},{'numerical_mode':'approximate'}):
            args={'rows':1,'context':128,'operation':'decode','allow_synthetic':True};args.update(kw)
            with self.subTest(kw=kw),self.assertRaises(ContractError):require_capability(x,**args)
    def test_inference_requires_session(self):
        x=fixture('work_unit');x['session_id']=None
        with self.assertRaises(ContractError):validate('work_unit',x)
    def test_decode_requires_budget(self):
        x=fixture('work_unit');x['max_new_tokens']=0
        with self.assertRaises(ContractError):validate('work_unit',x)
    def test_plan_requires_ticket(self):
        x=fixture('execution_plan');x['status']='submitted'
        with self.assertRaises(ContractError):validate('execution_plan',x)
    def test_root_event_needs_no_session(self):
        x=fixture('trace_event');x.update(event='worker_started',request_id=None,scope_id=None,session_id=None)
        validate('trace_event',x)
    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';p.write_text('{"a":1,"a":2}')
            with self.assertRaises(ContractError):load_json(p)
    def test_paths(self):
        for p in ('../a','/a','a/../b','a\\b','a//b','./a','C:a','a\x00'):
            with self.subTest(path=p),self.assertRaises(ContractError):safe_relative(p)
        self.assertEqual(str(safe_relative('tensors/weights.bin')),'tensors/weights.bin')
    def test_rust_source_field_agreement(self):
        # Source-level smoke only. Actual serde roundtrip is in the Rust test suite.
        import re
        pairs = [('resource_graph','ResourceGraph','resource_graph'),
                 ('capabilities','ExecutionCapabilities','execution_capabilities'),
                 ('state','StateObject','state_object'),('work_unit','WorkUnit','work_unit'),('worker','WorkerNode','worker_node')]
        for module,struct,schema in pairs:
            source=(ROOT/'crates/cere-core/src'/f'{module}.rs').read_text()
            body=re.search(r'pub struct '+struct+r' \{(.*?)\n\}',source,re.S).group(1)
            fields=set(re.findall(r'pub ([a-z0-9_]+):',body))
            expected=set(load_json(ROOT/'schemas'/f'{schema}.schema.json')['properties'])
            with self.subTest(struct=struct):self.assertEqual(fields,expected)


    def test_model_management_anti_thrash_config(self):
        import tomllib
        x=tomllib.loads((ROOT/'configs/runtime/dev-single-gpu.toml').read_text())
        x['model_management']['idle_unload_seconds']=10
        x['model_management']['min_resident_seconds']=20
        with self.assertRaises(ContractError):validate('runtime_config',x)

    def test_preload_cannot_exceed_slots(self):
        import tomllib
        x=tomllib.loads((ROOT/'configs/runtime/dev-single-gpu.toml').read_text())
        x['model_management']['preload_recipe_ids']=['a','b']
        with self.assertRaises(ContractError):validate('runtime_config',x)

    def test_model_recipe_rejects_direct_url(self):
        x=fixture('model_recipe');x['model_ref']='https://example.com/model'
        with self.assertRaises(ContractError):validate('model_recipe',x)

    def test_model_instance_unload_safety(self):
        x=fixture('model_instance');x['lifecycle_state']='unloading';x['active_request_count']=1
        with self.assertRaises(ContractError):validate('model_instance',x)

    def test_model_instance_failed_needs_reason(self):
        x=fixture('model_instance');x['lifecycle_state']='failed';x['resource_ids']=[];x['resident_bytes']=0
        with self.assertRaises(ContractError):validate('model_instance',x)

    def test_json_schema_capability_is_explicit(self):
        x=fixture('execution_capabilities');x['structured_outputs'].append('json_schema')
        with self.assertRaises(ContractError):validate('execution_capabilities',x)
        x['json_schema_dialect']='https://json-schema.org/draft/2020-12/schema';x['json_schema_keywords']=['type','properties','required']
        with self.assertRaises(ContractError):validate('execution_capabilities',x)
        x['json_schema_dialect']='json-schema-2020-12/xgrammar-0.2.1'
        validate('execution_capabilities',x)
        for bad in (['type','oneOf'],['type','not'],[]):
            with self.subTest(keywords=bad),self.assertRaises(ContractError):
                validate('execution_capabilities',dict(x,json_schema_keywords=bad))


    def test_worker_lost_cannot_advertise_models(self):
        x=fixture('worker_node');x['lifecycle_state']='lost';x['model_instance_ids']=['m1']
        with self.assertRaises(ContractError):validate('worker_node',x)

    def test_cluster_threshold_order(self):
        x=fixture('cluster_config');x['control_plane']['suspect_after_ms']=x['control_plane']['heartbeat_interval_ms']
        with self.assertRaises(ContractError):validate('cluster_config',x)
        x=fixture('cluster_config');x['control_plane']['lost_after_ms']=x['control_plane']['suspect_after_ms']
        with self.assertRaises(ContractError):validate('cluster_config',x)

    def test_canonical_digest(self):
        self.assertEqual(canonical_digest({'a':1,'b':2}),canonical_digest({'b':2,'a':1}))
        self.assertNotEqual(canonical_digest({'a':1}),canonical_digest({'a':2}))

import tomllib
LIVE_TOML='configs/runtime/sglang-t4-qwen38.toml'

def merge_patch(target,patch):
    """RFC 7396 JSON merge patch; `${REPO_ROOT}` in strings becomes the repository root."""
    if isinstance(patch,str):return patch.replace('${REPO_ROOT}',str(ROOT))
    if not isinstance(patch,dict):return copy.deepcopy(patch)
    out=copy.deepcopy(target) if isinstance(target,dict) else {}
    for k,v in patch.items():
        if v is None:out.pop(k,None)
        else:out[k]=merge_patch(out.get(k),v)
    return out

def runtime_toml(rel):return tomllib.loads((ROOT/rel).read_text())

def case_config(case):
    """One shared-table case (M5 D4): `patch` merged into the base, then `entry_patch` into `deployments[0]`. A
    `patch` with a top-level `backend` key is the pre-M5 shape and refused. Rust's tests/config_rev2.rs does the same."""
    if 'backend' in case['patch']:raise ValueError(f"case {case['name']}: patch.backend is the pre-M5 shape; use entry_patch")
    x=merge_patch(runtime_toml(case['base']),case['patch'])
    if 'entry_patch' in case:x['deployments'][0]=merge_patch(x['deployments'][0],case['entry_patch'])
    return x

def live_with(entry_patch,patch=None):
    """The T4 live config with `entry_patch` merged into its single deployment and `patch` into the document."""
    return case_config({'name':'live_with','base':LIVE_TOML,'patch':patch or {},'entry_patch':entry_patch})

def patched_deployment_root(tmp,patches,config):
    """Copy schemas/ and configs/deployments/ under `tmp`, merge-patch the parts of the deployment `config` names in
    deployments[0].deployment and re-pin the recipe bytes in the lock (unless the lock patch sets recipe_sha256), so
    only the patched constraint is under test. Rust's tests/config_rev2.rs builds the same tree."""
    import hashlib,shutil
    root=Path(tmp);shutil.copytree(ROOT/'schemas',root/'schemas')
    shutil.copytree(ROOT/'configs/deployments',root/'configs/deployments')
    prefix=root/config['deployments'][0]['deployment']
    for part in ('capabilities','recipe','lock'):
        p=Path(f'{prefix}.{part}.json');x=merge_patch(load_json(p),patches.get(part,{}))
        if part=='lock' and 'recipe_sha256' not in patches.get('lock',{}):
            x['recipe_sha256']=hashlib.sha256(Path(f'{prefix}.recipe.json').read_bytes()).hexdigest()
        p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
    return root

class RuntimeConfigRev2(unittest.TestCase):
    def test_every_shipped_runtime_toml_is_valid(self):
        paths=sorted((ROOT/'configs/runtime').glob('*.toml'))
        self.assertEqual([p.name for p in paths],sorted(['dev-single-gpu.toml','dev-two-replica.toml','amd-flashnext.toml','amd-serve-spec.toml','amd-serve.toml','federation-dev.toml','federation-flashnext-amd.toml','federation-gemma-amd.toml','federation-gemma.toml','multimodel-dev.toml','multimodel-g0g1.toml','native-2gpu.toml','native-amd-inkling-vk.toml','native-amd-s3-spec.toml','native-amd-s3-vk-spec.toml','native-amd-s3-vk.toml','native-amd-s3.toml','native-g0-dice8b-gemm-changing.toml','native-g0-dice8b.toml','native-g0-inkling-batch.toml','native-g0-inkling-plain.toml','native-g0-inkling-s13.toml','native-g0-inkling.toml','native-g0-k2s.toml','native-g0-p4.toml','native-g0-prefix.toml','native-g0-rows16-v58.toml','native-g0-sessions-v53.toml','native-g0-sessions-v59.toml','native-g0-sessions.toml','native-g0-spec-262k.toml','native-g0-spec-chains.toml','native-g0-spec-eval.toml','native-g0-spec-fp4local-eval.toml','native-g0-spec-fp4local-s3-eval.toml','native-g0-spec-fp4local-s3.toml','native-g0-spec-fp4local.toml','native-g0-spec-fp8w.toml','native-g0-spec-grammar.toml','native-g0-spec-k2s.toml','native-g0-spec-lookup.toml','native-g0-spec-lora.toml','native-g0-spec-nvfp4.toml','native-g0-spec-research.toml','native-g0-spec-tree.toml','native-g0-spec-v22.toml','native-g0-spec-v47.toml','native-g0-spec-v51.toml','native-g0-spec-v52.toml','native-g0-spec-v56.toml','native-g0-spec-vision-v54.toml','native-g0-spec-vision-v55.toml','native-g0-spec-vision.toml','native-g0-spec.toml','native-g0-vision.toml','native-g0.toml','native-g1-prefix.toml','native-g1.toml','prod-t4.toml','sglang-bench-2gpu.toml','sglang-bench-g0-replicas.toml','sglang-bench-g0.toml','sglang-t4-qwen38.toml']))
        for p in paths:
            with self.subTest(p=p.name):validate('runtime_config',tomllib.loads(p.read_text()))
    def test_expect_without_nullable_pins_is_valid(self):
        five=('quantization','speculative_algorithm','speculative_num_steps','speculative_eagle_topk','speculative_num_draft_tokens')
        validate('runtime_config',live_with({'expect':{k:None for k in five}}))
    def test_every_shipped_worker_toml_is_valid(self):
        paths=sorted((ROOT/'configs/worker').glob('*.toml'))
        self.assertEqual([p.name for p in paths],sorted(['node-b-gemma-managed.toml','node-b-gemma.toml','node-b-managed.toml','node-b-mock.toml']))
        for p in paths:
            with self.subTest(p=p.name):validate('worker_agent_config',tomllib.loads(p.read_text()))
    def test_shared_invalid_table_rejected(self):
        """Each row gets its verdict: `expect: invalid` refused, `expect: valid` (a boundary case only the specified
        rule admits, e.g. the D2(j) per-entry sum) accepted. Rust's tests/config_rev2.rs runs the same table."""
        cases=load_json(ROOT/'tests/fixtures/runtime_config_invalid.json')
        self.assertGreaterEqual(len(cases),40)
        self.assertEqual(len({c['name'] for c in cases}),len(cases))
        for c in cases:
            with self.subTest(case=c['name']):
                self.assertEqual(set(c)-{'deployment_patch','entry_patch'},{'name','base','patch','expect','reason'})
                self.assertIn(c['expect'],('invalid','valid'))
                self.assertTrue(c['base'].startswith('configs/runtime/'))
                x=case_config(c)
                with tempfile.TemporaryDirectory() as d:
                    root=patched_deployment_root(d,c['deployment_patch'],x) if 'deployment_patch' in c else ROOT
                    if c['expect']=='valid':validate('runtime_config',x,root);continue
                    with self.assertRaises(ContractError):validate('runtime_config',x,root)
    def test_visionlong_c3_image_combinations_at_the_text_shape(self):
        """VisionLong C3: the -v29 text profile plus image combinations at its shape (decode text/tool_calls and
        verify text, greedy and sampled, rows 1-4, 200,000 tokens; image prefill) passes the capability rules,
        the D5 twin rule included; an image decode row without its text twin's rows is still refused."""
        caps=load_json(ROOT/'configs/deployments/qwen38-flash-next-native-spec.capabilities.json')
        caps['modalities']=['text','image']
        images=[dict(c,modality='image') for c in caps['validated_combinations']
                if c['operation']=='prefill' or c['output'] in ('text','tool_calls') and c['operation'] in ('decode','verify')]
        self.assertEqual(sorted((c['operation'],c['output'],c['sampler_mode']) for c in images),
                         [('decode','text','greedy'),('decode','text','sampled'),('decode','tool_calls','greedy'),
                          ('decode','tool_calls','sampled'),('prefill','text','not_applicable'),
                          ('verify','text','greedy'),('verify','text','sampled'),('verify','tool_calls','greedy'),
                          ('verify','tool_calls','sampled')])
        self.assertTrue(all(c['max_context_tokens']==200000 for c in images))
        caps['validated_combinations']+=images
        validate('execution_capabilities',caps)
        bad=copy.deepcopy(caps)
        for c in bad['validated_combinations']:
            if (c['modality'],c['operation'],c['output'],c['sampler_mode'])==('image','decode','text','sampled'):
                c['max_context_tokens']=100000
        with self.assertRaisesRegex(ContractError,'image decode combination lacks the matching text'):
            validate('execution_capabilities',bad)
    def test_c9_capability_cases_hit_their_rule(self):
        """M2.4 C9: the D13 twin rule and the D12 max_top_logprobs range refuse their rows; M2.7 C9: the D11
        sampling_params names and a native sampled row without sampling_params; M2.13 C10: the D5 image twin rule and
        modalities consistency (Rust: config_rev2.rs)."""
        cases={c['name']:c for c in load_json(ROOT/'tests/fixtures/runtime_config_invalid.json')}
        rules={'capabilities_tool_calls_without_text':'tool_calls decode combination lacks the matching text decode combination',
               'capabilities_max_top_logprobs_range':"['max_top_logprobs']: 21 is greater than the maximum of 20",
               'capabilities_sampling_params_unknown':"'typical_p' is not one of",
               'capabilities_sampled_without_sampler':'native sampled combination lacks sampling_params',
               # M2.13 C10 (D5): the image twin rule and `modalities` consistency.
               'capabilities_image_without_text':'image decode combination lacks the matching text decode combination',
               'capabilities_image_modality_without_combination':'modalities lists image without an image decode combination'}
        for name,needle in rules.items():
            with self.subTest(case=name),tempfile.TemporaryDirectory() as d:
                c=cases[name]
                with self.assertRaises(ContractError) as caught:
                    x=case_config(c);validate('runtime_config',x,patched_deployment_root(d,c['deployment_patch'],x))
                self.assertIn(needle,str(caught.exception))
    def test_m212_c10_capability_cases_hit_their_rule(self):
        """M2.12 C10 (D2): token_controls items, thinking_toggle type, vocab_size >= 1 integer; BC1 leftover:
        prefill_signature 64 lowercase hex (Rust: config_rev2.rs)."""
        cases={c['name']:c for c in load_json(ROOT/'tests/fixtures/runtime_config_invalid.json')}
        rules={'capabilities_token_controls_unknown':"['token_controls', 1]: 'logit_bias' is not one of",
               'capabilities_thinking_toggle_non_bool':"['thinking_toggle']: 'yes' is not of type 'boolean'",
               'capabilities_vocab_size_zero':"['vocab_size']: 0 is less than the minimum of 1",
               'capabilities_vocab_size_non_integer':"['vocab_size']: 248320.5 is not of type 'integer'",
               # BC1 leftover: the prefill selection signature is 64 lowercase hex.
               'capabilities_prefill_signature_not_hex':"['prefill_signature']: 'ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAB' does not match"}
        for name,needle in rules.items():
            with self.subTest(case=name),tempfile.TemporaryDirectory() as d:
                c=cases[name]
                with self.assertRaises(ContractError) as caught:
                    x=case_config(c);validate('runtime_config',x,patched_deployment_root(d,c['deployment_patch'],x))
                self.assertIn(needle,str(caught.exception))
    def test_case_with_patch_backend_is_refused(self):
        """M5 D4: a case patching the pre-M5 top-level `backend` table is refused by the reader."""
        with self.assertRaisesRegex(ValueError,'patch.backend'):
            case_config({'name':'old_shape','base':LIVE_TOML,'patch':{'backend':{'base_url':None}},'expect':'invalid','reason':'r'})
        x=case_config({'name':'new_shape','base':LIVE_TOML,'patch':{},'entry_patch':{'base_url':None},'expect':'invalid','reason':'r'})
        self.assertNotIn('base_url',x['deployments'][0]);self.assertIn('served_model_name',x['deployments'][0])
    def test_backend_table_is_an_unknown_field(self):
        """M5 D1: no `[backend]` alias; the extra key itself is refused, `deployments` stays valid."""
        x=runtime_toml(LIVE_TOML);x['backend']=copy.deepcopy(x['deployments'][0])
        with self.assertRaisesRegex(ContractError,"'backend'"):validate('runtime_config',x)
    def test_multimodel_dev_round_trips_its_canonical_hash(self):
        """C2: the two-entry mock config validates and its canonical digest equals its pin (tests/fixtures/
        config_pins.json, which the Rust config_hash test reads too)."""
        rel='configs/runtime/multimodel-dev.toml'
        x=runtime_toml(rel)
        validate('runtime_config',x)
        self.assertEqual(len(x['deployments']),2)
        self.assertEqual(canonical_digest(x),pin_configs.load()['config_hash'][rel])
    def test_unpatched_deployment_root_is_valid(self):
        with tempfile.TemporaryDirectory() as d:
            x=runtime_toml(LIVE_TOML);validate('runtime_config',x,patched_deployment_root(d,{},x))
    def test_live_profile_pins(self):
        x=runtime_toml(LIVE_TOML);b=x['deployments'][0]
        self.assertEqual((b['kind'],b['base_url'],b['served_model_name'],b['shared_backend_tenancy']),
                         ('sglang_http','http://127.0.0.1:18080','Qwen3.8-Flash-Next','shared_acknowledged'))
        self.assertEqual(x['admission']['max_backend_inflight'],2)
        self.assertEqual(x['auth']['scopes'],[{'scope_id':'dev','token_env':'CERE_TOKEN_DEV'}])
        self.assertNotIn('chat_template',b['expect'])
        self.assertEqual((b['expect']['context_length'],b['expect']['max_running_requests']),(200000,4))
    def test_loopback_base_url_parse(self):
        for u,want in (('http://127.0.0.1:18080',('127.0.0.1',18080)),('http://127.255.0.9:1',('127.255.0.9',1)),
                       ('http://[::1]:65535',('::1',65535))):
            with self.subTest(u=u):self.assertEqual(parse_loopback_base_url(u),want)
        for u in ('http://127.0.0.1:65536','http://127.0.0.1:018080','http://127.1:80','http://[::1]','http://[0:0:0:0:0:0:0:1]:80',
                  'http://127.0.0.1:80\n','http://127.0.0.1:８０'):
            with self.subTest(u=u),self.assertRaises(ContractError):parse_loopback_base_url(u)
    def test_live_config_two_scopes_with_acknowledged_cross_scope_cache(self):
        x=live_with({'allow_cross_scope_prefix_cache':True},{'auth':{'scopes':[
            {'scope_id':'dev','token_env':'CERE_TOKEN_DEV'},{'scope_id':'ci','token_env':'CERE_TOKEN_CI'}]}})
        validate('runtime_config',x)
    def test_two_scopes_with_cache_salt_isolation_valid(self):
        x=live_with({'prefix_cache_isolation':'cache_salt','allow_cross_scope_prefix_cache':False},
            {'auth':{'scopes':[{'scope_id':'dev','token_env':'CERE_TOKEN_DEV'},{'scope_id':'ci','token_env':'CERE_TOKEN_CI'}]}})
        validate('runtime_config',x)
        x['deployments'][0]['prefix_cache_isolation']='none'
        with self.assertRaises(ContractError):validate('runtime_config',x)
    def test_single_scope_isolation_none_valid(self):
        validate('runtime_config',live_with({'prefix_cache_isolation':'none'}))
    def test_bench_profile_pins(self):
        x=runtime_toml('configs/runtime/sglang-bench-g0.toml');b=x['deployments'][0]
        self.assertEqual((b['base_url'],b['prefix_cache_isolation'],b['allow_cross_scope_prefix_cache']),
                         ('http://127.0.0.1:18308','cache_salt',False))
        self.assertTrue(b['expect']['enable_cache_report'])
        self.assertEqual(x['server']['bind'],'127.0.0.1:18391')
        self.assertEqual(x['auth']['scopes'],[{'scope_id':'bench-a','token_env':'CERE_TOKEN_BENCH_A'},
                                              {'scope_id':'bench-b','token_env':'CERE_TOKEN_BENCH_B'}])
        self.assertEqual(x['admission'],{'max_backend_inflight':4,'per_scope_inflight':4,'queue_depth':16})
        self.assertEqual(x['scheduler'],{'mode':'replica','devices':['gpu0']})
        self.assertNotEqual(x['ledger']['state_dir'],runtime_toml(LIVE_TOML)['ledger']['state_dir'])
    def test_capability_rows_one_to_four(self):
        p=load_json(ROOT/'configs/deployments/qwen38-flash-next.capabilities.json')
        self.assertEqual((p['decode_batch_max'],p['evidence']),(4,'declared'))
        for c in p['validated_combinations']:self.assertEqual(c['row_counts'],[1,2,3,4])
    def test_sglang_profile_advertises_grammar(self):
        """M2.8 D10: the four grammar outputs, each with greedy and sampled decode, the D5 dialect and keywords."""
        p=load_json(ROOT/'configs/deployments/qwen38-flash-next.capabilities.json')
        validate('execution_capabilities',p)
        grammar=['json_schema','regex','ebnf','tool_calls_constrained']
        self.assertEqual(set(p['structured_outputs']),{'text','tool_calls',*grammar})
        self.assertEqual(p['json_schema_dialect'],'json-schema-2020-12/xgrammar-0.2.1')
        self.assertTrue(set(p['json_schema_keywords'])<=set(JSON_SCHEMA_KEYWORDS))
        text={c['sampler_mode']:c for c in p['validated_combinations'] if (c['operation'],c['output'])==('decode','text')}
        for output in grammar:
            for mode in ('greedy','sampled'):
                with self.subTest(output=output,mode=mode):
                    rows=[c for c in p['validated_combinations'] if (c['operation'],c['output'],c['sampler_mode'])==('decode',output,mode)]
                    self.assertEqual(len(rows),1)
                    self.assertEqual((rows[0]['row_counts'],rows[0]['max_context_tokens']),(text[mode]['row_counts'],text[mode]['max_context_tokens']))
        self.assertFalse(any(c['output'] in grammar and c['operation']!='decode' for c in p['validated_combinations']))
    def test_grammar_keywords_and_formats_follow_the_c4_matrix(self):
        """M2.8 C4/D5: the SGLang profile's keywords are the C4 passing keywords, the annotations and the integer
        bounds the walker value-restricts; the Rust ENFORCED_FORMATS are exactly the C4 passing formats."""
        import re
        matrix=load_json(ROOT/'tests/fixtures/native/grammar_keywords.json')
        p=load_json(ROOT/'configs/deployments/qwen38-flash-next.capabilities.json')
        self.assertEqual(p['json_schema_dialect'],matrix['dialect'])
        bounds=set(matrix['value_restricted_keywords'])
        self.assertEqual(bounds,{'minimum','maximum','exclusiveMinimum','exclusiveMaximum'})
        # constraint.rs bounds_enforced: type "integer", exactly one upper bound, at most one lower bound.
        upper,lower=('maximum','exclusiveMaximum'),('minimum','exclusiveMinimum')
        rust=sorted(sorted([u,*l]) for u in upper for l in ([],*([x] for x in lower)))
        self.assertEqual(rust,sorted(sorted(f) for f in matrix['bound_forms']['integer']))
        self.assertEqual(set(p['json_schema_keywords']),set(matrix['keywords'])|set(matrix['annotations'])|bounds)
        self.assertFalse(set(p['json_schema_keywords'])&(set(matrix['failed_keywords'])-bounds))
        source=(ROOT/'crates/cere-runtime/src/constraint.rs').read_text()
        block=re.search(r'pub const ENFORCED_FORMATS: &\[&str\] = &\[(.*?)\];',source,re.S).group(1)
        self.assertEqual(sorted(re.findall(r'"([^"]+)"',block)),sorted(matrix['formats']))
    def test_deployment_identity(self):
        import hashlib
        d=validate_deployment(ROOT,'configs/deployments/qwen38-flash-next')
        self.assertEqual(d['recipe']['recipe_id'],'qwen38-flash-next')
        self.assertEqual(d['profile']['execution_mode'],'backend_managed')
        lock=(ROOT/'configs/deployments/qwen38-flash-next.lock.json').read_bytes()
        self.assertEqual(d['lock_sha256'],hashlib.sha256(lock).hexdigest())
    def test_deployment_tamper_rejected(self):
        import shutil
        def copy_root(d):
            root=Path(d);shutil.copytree(ROOT/'schemas',root/'schemas')
            shutil.copytree(ROOT/'configs/deployments',root/'configs/deployments');return root
        edits={'recipe':('context_tokens',131072),'lock':('backend_profile_id','other'),'capabilities':('artifact_sha256','0'*64)}
        for part,(key,value) in edits.items():
            with self.subTest(part=part),tempfile.TemporaryDirectory() as d:
                root=copy_root(d);p=root/f'configs/deployments/qwen38-flash-next.{part}.json'
                x=load_json(p);x[key]=value;p.write_text(json.dumps(x,indent=2)+'\n')
                with self.assertRaises(ContractError):validate_deployment(root,'configs/deployments/qwen38-flash-next')
        with tempfile.TemporaryDirectory() as d:
            root=copy_root(d);(root/'configs/deployments/qwen38-flash-next.lock.json').unlink()
            with self.assertRaises(ContractError):validate_deployment(root,'configs/deployments/qwen38-flash-next')
class CapabilityParity(unittest.TestCase):
    def test_shared_capability_case_table(self):
        table=load_json(ROOT/'crates/cere-core/tests/fixtures/capabilities_cases.json')
        base=load_json(ROOT/table['base'])
        for case in table['cases']:
            doc=copy.deepcopy(base);doc.update(copy.deepcopy(case['set']))
            with self.subTest(case=case['name']):
                if case['valid']:validate('execution_capabilities',doc)
                else:
                    with self.assertRaises(ContractError):validate('execution_capabilities',doc)
class ModelInstanceReferences(unittest.TestCase):
    def failed(self):
        x=fixture('model_instance');x.update(lifecycle_state='failed',failure_reason='partial',cleanup_pending=True,lease_count=0,resident_bytes=0)
        validate('model_instance',x);return x
    def test_refs_rejected_while_loading_or_failed(self):
        for state in ('preparing','loading','failed'):
            for field in ('lease_count','active_request_count'):
                x=self.failed();x.update(lifecycle_state=state,cleanup_pending=state=='failed',failure_reason='partial' if state=='failed' else None)
                x[field]=1
                with self.subTest(state=state,field=field),self.assertRaises(ContractError):validate('model_instance',x)
    def test_reservation_released_after_acknowledged_cleanup(self):
        x=self.failed();x['cleanup_pending']=False
        with self.assertRaises(ContractError):validate('model_instance',x)
        x['reserved_bytes']=0;x['resource_ids']=[];validate('model_instance',x)
if __name__=='__main__':unittest.main()
