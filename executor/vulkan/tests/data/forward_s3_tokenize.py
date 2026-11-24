import sys, json
from transformers import AutoTokenizer
art = sys.argv[1]; doc = open(sys.argv[2]).read(); out = sys.argv[3]
tok = AutoTokenizer.from_pretrained(art)
def chat(text):
    r = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                                enable_thinking=False)
    return list(r if isinstance(r, list) else r["input_ids"])
prompts = [
    ("capital", "What is the capital of France? Answer in one sentence."),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively."),
    ("explain", "Explain in three sentences why the sky is blue."),
]
rows = []
for name, text in prompts:
    ids = chat(text); ids = ids if isinstance(ids, list) else ids["input_ids"]
    rows.append((name, ids))
# long prompt: a document truncated so the whole chat prompt is ~3,000 tokens (crosses 2,048 positions)
body = tok(doc, add_special_tokens=False)["input_ids"]
lo, hi = 100, len(body)
while lo < hi:
    mid = (lo + hi + 1) // 2
    t = tok.decode(body[:mid])
    n = len(chat("Summarize the following document in five bullet points.\n\n" + t))
    if n <= 3000: lo = mid
    else: hi = mid - 1
ids = chat("Summarize the following document in five bullet points.\n\n" + tok.decode(body[:lo]))
ids = ids if isinstance(ids, list) else ids["input_ids"]
rows.append(("long", ids))
with open(out, "w") as f:
    f.write("# forward_s3 prompts: name count ids... (tokenized with the S3 artifact's tokenizer, chat template, enable_thinking=False)\n")
    for name, ids in rows:
        f.write(name + " " + str(len(ids)) + " " + " ".join(map(str, ids)) + "\n")
for name, ids in rows: print(name, len(ids), ids[:12], tok.decode(ids[-8:]))
import transformers, tokenizers; print("transformers", transformers.__version__, "tokenizers", tokenizers.__version__)
