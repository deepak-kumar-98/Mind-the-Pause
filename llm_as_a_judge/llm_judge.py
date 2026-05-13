# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# import csv
# import json
# import re
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Optional

# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams

# # ------------- Configurable label set -------------
# LABELS = {"Win", "Loss", "Draw"}
# LANG_MAP = {"hi": "Hindi", "mr": "Marathi", "bn": "Bengali"}

# # ------------- Judge prompts (no scores) -------------
# JUDGE_SYSTEM_PROMPT = """You are an expert in disfluency correction for ASR outputs in Indo-Aryan languages (Hindi, Marathi, Bengali).
# You will ALWAYS be given a gold/reference fluent sentence. Treat it as the ground truth for meaning/adequacy.
# Compare TWO candidate corrections (Answer A and Answer B) for the SAME disfluent input.

# Judge ONLY on:
# 1) Fluency (grammaticality, naturalness),
# 2) Adequacy (faithfulness to the reference fluent sentence),
# 3) Disfluency removal quality (removes fillers, repetitions, false starts).

# Return STRICT JSON with exactly these keys:
# {
#     "rationale": "brief why",
#     "final_verdict": "A|B|Draw",
  
# }
# IMPORTANT: The FIRST character of your response MUST be '{' and the LAST character MUST be '}'. Do NOT include any extra text before or after the JSON.
# """

# USER_PROMPT_TEMPLATE = """Language: {lang_name}
# Disfluent ASR sentence:
# {disfluent}

# Reference fluent (gold):
# {reference_fluent}

# Answer A:
# {answer_a}

# Answer B:
# {answer_b}
# """

# # ------------- Small helpers -------------
# def read_lines(path: str) -> List[str]:
#     with open(path, "r", encoding="utf-8") as f:
#         return [line.rstrip("\n\r") for line in f]

# def lang_code_to_name(code: str) -> str:
#     return LANG_MAP.get((code or "").strip().lower(), "Unknown")

# def normalize_label(x: str) -> str:
#     x = (x or "").strip().title()
#     if x in {"Loser", "Losser"}:
#         return "Loss"
#     return x

# def extract_last_json(text: str) -> Optional[dict]:
#     """Return the LAST well-formed top-level JSON object found in `text`."""
#     candidates = []
#     in_str = False
#     esc = False
#     depth = 0
#     start = None
#     for i, ch in enumerate(text):
#         if in_str:
#             if esc:
#                 esc = False
#             elif ch == "\\":
#                 esc = True
#             elif ch == '"':
#                 in_str = False
#         else:
#             if ch == '"':
#                 in_str = True
#             elif ch == "{":
#                 if depth == 0:
#                     start = i
#                 depth += 1
#             elif ch == "}":
#                 if depth > 0:
#                     depth -= 1
#                     if depth == 0 and start is not None:
#                         candidates.append(text[start:i + 1])
#     for s in reversed(candidates):
#         try:
#             return json.loads(s)
#         except Exception:
#             continue
#     return None

# def coerce_json_block(text: str) -> str:
#     """
#     Make a best-effort valid JSON object string from model output:
#     - strip code fences
#     - ensure it starts with '{' and ends with '}'
#     - remove trailing commas before the final '}'
#     """
#     t = (text or "").strip()

#     # strip code fences
#     if t.startswith("```"):
#         t = t.strip("`")
#         if t.lower().startswith("json"):
#             t = t[4:].lstrip()

#     # ensure opening brace
#     if not t.startswith("{"):
#         t = "{\n" + t

#     # trim right, remove trailing comma at very end
#     t = t.rstrip()
#     if t.endswith(","):
#         t = t[:-1]

#     # ensure closing brace
#     if not t.endswith("}"):
#         t = t + "\n}"

#     # remove any trailing commas before }
#     t = re.sub(r",\s*}", "}", t)
#     return t

# def salvage_json_to_verdict(text: str) -> Optional[dict]:
#     """
#     Try to coerce near-JSON or label-y text into a verdict dict.
#     Returns dict or None.
#     """
#     if not text:
#         return None

#     t = text

#     # If braces exist, keep inner slice
#     if "{" in t and "}" in t:
#         t = t[t.find("{"): t.rfind("}") + 1]

#     # Normalize unicode quotes and single->double for keys/values
#     t = t.replace("’", "'").replace("“", '"').replace("”", '"')
#     # keys: 'key': -> "key":
#     t = re.sub(r"([{,]\s*)'([^']+?)'\s*:", r'\1"\2":', t)
#     # values 'Win'/'Loss'/'Draw' -> "Win"/"Loss"/"Draw"
#     t = re.sub(r':\s*\'(Win|Loss|Draw)\'', lambda m: ':"%s"' % m.group(1), t, flags=re.I)
#     # remove trailing commas before }
#     t = re.sub(r",\s*}", "}", t)

#     try:
#         d = json.loads(t)
#         return d
#     except Exception:
#         pass

#     # Regex fallback for free-text patterns
#     lab = r"(win|loss|draw)"
#     mA = re.search(r"answer[\s_]*a[^:]*:\s*"+lab, t, flags=re.I)
#     mB = re.search(r"answer[\s_]*b[^:]*:\s*"+lab, t, flags=re.I)
#     mV = re.search(r"(final[_\s-]*verdict|verdict)[^:]*:\s*(A|B|Draw)", t, flags=re.I)

#     def norm_lab(s: str) -> str:
#         return {"win": "Win", "loss": "Loss", "draw": "Draw"}.get(s.lower(), "Draw")

#     out = {}
#     # if mA: out["answer_A"] = norm_lab(mA.group(1))
#     # if mB: out["answer_B"] = norm_lab(mB.group(1))
#     if mV: out["final_verdict"] = mV.group(2).title()

#     if "final_verdict" not in out and "answer_A" in out and "answer_B" in out:
#         A, B = out["answer_A"], out["answer_B"]
#         if A == "Win" and B == "Loss":
#             out["final_verdict"] = "A"
#         elif A == "Loss" and B == "Win":
#             out["final_verdict"] = "B"
#         else:
#             out["final_verdict"] = "Draw"

#     # If we at least recovered a final verdict, return it.
#     if "final_verdict" in out:
#         return out
#     return None

# def ensure_dir(p: Path):
#     p.mkdir(parents=True, exist_ok=True)

# # ------------- CLI / main -------------
# @dataclass
# class Args:
#     model: str
#     disfluent: str
#     fluent: str
#     cand_a: str
#     cand_b: str
#     default_lang: str
#     lang_file: Optional[str]
#     out_csv: str
#     out_jsonl: str
#     tp: int
#     dtype: str
#     max_tokens: int
#     temperature: float
#     gpu_util: float
#     max_model_len: int
#     debug: bool

# def parse_args() -> Args:
#     ap = argparse.ArgumentParser(description="LLM-as-a-Judge using vLLM (batched generate).")
#     ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
#                     help="HF model id or local path (chat-instruct model recommended).")
#     ap.add_argument("--disfluent", required=True)
#     ap.add_argument("--fluent", required=True, help="Gold/reference fluent (mandatory; no empty lines).")
#     ap.add_argument("--cand_a", required=True)
#     ap.add_argument("--cand_b", required=True)
#     ap.add_argument("--default_lang", default="hi", choices=["hi", "mr", "bn"])
#     ap.add_argument("--lang_file", default=None, help="Optional per-line language codes (hi/mr/bn).")
#     ap.add_argument("--out_csv", default="judgements.csv")
#     ap.add_argument("--out_jsonl", default="judgements.jsonl")
#     ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
#     ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
#     ap.add_argument("--max_tokens", type=int, default=160)
#     ap.add_argument("--temperature", type=float, default=0.0)
#     ap.add_argument("--gpu_util", type=float, default=0.55,
#                     help="vLLM gpu_memory_utilization (0.1–0.95).")
#     ap.add_argument("--max_model_len", type=int, default=2048,
#                     help="Reduce to shrink KV cache if GPU memory is tight (e.g., 1024–4096).")
#     ap.add_argument("--debug", action="store_true", help="Print first few raw generations.")
#     a = ap.parse_args()
#     return Args(**vars(a))

# def main():
#     args = parse_args()

#     # ---- Load data
#     dis = read_lines(args.disfluent)
#     ref = read_lines(args.fluent)
#     a = read_lines(args.cand_a)
#     b = read_lines(args.cand_b)

#     n_set = {len(dis), len(ref), len(a), len(b)}
#     if len(n_set) != 1:
#         raise SystemExit(f"[error] Files must be equal length. "
#                          f"got disfluent={len(dis)}, reference={len(ref)}, A={len(a)}, B={len(b)}")
#     if any(not x.strip() for x in ref):
#         raise SystemExit("[error] Reference fluent is mandatory; found empty lines.")
#     n = len(dis)

#     # ---- Languages
#     if args.lang_file:
#         lang_lines = read_lines(args.lang_file)
#         if len(lang_lines) != n:
#             raise SystemExit(f"[error] lang_file length {len(lang_lines)} != data length {n}")
#         langs = [x.strip().lower() if x.strip().lower() in {"hi", "mr", "bn"} else args.default_lang
#                  for x in lang_lines]
#     else:
#         langs = [args.default_lang] * n

#     # ---- Tokenizer (for chat template)
#     tokenizer = AutoTokenizer.from_pretrained(args.model)

#     # ---- Build chat-formatted prompts (batched)
#     prompts: List[str] = []
#     for i in range(n):
#         messages = [
#             {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
#             {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
#                 lang_name=lang_code_to_name(langs[i]),
#                 disfluent=dis[i].replace("```", "``` "),
#                 reference_fluent=ref[i].replace("```", "``` "),
#                 answer_a=a[i].replace("```", "``` "),
#                 answer_b=b[i].replace("```", "``` "),
#             )},
#         ]
#         # Chat template for system+user, then *prefill* assistant with '{' to force JSON start
#         base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         prompt = base + "{"
#         prompts.append(prompt)

#     print(f"Total judge prompts: {len(prompts)}")
#     if prompts and args.debug:
#         print("Example chat-formatted input (truncated):\n" + prompts[0][:500] + ("..." if len(prompts[0]) > 500 else ""))

#     # ---- vLLM model
#     llm = LLM(
#         model=args.model,
#         tensor_parallel_size=args.tp,
#         dtype=args.dtype,
#         trust_remote_code=False,
#         gpu_memory_utilization=args.gpu_util,
#         max_model_len=args.max_model_len,
#     )

#     # ---- Sampling (short JSON). Avoid stopping on the lone '}' so we can capture it.
#     stops = ["```"]  # keep minimal; eos_token added below if present
#     if tokenizer.eos_token:
#         stops.append(tokenizer.eos_token)

#     sampling_params = SamplingParams(
#         temperature=args.temperature,
#         max_tokens=args.max_tokens,
#         stop=stops,
#         seed=7,
#     )

#     # ---- Single batched call
#     outputs = llm.generate(prompts, sampling_params)
#     raw_texts = [out.outputs[0].text if out.outputs else "" for out in outputs]

#     if args.debug:
#         for j, t in enumerate(raw_texts[:3]):
#             print(f"\n[DEBUG raw #{j}] {repr(t[:300])} ...")

#     # ---- Parse JSON + normalize labels
#     def coerce_verdict(d: dict) -> dict:
#         out = {
#             "final_verdict": (d.get("final_verdict") or "Draw").strip().title(),
#             "rationale": (d.get("rationale") or "").strip(),
#         }
#         if out["final_verdict"] not in {"A", "B", "Draw"}:
#             out["final_verdict"] = "Draw"
#         if len(out["rationale"]) > 1000:
#             out["rationale"] = out["rationale"][:1000]
#         return out

#     results = []
#     for i, txt in enumerate(raw_texts):
#         # Ensure braces around the block we got from the model
#         t = coerce_json_block(txt)

#         # First try direct load, then extractor, then salvage
#         data = None
#         try:
#             data = json.loads(t)
#         except Exception:
#             data = extract_last_json(t)
#             if not data:
#                 data = salvage_json_to_verdict(t)

#         if not data:
#             data = {
#                 "rationale": "Unparseable judge output.",
#                 "final_verdict": "Draw"
#             }

#         verdict = coerce_verdict(data)
#         results.append({
#             "idx": i,
#             "lang": langs[i],
#             "disfluent": dis[i],
#             "reference_fluent": ref[i],
#             "answer_a": a[i],
#             "answer_b": b[i],
#             "judge_rationale": verdict["rationale"],
#             "judge_final_verdict": verdict["final_verdict"],
#         })

#     # ---- Save CSV + JSONL
#     out_csv = Path(args.out_csv)
#     out_jsonl = Path(args.out_jsonl)
#     out_csv.parent.mkdir(parents=True, exist_ok=True)
#     out_jsonl.parent.mkdir(parents=True, exist_ok=True)

#     with open(out_csv, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
#         writer.writeheader()
#         writer.writerows(results)
#     with open(out_jsonl, "w", encoding="utf-8") as f:
#         for r in results:
#             obj = {
#                 "rationale": r["judge_rationale"],
#                 "final_verdict": r["judge_final_verdict"],  
#             }
#             f.write(json.dumps(obj, ensure_ascii=False) + "\n")

#     print(f"✅ Wrote: {out_csv} and {out_jsonl}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#working code for single -single

# import argparse
# import csv
# import json
# import re
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Optional

# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams

# # ------------- Language map -------------
# LANG_MAP = {"hi": "Hindi", "mr": "Marathi", "bn": "Bengali"}

# # ------------- Judge prompts (English-only, exactly two keys) -------------
# JUDGE_SYSTEM_PROMPT = """You are an expert in disfluency correction for ASR outputs in Indo-Aryan languages (Hindi, Marathi, Bengali).
# You will ALWAYS be given a gold/reference fluent sentence. Treat it as the ground truth for meaning/adequacy.
# Compare TWO candidate corrections (Answer A and Answer B) for the SAME disfluent input.

# Judge ONLY on:
# 1) Fluency (grammaticality, naturalness),
# 2) Adequacy (faithfulness to the reference fluent sentence),
# 3) Disfluency removal quality (removes fillers, repetitions, false starts).

# Return STRICT JSON with exactly these keys (EXACTLY TWO KEYS, no extras), and write the rationale STRICTLY IN ENGLISH:
# {
#     "rationale": "brief why",
#     "final_verdict": "A|B|Draw"
# }
# IMPORTANT: The FIRST character of your response MUST be '{' and the LAST character MUST be '}'. Do NOT include any extra text, code fences, labels, or commentary before or after the JSON.
# """

# USER_PROMPT_TEMPLATE = """Language: {lang_name}
# Disfluent ASR sentence:
# {disfluent}

# Reference fluent (gold):
# {reference_fluent}

# Answer A:
# {answer_a}

# Answer B:
# {answer_b}
# """

# # ------------- Small helpers -------------
# def read_lines(path: str) -> List[str]:
#     with open(path, "r", encoding="utf-8") as f:
#         return [line.rstrip("\n\r") for line in f]

# def lang_code_to_name(code: str) -> str:
#     return LANG_MAP.get((code or "").strip().lower(), "Unknown")

# def extract_last_json(text: str) -> Optional[dict]:
#     """Return the LAST well-formed top-level JSON object found in `text`."""
#     if not text:
#         return None
#     candidates = []
#     in_str = False
#     esc = False
#     depth = 0
#     start = None
#     for i, ch in enumerate(text):
#         if in_str:
#             if esc:
#                 esc = False
#             elif ch == "\\":
#                 esc = True
#             elif ch == '"':
#                 in_str = False
#         else:
#             if ch == '"':
#                 in_str = True
#             elif ch == "{":
#                 if depth == 0:
#                     start = i
#                 depth += 1
#             elif ch == "}":
#                 if depth > 0:
#                     depth -= 1
#                     if depth == 0 and start is not None:
#                         candidates.append(text[start:i + 1])
#     for s in reversed(candidates):
#         try:
#             return json.loads(s)
#         except Exception:
#             continue
#     return None

# def coerce_json_block(text: str) -> str:
#     """
#     Make a best-effort valid JSON object string from model output:
#     - strip code fences
#     - slice to outermost braces if present
#     - ensure it starts with '{' and ends with '}'
#     - remove trailing commas before the final '}'
#     """
#     t = (text or "").strip()

#     # strip code fences
#     if t.startswith("```"):
#         t = t.strip("`")
#         if t.lower().startswith("json"):
#             t = t[4:].lstrip()

#     # If braces exist, slice to outermost braces
#     if "{" in t and "}" in t:
#         t = t[t.find("{"): t.rfind("}") + 1]

#     # ensure opening brace
#     if not t.startswith("{"):
#         t = "{\n" + t

#     # trim right, remove trailing comma at very end
#     t = t.rstrip()
#     if t.endswith(","):
#         t = t[:-1]

#     # ensure closing brace
#     if not t.endswith("}"):
#         t = t + "\n}"

#     # remove any trailing commas before }
#     t = re.sub(r",\s*}", "}", t)
#     return t

# def salvage_json_to_verdict(text: str) -> Optional[dict]:
#     """
#     Try to coerce near-JSON or free-text into a verdict dict with keys:
#     - final_verdict: "A" | "B" | "Draw"
#     Optionally returns rationale if present.
#     """
#     if not text:
#         return None

#     t = text

#     # If braces exist, keep inner slice
#     if "{" in t and "}" in t:
#         t = t[t.find("{"): t.rfind("}") + 1]

#     # Normalize unicode quotes and single->double for keys
#     t = t.replace("’", "'").replace("“", '"').replace("”", '"')
#     t = re.sub(r"([{,]\s*)'([^']+?)'\s*:", r'\1"\2":', t)
#     t = re.sub(r",\s*}", "}", t)

#     # Try straightforward JSON first after normalization
#     try:
#         d = json.loads(t)
#         if isinstance(d, dict) and ("final_verdict" in d or "rationale" in d):
#             return d
#     except Exception:
#         pass

#     # Regex for verdict
#     mV = re.search(r"(final[_\s-]*verdict|winner|choice|pick|decision|verdict)\s*[:=]\s*([AB]|Draw)\b", t, flags=re.I)
#     # Tie/draw patterns expressed in words
#     mTie = re.search(r"\b(tie|both\s+are\s+(good|similar)|equal(?:ly)?\s+(good|bad)|cannot\s+decide|no\s+clear\s+winner)\b", t, flags=re.I)

#     out: dict = {}
#     if mV:
#         out["final_verdict"] = mV.group(2).title()
#     elif mTie:
#         out["final_verdict"] = "Draw"
#     else:
#         mAB = re.search(r"(?:winner|better|preferred|superior|select|choose)\s*[:=]?\s*([AB])\b", t, flags=re.I)
#         if mAB:
#             out["final_verdict"] = mAB.group(1).upper()

#     if "final_verdict" in out:
#         return out
#     return None

# def ensure_dir(p: Path):
#     p.mkdir(parents=True, exist_ok=True)

# # ------------- CLI / main -------------
# @dataclass
# class Args:
#     model: str
#     disfluent: str
#     fluent: str
#     cand_a: str
#     cand_b: str
#     default_lang: str
#     lang_file: Optional[str]
#     out_csv: str
#     out_jsonl: str
#     tp: int
#     dtype: str
#     max_tokens: int
#     temperature: float
#     gpu_util: float
#     max_model_len: int
#     debug: bool

# def parse_args() -> Args:
#     ap = argparse.ArgumentParser(description="LLM-as-a-Judge using vLLM (batched generate).")
#     ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
#                     help="HF model id or local path (chat-instruct model recommended).")
#     ap.add_argument("--disfluent", required=True)
#     ap.add_argument("--fluent", required=True, help="Gold/reference fluent (mandatory; no empty lines).")
#     ap.add_argument("--cand_a", required=True)
#     ap.add_argument("--cand_b", required=True)
#     ap.add_argument("--default_lang", default="hi", choices=["hi", "mr", "bn"])
#     ap.add_argument("--lang_file", default=None, help="Optional per-line language codes (hi/mr/bn).")
#     ap.add_argument("--out_csv", default="judgements.csv")
#     ap.add_argument("--out_jsonl", default="judgements.jsonl")
#     ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
#     ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
#     ap.add_argument("--max_tokens", type=int, default=200)
#     ap.add_argument("--temperature", type=float, default=0.0)
#     ap.add_argument("--gpu_util", type=float, default=0.55,
#                     help="vLLM gpu_memory_utilization (0.1–0.95).")
#     ap.add_argument("--max_model_len", type=int, default=2048,
#                     help="Reduce to shrink KV cache if GPU memory is tight (e.g., 1024–4096).")
#     ap.add_argument("--debug", action="store_true", help="Print first few raw generations.")
#     a = ap.parse_args()
#     return Args(**vars(a))

# def main():
#     args = parse_args()

#     # ---- Load data
#     dis = read_lines(args.disfluent)
#     ref = read_lines(args.fluent)
#     a = read_lines(args.cand_a)
#     b = read_lines(args.cand_b)

#     n_set = {len(dis), len(ref), len(a), len(b)}
#     if len(n_set) != 1:
#         raise SystemExit(f"[error] Files must be equal length. "
#                          f"got disfluent={len(dis)}, reference={len(ref)}, A={len(a)}, B={len(b)}")
#     if any(not x.strip() for x in ref):
#         raise SystemExit("[error] Reference fluent is mandatory; found empty lines.")
#     n = len(dis)

#     # ---- Languages
#     if args.lang_file:
#         lang_lines = read_lines(args.lang_file)
#         if len(lang_lines) != n:
#             raise SystemExit(f"[error] lang_file length {len(lang_lines)} != data length {n}")
#         langs = [x.strip().lower() if x.strip().lower() in {"hi", "mr", "bn"} else args.default_lang
#                  for x in lang_lines]
#     else:
#         langs = [args.default_lang] * n

#     # ---- Tokenizer (for chat template)
#     tokenizer = AutoTokenizer.from_pretrained(args.model)

#     # ---- Build chat-formatted prompts (batched)
#     prompts: List[str] = []
#     prefixes: List[str] = []  # store the JSON prefix we prefill
#     for i in range(n):
#         messages = [
#             {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
#             {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
#                 lang_name=lang_code_to_name(langs[i]),
#                 disfluent=dis[i].replace("```", "``` "),
#                 reference_fluent=ref[i].replace("```", "``` "),
#                 answer_a=a[i].replace("```", "``` "),
#                 answer_b=b[i].replace("```", "``` "),
#             )},
#         ]
#         base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         json_prefix = '{"rationale":"'
#         prompt = base + json_prefix
#         prompts.append(prompt)
#         prefixes.append(json_prefix)

#     print(f"Total judge prompts: {len(prompts)}")

#     # ---- vLLM model
#     llm = LLM(
#         model=args.model,
#         tensor_parallel_size=args.tp,
#         dtype=args.dtype,
#         trust_remote_code=False,
#         gpu_memory_utilization=args.gpu_util,
#         max_model_len=args.max_model_len,
#     )

#     # ---- Sampling (short JSON). Avoid stopping on the lone '}' so we can capture it.
#     # Keep minimal; \n\n helps cut trailing chatter if any.
#     stops = ["```", "\n\n"]
#     if tokenizer.eos_token:
#         stops.append(tokenizer.eos_token)

#     sampling_params = SamplingParams(
#         temperature=args.temperature,
#         max_tokens=args.max_tokens,
#         stop=stops,
#         seed=7,
#     )

#     # ---- Single batched call
#     outputs = llm.generate(prompts, sampling_params)
#     # vLLM returns ONLY the continuation. Reconstruct full text = prefix + continuation.
#     cont_texts = [out.outputs[0].text if out.outputs else "" for out in outputs]
#     raw_texts = [prefixes[i] + cont_texts[i] for i in range(len(cont_texts))]

#     # Optional debug
#     # print first few fully-reconstructed texts
#     # if args.debug:
#     #     for j, t in enumerate(raw_texts[:3]):
#     #         print(f"\n[DEBUG full #{j}] {repr(t[:300])} ...")

#     # ---- Parse JSON (robust)
#     def coerce_verdict(d: dict) -> dict:
#         out = {
#             "final_verdict": (d.get("final_verdict") or "Draw").strip().title(),
#             "rationale": (d.get("rationale") or "").strip(),
#         }
#         if out["final_verdict"] not in {"A", "B", "Draw"}:
#             out["final_verdict"] = "Draw"
#         if len(out["rationale"]) > 1000:
#             out["rationale"] = out["rationale"][:1000]
#         return out

#     results = []
#     for i, txt in enumerate(raw_texts):
#         # 1) Try parsing RAW full text as-is
#         data = extract_last_json(txt)
#         if not data:
#             try:
#                 data = json.loads(txt)
#             except Exception:
#                 data = None

#         # 2) Try salvage on RAW
#         if not data:
#             data = salvage_json_to_verdict(txt)

#         # 3) As a last resort, coerce then parse
#         if not data:
#             t = coerce_json_block(txt)
#             try:
#                 data = json.loads(t)
#             except Exception:
#                 data = extract_last_json(t) or salvage_json_to_verdict(t)

#         if not data:
#             data = {
#                 "rationale": "Unparseable judge output.",
#                 "final_verdict": "Draw"
#             }

#         verdict = coerce_verdict(data)

#         # Soft guard: if rationale contains non-ASCII letters (likely not English), flag it
#         if re.search(r"[^\x00-\x7F]", verdict["rationale"]):
#             verdict["rationale"] = "(Expected English) " + verdict["rationale"]

#         results.append({
#             "idx": i,
#             "lang": langs[i],
#             "disfluent": dis[i],
#             "reference_fluent": ref[i],
#             "answer_a": a[i],
#             "answer_b": b[i],
#             "judge_rationale": verdict["rationale"],
#             "judge_final_verdict": verdict["final_verdict"],
#         })

#     # ---- Save CSV + JSONL
#     out_csv = Path(args.out_csv)
#     out_jsonl = Path(args.out_jsonl)
#     out_csv.parent.mkdir(parents=True, exist_ok=True)
#     out_jsonl.parent.mkdir(parents=True, exist_ok=True)

#     with open(out_csv, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
#         writer.writeheader()
#         writer.writerows(results)

#     with open(out_jsonl, "w", encoding="utf-8") as f:
#         for r in results:
#             obj = {
#                 "rationale": r["judge_rationale"],
#                 "final_verdict": r["judge_final_verdict"],
#             }
#             f.write(json.dumps(obj, ensure_ascii=False) + "\n")

#     print(f"✅ Wrote: {out_csv} and {out_jsonl}")

# if __name__ == "__main__":
#     main()





#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ------------- Language map -------------
LANG_MAP = {"hi": "Hindi", "mr": "Marathi", "bn": "Bengali"}

# ------------- Judge prompt (English-only, exactly two keys) -------------
JUDGE_SYSTEM_PROMPT = """You are an expert in disfluency correction for ASR outputs in Indo-Aryan languages (Hindi, Marathi, Bengali).
You will ALWAYS be given a gold/reference fluent sentence. Treat it as the ground truth for meaning/adequacy.
Compare TWO candidate corrections for the SAME disfluent input.

Judge ONLY on:
1) Fluency (grammaticality, naturalness),
2) Adequacy (faithfulness to the reference fluent sentence),
3) Disfluency removal quality (removes fillers, repetitions, false starts).

CRITICAL: Do NOT prefer any candidate because of its position or label. Evaluate content ONLY.

Return STRICT JSON with exactly these keys (EXACTLY TWO KEYS, no extras), and write the rationale STRICTLY IN ENGLISH:
{
    "rationale": "brief why",
    "final_verdict": "A|B|Draw"
}
IMPORTANT: The FIRST character of your response MUST be '{' and the LAST character MUST be '}'. Do NOT include any extra text, code fences, labels, or commentary before or after the JSON.
"""

USER_PROMPT_TEMPLATE = """Language: {lang_name}
Disfluent ASR sentence:
{disfluent}

Reference fluent (gold):
{reference_fluent}

Answer A:
{answer_a}

Answer B:
{answer_b}
"""

# ------------- Small helpers -------------
def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n\r") for line in f]

def lang_code_to_name(code: str) -> str:
    return LANG_MAP.get((code or "").strip().lower(), "Unknown")

def extract_last_json(text: str) -> Optional[dict]:
    """Return the LAST well-formed top-level JSON object found in `text`."""
    if not text:
        return None
    candidates = []
    in_str = False
    esc = False
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidates.append(text[start:i + 1])
    for s in reversed(candidates):
        try:
            return json.loads(s)
        except Exception:
            continue
    return None

def coerce_json_block(text: str) -> str:
    """
    Make a best-effort valid JSON object string from model output:
    - strip code fences
    - slice to outermost braces if present
    - ensure it starts with '{' and ends with '}'
    - remove trailing commas before the final '}'
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
    if "{" in t and "}" in t:
        t = t[t.find("{"): t.rfind("}") + 1]
    if not t.startswith("{"):
        t = "{\n" + t
    t = t.rstrip()
    if t.endswith(","):
        t = t[:-1]
    if not t.endswith("}"):
        t = t + "\n}"
    t = re.sub(r",\s*}", "}", t)
    return t

def salvage_json_to_verdict(text: str) -> Optional[dict]:
    """
    Try to coerce near-JSON or free-text into a verdict dict with keys:
    - final_verdict: "A" | "B" | "Draw"
    Optionally returns rationale if present.
    """
    if not text:
        return None
    t = text
    if "{" in t and "}" in t:
        t = t[t.find("{"): t.rfind("}") + 1]
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"([{,]\s*)'([^']+?)'\s*:", r'\1"\2":', t)
    t = re.sub(r",\s*}", "}", t)
    try:
        d = json.loads(t)
        if isinstance(d, dict) and ("final_verdict" in d or "rationale" in d):
            return d
    except Exception:
        pass
    mV = re.search(r"(final[_\s-]*verdict|winner|choice|pick|decision|verdict)\s*[:=]\s*([AB]|Draw)\b", t, flags=re.I)
    mTie = re.search(r"\b(tie|both\s+are\s+(good|similar)|equal(?:ly)?\s+(good|bad)|cannot\s+decide|no\s+clear\s+winner)\b", t, flags=re.I)
    out: dict = {}
    if mV:
        out["final_verdict"] = mV.group(2).title()
    elif mTie:
        out["final_verdict"] = "Draw"
    else:
        mAB = re.search(r"(?:winner|better|preferred|superior|select|choose)\s*[:=]?\s*([AB])\b", t, flags=re.I)
        if mAB:
            out["final_verdict"] = mAB.group(1).upper()
    if "final_verdict" in out:
        return out
    return None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ------------- CLI / main -------------
@dataclass
class Args:
    model: str
    disfluent: str
    fluent: str
    cand_a: str
    cand_b: str
    default_lang: str
    lang_file: Optional[str]
    out_csv: str
    out_jsonl: str
    tp: int
    dtype: str
    max_tokens: int
    temperature: float
    gpu_util: float
    max_model_len: int
    debug: bool

def parse_args() -> Args:
    ap = argparse.ArgumentParser(description="Order-debiased LLM-as-a-Judge using vLLM (batched generate).")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                    help="HF model id or local path (chat-instruct model recommended).")
    ap.add_argument("--disfluent", required=True)
    ap.add_argument("--fluent", required=True, help="Gold/reference fluent (mandatory; no empty lines).")
    ap.add_argument("--cand_a", required=True)
    ap.add_argument("--cand_b", required=True)
    ap.add_argument("--default_lang", default="hi", choices=["hi", "mr", "bn"])
    ap.add_argument("--lang_file", default=None, help="Optional per-line language codes (hi/mr/bn).")
    ap.add_argument("--out_csv", default="judgements.csv")
    ap.add_argument("--out_jsonl", default="judgements.jsonl")
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu_util", type=float, default=0.55,
                    help="vLLM gpu_memory_utilization (0.1–0.95).")
    ap.add_argument("--max_model_len", type=int, default=2048,
                    help="Reduce to shrink KV cache if GPU memory is tight (e.g., 1024–4096).")
    ap.add_argument("--debug", action="store_true", help="Print first few raw generations.")
    a = ap.parse_args()
    return Args(**vars(a))

def build_prompt(tokenizer, lang_name: str, dis: str, ref: str, candA: str, candB: str) -> Tuple[str, str]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            lang_name=lang_name,
            disfluent=dis.replace("```", "``` "),
            reference_fluent=ref.replace("```", "``` "),
            answer_a=candA.replace("```", "``` "),
            answer_b=candB.replace("```", "``` "),
        )},
    ]
    base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    json_prefix = '{"rationale":"'
    prompt = base + json_prefix  # vLLM will generate the continuation only
    return prompt, json_prefix

def parse_one(raw_text: str) -> dict:
    """Robust parse of a single model output (already reconstructed with prefix)."""
    data = extract_last_json(raw_text)
    if not data:
        try:
            data = json.loads(raw_text)
        except Exception:
            data = None
    if not data:
        data = salvage_json_to_verdict(raw_text)
    if not data:
        t = coerce_json_block(raw_text)
        try:
            data = json.loads(t)
        except Exception:
            data = extract_last_json(t) or salvage_json_to_verdict(t)
    if not data:
        data = {"rationale": "Unparseable judge output.", "final_verdict": "Draw"}

    out = {
        "final_verdict": (data.get("final_verdict") or "Draw").strip().title(),
        "rationale": (data.get("rationale") or "").strip(),
    }
    if out["final_verdict"] not in {"A", "B", "Draw"}:
        out["final_verdict"] = "Draw"
    if re.search(r"[^\x00-\x7F]", out["rationale"]):
        out["rationale"] = "(Expected English) " + out["rationale"]
    if len(out["rationale"]) > 1000:
        out["rationale"] = out["rationale"][:1000]
    return out

def map_verdict_from_swapped(v: str) -> str:
    """Map verdict from (B,A) order back to original (A,B) order."""
    if v == "A":
        return "B"
    elif v == "B":
        return "A"
    return "Draw"

def aggregate_verdict(v_ab: str, v_ba_mapped: str) -> str:
    """Aggregate two verdicts from AB and (BA mapped->AB)."""
    if v_ab == v_ba_mapped:
        return v_ab
    # If one is Draw and the other A/B, keep the non-draw (more informative).
    if v_ab == "Draw" and v_ba_mapped in {"A", "B"}:
        return v_ba_mapped
    if v_ba_mapped == "Draw" and v_ab in {"A", "B"}:
        return v_ab
    # Otherwise conflict A vs B -> Draw
    return "Draw"

def main():
    args = parse_args()

    # ---- Load data
    dis = read_lines(args.disfluent)
    ref = read_lines(args.fluent)
    a = read_lines(args.cand_a)
    b = read_lines(args.cand_b)

    n_set = {len(dis), len(ref), len(a), len(b)}
    if len(n_set) != 1:
        raise SystemExit(f"[error] Files must be equal length. "
                         f"got disfluent={len(dis)}, reference={len(ref)}, A={len(a)}, B={len(b)}")
    if any(not x.strip() for x in ref):
        raise SystemExit("[error] Reference fluent is mandatory; found empty lines.")
    n = len(dis)

    # ---- Languages
    if args.lang_file:
        lang_lines = read_lines(args.lang_file)
        if len(lang_lines) != n:
            raise SystemExit(f"[error] lang_file length {len(lang_lines)} != data length {n}")
        langs = [x.strip().lower() if x.strip().lower() in {"hi", "mr", "bn"} else args.default_lang
                 for x in lang_lines]
    else:
        langs = [args.default_lang] * n

    # ---- Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ---- Build BOTH orders' prompts per example
    prompts: List[str] = []
    prefixes: List[str] = []  # json prefixes to reconstruct
    index_order: List[Tuple[int, str]] = []  # (idx, "AB" or "BA")

    for i in range(n):
        lang_name = lang_code_to_name(langs[i])

        # AB order
        p_ab, pref_ab = build_prompt(tokenizer, lang_name, dis[i], ref[i], a[i], b[i])
        prompts.append(p_ab)
        prefixes.append(pref_ab)
        index_order.append((i, "AB"))

        # BA order (swap candidates)
        p_ba, pref_ba = build_prompt(tokenizer, lang_name, dis[i], ref[i], b[i], a[i])
        prompts.append(p_ba)
        prefixes.append(pref_ba)
        index_order.append((i, "BA"))

    print(f"Total judge prompts (debiased): {len(prompts)} (2x per item)")

    # ---- vLLM model
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
        trust_remote_code=False,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
    )

    # ---- Sampling
    stops = ["```", "\n\n"]
    if tokenizer.eos_token:
        stops.append(tokenizer.eos_token)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=stops,
        seed=7,
    )

    # ---- Batched call
    outputs = llm.generate(prompts, sampling_params)
    continuations = [out.outputs[0].text if out.outputs else "" for out in outputs]
    full_texts = [prefixes[i] + continuations[i] for i in range(len(continuations))]

    if args.debug:
        for j, t in enumerate(full_texts[:4]):
            print(f"\n[DEBUG reconstructed #{j}] {repr(t[:300])} ...")

    # ---- Parse all, then aggregate per original index
    interim = {}  # idx -> {"AB": (verdict, rationale), "BA": (mapped_verdict, rationaleBA)}
    for k, txt in enumerate(full_texts):
        idx, order = index_order[k]
        parsed = parse_one(txt)
        verdict = parsed["final_verdict"]
        rationale = parsed["rationale"]

        if order == "BA":
            verdict = map_verdict_from_swapped(verdict)  # map back to original AB meaning

        if idx not in interim:
            interim[idx] = {}
        interim[idx][order] = (verdict, rationale)

    # ---- Build final results
    results = []
    for i in range(n):
        v_ab, r_ab = interim[i].get("AB", ("Draw", "No AB rationale."))  # should exist
        v_ba, r_ba = interim[i].get("BA", ("Draw", "No BA rationale."))  # mapped

        final_v = aggregate_verdict(v_ab, v_ba)

        # Choose rationale from the order that matches final verdict; if Draw due to disagreement, show both (short)
        if final_v == v_ab and final_v != "Draw":
            chosen_r = r_ab
        elif final_v == v_ba and final_v != "Draw":
            chosen_r = r_ba
        else:
            # Draw: combine succinctly
            # Trim to avoid huge cells
            ra = r_ab[:300].strip()
            rb = r_ba[:300].strip()
            chosen_r = f"Order-debiased: AB says {v_ab}. BA says {v_ba}. AB rationale: {ra} | BA rationale: {rb}"

        results.append({
            "idx": i,
            "lang": langs[i],
            "disfluent": dis[i],
            "reference_fluent": ref[i],
            "answer_a": a[i],
            "answer_b": b[i],
            "judge_rationale": chosen_r,
            "judge_final_verdict": final_v,
        })

    # ---- Save CSV + JSONL
    out_csv = Path(args.out_csv)
    out_jsonl = Path(args.out_jsonl)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            obj = {
                "rationale": r["judge_rationale"],
                "final_verdict": r["judge_final_verdict"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Wrote: {out_csv} and {out_jsonl}")

if __name__ == "__main__":
    main()
