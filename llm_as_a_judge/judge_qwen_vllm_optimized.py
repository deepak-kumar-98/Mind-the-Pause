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
    mTie = re.search(r"\b(tie|both\s+are\s+(good|similar)|equal(?:ly)?\s+(good|bad)|cannot\s+decide|no\s+clear\s+winner|draw)\b", t, flags=re.I)
    out: dict = {}
    if mV:
        out["final_verdict"] = mV.group(2).title()
    elif mTie:
        out["final_verdict"] = "Draw"
    else:
        mAB = re.search(r"(?:winner|better|preferred|superior|select|choose|pick)\s*[:=]?\s*([AB])\b", t, flags=re.I)
        if mAB:
            out["final_verdict"] = mAB.group(1).upper()
    if "final_verdict" in out:
        return out
    return None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------- NEW: infer verdict from rationale (consistency fixer) ----------
def infer_verdict_from_text(t: str) -> Optional[str]:
    """
    Heuristically infer verdict from English rationale.
    Returns "A" | "B" | "Draw" | None
    """
    if not t:
        return None
    s = t.lower()

    # explicit draw / tie
    if re.search(r"\b(tie|both\s+are\s+(good|similar)|equally\s+(good|bad)|no\s+clear\s+winner|draw)\b", s):
        return "Draw"

    # strong "prefer/choose/winner/better" patterns close to A/B
    m = re.search(
        r"\b(prefer|choose|selected?|winner|wins?|better|superior|stronger|more\s+(?:fluent|adequate|faithful|natural|accurate|correct))\b.*\b(a|b)\b",
        s,
    )
    if m:
        return m.group(2).upper()

    # “A is more … / B is more …”
    if re.search(r"\ba\s+is\s+(?:more|most|clearly|significantly)\b", s):
        return "A"
    if re.search(r"\bb\s+is\s+(?:more|most|clearly|significantly)\b", s):
        return "B"

    # Mentions like "Answer A wins", "Candidate B preferred"
    if re.search(r"\b(answer|candidate|option)\s*a\b.*\b(better|prefer|chosen|wins?)\b", s):
        return "A"
    if re.search(r"\b(answer|candidate|option)\s*b\b.*\b(better|prefer|chosen|wins?)\b", s):
        return "B"

    # Fallback: lone mentions (weak; only if one side appears)
    has_a = re.search(r"\b(answer|candidate|option)\s*a\b", s) is not None
    has_b = re.search(r"\b(answer|candidate|option)\s*b\b", s) is not None
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    return None

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
    ap = argparse.ArgumentParser(description="Order-debiased LLM-as-a-Judge using vLLM (Qwen2.5-3B-Instruct) with rationale-consistency fix.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
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
    # Qwen chat template works via apply_chat_template
    base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    json_prefix = '{"rationale":"'
    prompt = base + json_prefix
    return prompt, json_prefix

def parse_one(raw_text: str) -> dict:
    """Robust parse and then fix verdict if rationale implies a different choice."""
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

    # ---- Consistency fix: infer from rationale and override if disagree
    inferred = infer_verdict_from_text(out["rationale"])
    if inferred in {"A", "B", "Draw"} and inferred != out["final_verdict"]:
        out["final_verdict"] = inferred

    return out

def map_verdict_from_swapped(v: str) -> str:
    if v == "A":
        return "B"
    elif v == "B":
        return "A"
    return "Draw"

def aggregate_verdict(v_ab: str, v_ba_mapped: str) -> str:
    if v_ab == v_ba_mapped:
        return v_ab
    if v_ab == "Draw" and v_ba_mapped in {"A", "B"}:
        return v_ba_mapped
    if v_ba_mapped == "Draw" and v_ab in {"A", "B"}:
        return v_ab
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)

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
    interim = {}
    for k, txt in enumerate(full_texts):
        idx, order = index_order[k]
        parsed = parse_one(txt)
        verdict = parsed["final_verdict"]
        rationale = parsed["rationale"]

        if order == "BA":
            verdict = map_verdict_from_swapped(verdict)

        if idx not in interim:
            interim[idx] = {}
        interim[idx][order] = (verdict, rationale)

    # ---- Build final results
    results = []
    for i in range(n):
        v_ab, r_ab = interim[i].get("AB", ("Draw", "No AB rationale."))
        v_ba, r_ba = interim[i].get("BA", ("Draw", "No BA rationale."))

        final_v = aggregate_verdict(v_ab, v_ba)

        if final_v == v_ab and final_v != "Draw":
            chosen_r = r_ab
        elif final_v == v_ba and final_v != "Draw":
            chosen_r = r_ba
        else:
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
