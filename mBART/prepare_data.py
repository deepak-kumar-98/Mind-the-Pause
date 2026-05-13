# prepare_data.py
import argparse, json, unicodedata, re, os, glob, random
import pandas as pd

LANG2MBART = {
    "hi": "hi_IN",
    "mr": "mr_IN",
    "bn": "bn_IN",
    "ta": "ta_IN",
}

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u200c", "").replace("\u200d", "")  # ZWNJ/ZWJ
    s = re.sub(r"\s+", " ", s).strip()
    return s

def read_any(path):
    # Auto-detect delimiter by simple heuristic
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)  # assume CSV

def load_frames(input_path):
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.*sv")))
        if not files:
            raise FileNotFoundError(f"No CSV/TSV files in {input_path}")
        dfs = [read_any(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = read_any(input_path)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True,
                    help="CSV/TSV file OR directory containing multiple CSV/TSV files")
    ap.add_argument("--output_dir", required=True, help="Where to write JSONL train/valid")
    ap.add_argument("--valid_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_frames(args.input_path)
    cols = {c.lower(): c for c in df.columns}
    required = ["lang", "disfluent", "fluent"]
    for r in required:
        if r not in cols:
            raise ValueError(f"Missing required column '{r}' (case-insensitive). Found: {df.columns.tolist()}")

    df = df.rename(columns={cols["lang"]: "lang",
                            cols["disfluent"]: "disfluent",
                            cols["fluent"]: "fluent"})

    # Basic cleaning
    df["lang"] = df["lang"].str.strip().str.lower()
    df["src"] = df["disfluent"].apply(clean_text)
    df["tgt"] = df["fluent"].apply(clean_text)

    # Filter supported langs
    df = df[df["lang"].isin(LANG2MBART.keys())].copy()

    # Drop empty and duplicates
    df = df[(df["src"] != "") & (df["tgt"] != "")]
    df = df.drop_duplicates(subset=["lang", "src", "tgt"])

    # Map to mbart codes (same src/tgt language because it's correction)
    df["mbart_lang"] = df["lang"].map(LANG2MBART)

    # Shuffle and split
    random.seed(args.seed)
    idx = list(range(len(df)))
    random.shuffle(idx)
    cut = int(len(idx) * (1 - args.valid_ratio))
    train_idx, valid_idx = idx[:cut], idx[cut:]

    def to_jsonl(rows, path):
        with open(path, "w", encoding="utf-8") as f:
            for _, r in rows.iterrows():
                ex = {
                    "lang": r["lang"],
                    "mbart_lang": r["mbart_lang"],
                    "source": r["src"],
                    "target": r["tgt"],
                }
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    to_jsonl(df.iloc[train_idx], os.path.join(args.output_dir, "train.jsonl"))
    to_jsonl(df.iloc[valid_idx], os.path.join(args.output_dir, "valid.jsonl"))

    # Tiny stats
    stats = df.groupby("lang").size().to_dict()
    with open(os.path.join(args.output_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"total": int(len(df)), "by_lang": stats}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(train_idx)} train and {len(valid_idx)} valid examples to {args.output_dir}")

if __name__ == "__main__":
    main()
