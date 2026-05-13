import csv

dis_file = "/home/paritosh/zero-shot-disfluency-detection/mBART/mar_data.dis"
flu_file = "/home/paritosh/zero-shot-disfluency-detection/mBART/mar_data.flu"
out_file = "/home/paritosh/zero-shot-disfluency-detection/mBART/mar_data.tsv"

with open(dis_file, "r", encoding="utf-8") as f1, \
     open(flu_file, "r", encoding="utf-8") as f2, \
     open(out_file, "w", encoding="utf-8", newline="") as out:
    
    dis_lines = [line.strip() for line in f1]
    flu_lines = [line.strip() for line in f2]

    if len(dis_lines) != len(flu_lines):
        raise ValueError(f"Line mismatch: {len(dis_lines)} disfluent vs {len(flu_lines)} fluent")

    writer = csv.writer(out, delimiter="\t")
    # header
    writer.writerow(["lang", "disfluent", "fluent"])
    # rows
    for d, f in zip(dis_lines, flu_lines):
        writer.writerow(["mr", d, f])

print(f"Wrote {out_file} with {len(dis_lines)} rows")
