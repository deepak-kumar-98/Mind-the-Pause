import os
from pathlib import Path

# Folder path
folder = Path("/home/paritosh/zero-shot-disfluency-detection/detection_hi_bn_mr/iiith_data")

# Output files
combined_txt = folder / "combined.txt"
combined_label = folder / "combined.label"

# Collect all .txt and .label files
txt_files = sorted(folder.glob("*.txt"))
label_files = sorted(folder.glob("*.label"))

with open(combined_txt, "w", encoding="utf-8") as txt_out, \
     open(combined_label, "w", encoding="utf-8") as label_out:
    
    for txt_file, label_file in zip(txt_files, label_files):
        with open(txt_file, "r", encoding="utf-8") as tf, \
             open(label_file, "r", encoding="utf-8") as lf:
            
            for text_line, label_line in zip(tf, lf):
                text_line = text_line.strip()
                label_line = label_line.strip()
                
                if not text_line or not label_line:
                    continue  # skip empty lines
                
                # Convert labels: O -> 0, everything else -> 1
                new_labels = " ".join(["0" if lbl == "O" else "1" for lbl in label_line.split()])
                
                # Write to combined files
                txt_out.write(text_line + "\n")
                label_out.write(new_labels + "\n")

print("✅ Combined dataset created:")
print(f" - Sentences: {combined_txt}")
print(f" - Labels: {combined_label}")
