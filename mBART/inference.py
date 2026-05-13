from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

# Path to your fine-tuned model checkpoint
model_dir = "/home/paritosh/zero-shot-disfluency-detection/mBART/checkpoint-18000 copy"

# Load tokenizer and model
tokenizer = MBart50TokenizerFast.from_pretrained(model_dir)
model = MBartForConditionalGeneration.from_pretrained(model_dir)

# Move model to GPU (FP16 if you want speed + less memory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Optional: model = model.half()   # saves VRAM if you’re not using bf16

# Set language codes
src_lang = "mr_IN"
tokenizer.src_lang = src_lang
tokenizer.tgt_lang = src_lang

# Input and output files
input_file = "/home/paritosh/zero-shot-disfluency-detection/evaluation dataset/real disfluency correction data/marathi/test.processed.dis"
output_file = "/home/paritosh/zero-shot-disfluency-detection/mBART/prediction/real_marathi_test_predict.flu"

with open(input_file, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

batch_size = 16
outputs = []

for i in range(0, len(sentences), batch_size):
    print(f"Processing batch {i//batch_size+1}")
    batch = sentences[i : i + batch_size]

    encodings = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)  # <-- move inputs to GPU

    with torch.no_grad():  # no gradient calc during inference
        generated_ids = model.generate(
            **encodings,
            max_length=256,
            num_beams=5
        )

    decoded_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    outputs.extend(decoded_batch)

with open(output_file, "w", encoding="utf-8") as f:
    for line in outputs:
        f.write(line.strip() + "\n")

print(f"✅ Done. Wrote {len(outputs)} sentences to {output_file}")
