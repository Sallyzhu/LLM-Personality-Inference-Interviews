# =========================
# 0. Imports
# =========================
import re
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

# =========================
# 1. Config
# =========================
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
TRAIN_PATH = "train_supervised.jsonl"
TEST_PATH = "test_supervised.jsonl"
MAX_LENGTH = 2048

# =========================
# 2. Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# 3. Load Dataset
# =========================
dataset = load_dataset("json", data_files=TRAIN_PATH)["train"]

print("Train samples:", len(dataset))
print(dataset[0]["text"][:300])

# 
#dataset = dataset.filter(lambda x: len(x["text"]) > 10)
#dataset = dataset.select(range(20))
# =========================
# 4. QLoRA config
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

# =========================
# 5. LoRA
# =========================
lora_config = LoraConfig(
    r=16,                    
    lora_alpha=64,            
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

model.print_trainable_parameters()
# =========================
# 6. Training args
# =========================
training_args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,   
    logging_steps=10,
    save_strategy="epoch",
    weight_decay=0.01,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

# =========================
# 7. Trainer
# =========================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_LENGTH,
    tokenizer=tokenizer,
    args=training_args
)

# =========================
# 8. Train
# =========================
trainer.train()

trainer.model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

print("✅ Training done!")

# =========================
# 9. Inference model
# =========================
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

model_infer = PeftModel.from_pretrained(base_model, "lora_model")
model_infer.eval()

# =========================
# 10. Prompt（⚠️ 必须和训练一致）
# =========================
def build_prompt(text):
    return f"""User: Analyze personality traits from the interview.

{text}

Return ONLY in this format(no explanation):
Openness: X.X
Conscientiousness: X.X
Extraversion: X.X
Agreeableness: X.X
Neuroticism: X.X

Assistant:
"""

# =========================
# 11. Generate
# =========================
def generate(text):
    inputs = tokenizer(
        build_prompt(text[:1500]),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(model_infer.device)

    with torch.no_grad():
        outputs = model_infer.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.0,  
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# =========================
# 12. Parse
# =========================
def parse_scores(output):
    pattern = r"Openness:\s*([\d.]+).*?Conscientiousness:\s*([\d.]+).*?Extraversion:\s*([\d.]+).*?Agreeableness:\s*([\d.]+).*?Neuroticism:\s*([\d.]+)"
    match = re.search(pattern, output, re.DOTALL)

    if match:
        return list(map(float, match.groups()))
    return [None]*5

# =========================
# 13. Run on test set
# =========================
test_dataset = load_dataset("json", data_files=TEST_PATH)["train"]

# ✅ 
#test_dataset = test_dataset.select(range(5))

print("Test samples:", len(test_dataset))
test_dataset=test_dataset
results = []

for i in range(len(test_dataset)):
    text = test_dataset[i]["text"]

    # 
    text = text.split("Assistant:")[0]

    output = generate(text)
    print(f"\n🔥 {i}:", output)

    scores = parse_scores(output)
    results.append(scores)

print("✅ Inference done!")

import pandas as pd
import numpy as np

# 
pred_df = pd.DataFrame(results, columns=[
    "Pred_Openness",
    "Pred_Conscientiousness",
    "Pred_Extraversion",
    "Pred_Agreeableness",
    "Pred_Neuroticism"
])

print(pred_df)

import re

def extract_gt(text):
    pattern = r"Openness:\s*([\d.]+).*?Conscientiousness:\s*([\d.]+).*?Extraversion:\s*([\d.]+).*?Agreeableness:\s*([\d.]+).*?Neuroticism:\s*([\d.]+)"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return list(map(float, match.groups()))
    return [None]*5

gt_list = []

for i in range(len(test_dataset)):
    text = test_dataset[i]["text"]
    gt_list.append(extract_gt(text))

gt_df = pd.DataFrame(gt_list, columns=[
    "GT_Openness",
    "GT_Conscientiousness",
    "GT_Extraversion",
    "GT_Agreeableness",
    "GT_Neuroticism"
])

print(gt_df)

df_final = pd.concat([gt_df, pred_df], axis=1)

print(df_final.head())

from scipy.stats import pearsonr

def correlation_stats(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 3:
        return None, None, None, None, None

    r, p = pearsonr(y_true, y_pred)

    z = np.arctanh(r)
    se = 1 / np.sqrt(len(y_true) - 3)

    z_low = z - 1.96 * se
    z_high = z + 1.96 * se

    r_low = np.tanh(z_low)
    r_high = np.tanh(z_high)

    return r, p, se, r_low, r_high

traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

results_stats = []

for trait in traits:
    y_true = df_final[f"GT_{trait}"].values
    y_pred = df_final[f"Pred_{trait}"].values

    r, p, se, low, high = correlation_stats(y_true, y_pred)

    results_stats.append([trait, r, p, se, low, high])

stats_df = pd.DataFrame(results_stats, columns=[
    "Trait", "Pearson_r", "p_value", "SE", "CI_lower", "CI_upper"
])

print(stats_df)

df_final.to_csv("lora_predictions_full.csv", index=False)
stats_df.to_csv("lora_correlation_results.csv", index=False)

print("✅ Saved results!")
