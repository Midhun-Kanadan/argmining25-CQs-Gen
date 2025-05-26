import os
import json
from transformers import pipeline
import torch._dynamo
from tqdm import tqdm
from itertools import combinations

torch._dynamo.config.suppress_errors = True

# Load ModernBERT classifier
classifier = pipeline("text-classification", model="MidhunKanadan/ModernBERT-CritiQ-V2")

# === Configuration: Manual or All Combinations ===
# Manually specify LLMs to combine (leave empty to auto-generate all pairwise combinations)
MANUAL_LLMS = ["gemma2", "phi4"]  # e.g., ["gemma2", "phi4"] or [] for all pairs
# MANUAL_LLMS = []

# CQ types to process
# cq_types = ["Test_CQs", "Sample_CQs", "Validation_CQs"]
cq_types = ["Test_CQs"]

# Define input and output directories
input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "generated_cqs"))
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reranked_cqs"))

# === Detect LLM Folders and Create Combinations ===
all_llms_available = sorted([
    d for d in os.listdir(input_dir)
    if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith(".")
])

if MANUAL_LLMS:
    llm_combinations = [MANUAL_LLMS]
else:
    llm_combinations = list(combinations(all_llms_available, 2))  # Only pairs

# === Detect available prompts ===
def get_available_prompts(llm_name):
    llm_dir = os.path.join(input_dir, llm_name)
    if not os.path.exists(llm_dir):
        return []
    return sorted([
        int(name) for name in os.listdir(llm_dir)
        if name.isdigit() and os.path.isdir(os.path.join(llm_dir, name))
    ])

# Use the first available model to get prompt numbers
prompts = get_available_prompts(all_llms_available[0])
print(f"🧠 Detected available prompts: {prompts}")

# === Path helpers ===
def get_llm_json_path(llm, prompt, cq_type):
    filename = f"{llm}_prompt{prompt}_{cq_type.lower().replace('_cqs', '')}.json"
    return os.path.join(input_dir, llm, str(prompt), cq_type, filename)

def get_save_path(llm_tag, prompt, cq_type):
    save_folder = os.path.join(output_dir, llm_tag, str(prompt), cq_type)
    os.makedirs(save_folder, exist_ok=True)
    filename = f"{llm_tag}_prompt{prompt}_{cq_type.lower().replace('_cqs', '')}.json"
    return os.path.join(save_folder, filename)

# === Main reranking loop ===
for llms in llm_combinations:
    combined_tag = "-".join(llms)
    print(f"\n🔄 Reranking combination: {combined_tag}")

    for prompt in prompts:
        for cq_type in cq_types:
            all_cqs = {}

            # Load CQs from each LLM in the current combination
            for llm in llms:
                json_path = get_llm_json_path(llm, prompt, cq_type)
                if not os.path.exists(json_path):
                    print(f"⚠️ Missing: {json_path}")
                    continue

                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key, val in data.items():
                        if key not in all_cqs:
                            all_cqs[key] = {
                                "intervention_id": val["intervention_id"],
                                "intervention": val["intervention"],
                                "dataset": val["dataset"],
                                "schemes": val.get("schemes", []),
                                "cqs": []
                            }
                        all_cqs[key]["cqs"].extend(val["cqs"])

            # Evaluate and select top 3 Useful
            filtered_output = {}
            total = len(all_cqs)
            print(f"🔍 Reranking {total} interventions for Prompt {prompt}, Type {cq_type}...")
            for key, entry in tqdm(all_cqs.items(), total=total, desc=f"Reranking P{prompt}-{cq_type}"):
                intervention = entry["intervention"]
                evaluated = []

                for cq in entry["cqs"]:
                    prompt_text = f"Intervention: {intervention} [SEP] Critical Question: {cq['cq']}"
                    pred = classifier(prompt_text)[0]
                    if pred["label"] == "Useful":
                        evaluated.append({"cq": cq["cq"], "score": pred["score"]})

                top_3 = sorted(evaluated, key=lambda x: x["score"], reverse=True)[:3]
                filtered_output[key] = {
                    "intervention_id": entry["intervention_id"],
                    "intervention": entry["intervention"],
                    "dataset": entry["dataset"],
                    "schemes": entry["schemes"],
                    "cqs": [{"id": i, "cq": x["cq"]} for i, x in enumerate(top_3)]
                }

            # Save to structured folder
            save_path = get_save_path(combined_tag, prompt, cq_type)
            with open(save_path, "w", encoding="utf-8") as out:
                json.dump(filtered_output, out, indent=4, ensure_ascii=False)

            print(f"✅ Saved: {save_path}")
