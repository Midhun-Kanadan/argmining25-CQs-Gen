import json
import os
import logging
import re
import pandas as pd
from tqdm import tqdm
import ollama
import subprocess
import glob
import concurrent.futures
import time

# Logging Setup
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "generated_cqs")
os.makedirs(RESULTS_DIR, exist_ok=True)
# DATASETS = {"sample": "sample.json", "validation": "validation.json", "test": "test.json"}
DATASETS = {"test": "test.json"}

# List of models to process
MODELS = ["gemma2", "phi4"]
# MODELS = ["gemma2", "phi4", "gemma3"]
# MODELS = ["gemma2", "gemma3", "mistral","llama3.2", "qwen2.5", "phi4"]

# List of selected prompts for controlled testing (leave empty to use all)
SELECTED_PROMPTS = ["prompt_1.txt", "prompt_9.txt"]
# SELECTED_PROMPTS = []

# Ollama Server Config
OLLAMA_URL = "http://gammaweb08.medien.uni-weimar.de:11725"

# Initialize Ollama Client
client = ollama.Client(host=OLLAMA_URL)

def is_model_available(model_name):
    try:
        available_models = [m["name"] for m in client.list()["models"]]
        return model_name in available_models
    except Exception as e:
        logger.error(f"Error checking available models: {str(e)}")
        return False

def pull_model_if_needed(model_name):
    if is_model_available(model_name):
        logger.info(f"Model '{model_name}' is already available. Skipping download.")
    else:
        try:
            logger.info(f"Pulling model '{model_name}' from Ollama server...")
            client.pull(model_name)
            logger.info(f"Model '{model_name}' pulled successfully.")
        except Exception as e:
            logger.error(f"Failed to pull model '{model_name}': {str(e)}")

def get_prompt_files():
    all_prompts = sorted([f for f in os.listdir(PROMPT_DIR) if f.startswith("prompt_") and f.endswith(".txt")])
    if not all_prompts:
        logger.error("No prompt files found in the Prompts directory!")
        return []
    if SELECTED_PROMPTS:
        filtered_prompts = [p for p in all_prompts if p in SELECTED_PROMPTS]
        missing_prompts = [p for p in SELECTED_PROMPTS if p not in all_prompts]
        if missing_prompts:
            logger.warning(f"The following selected prompts were not found and will be skipped: {missing_prompts}")
        logger.info(f"Using selected prompts: {filtered_prompts}")
        return filtered_prompts
    logger.info(f"Using all available prompts: {all_prompts}")
    return all_prompts

def read_prompt(prompt_file):
    prompt_path = os.path.join(PROMPT_DIR, prompt_file)
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt_text = file.read().strip()
        if not prompt_text:
            logger.warning(f"Empty prompt file detected: {prompt_file}. Skipping.")
            return None
        if "{text}" not in prompt_text:
            logger.warning(f"Prompt '{prompt_file}' does not contain '{{text}}' placeholder. Ensure proper formatting.")
        return prompt_text
    except Exception as e:
        logger.error(f"Error reading prompt '{prompt_file}': {str(e)}")
        return None

def generate_cqs_with_retry(model, prompt_template, argument_text, max_retries=3):
    prompt = prompt_template.replace('"{text}"', f'"{argument_text}"')
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_ctx": 4096, "top_k": 40, "top_p": 0.9}
            )
            return response["message"]["content"]
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed for CQ generation: {str(e)}")
            time.sleep(2 * attempt)
    logger.error(f"Failed to generate CQs after {max_retries} attempts.")
    return "Error generating response"

def sanitize_model_name(model_name):
    return re.sub(r'[<>:"/\\|?*]', '-', model_name)

def setup_folders(model, prompt_number):
    safe_model_name = sanitize_model_name(model)
    model_dir = os.path.join(RESULTS_DIR, safe_model_name)
    prompt_dir = os.path.join(model_dir, str(prompt_number))
    sample_dir = os.path.join(prompt_dir, "Sample_CQs")
    validation_dir = os.path.join(prompt_dir, "Validation_CQs")
    test_dir = os.path.join(prompt_dir, "Test_CQs")
    for directory in [sample_dir, validation_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    return sample_dir, validation_dir, test_dir

def structure_output(whole_text):
    cqs_list = whole_text.split('\n')
    final, valid, not_valid = [], [], []
    for cq in cqs_list:
        if re.match(r'.*\?(\")?( )?(\([a-zA-Z0-9\.\'\-,\? ]*\))?([a-zA-Z \.,\"\']*)?(\")?$', cq):
            valid.append(cq)
        else:
            not_valid.append(cq)
    for text in not_valid:
        new_cqs = re.split(r'\?"', text + 'end')
        if len(new_cqs) > 1:
            for cq in new_cqs[:-1]:
                valid.append(cq + '?"')
    for cq in valid:
        occurrence = re.search(r'[A-Z]', cq)
        if occurrence:
            final.append(cq[occurrence.start():])
    if len(final) >= 3:
        return [{'id': i, 'cq': final[i]} for i in range(3)]
    logger.warning(f"Less than 3 valid CQs detected ({len(final)} found).")
    return []

def process_intervention(model, prompt_text, key, line):
    text = line["intervention"]
    cqs = generate_cqs_with_retry(model, prompt_text, text)
    structured_cqs = structure_output(cqs)
    clean_line = {
        "intervention_id": key,
        "intervention": line.get("intervention", ""),
        "dataset": line.get("dataset", ""),
        "cqs": structured_cqs
    }
    return key, clean_line

def extract_prompt_number(prompt_filename):
    match = re.search(r'prompt_(\d+)', prompt_filename)
    return int(match.group(1)) if match else None

def main():
    prompt_files = get_prompt_files()
    if not prompt_files:
        logger.error("No valid prompts found. Exiting program.")
        return

    logger.info(f"Starting CQ generation using {len(prompt_files)} prompts.")
    generated_outputs = []

    for model in MODELS:
        pull_model_if_needed(model)
        for prompt_file in prompt_files:
            prompt_number = extract_prompt_number(prompt_file)
            if prompt_number is None:
                logger.warning(f"Could not determine prompt number for {prompt_file}. Skipping.")
                continue

            prompt_text = read_prompt(prompt_file)
            if prompt_text is None:
                logger.warning(f"Skipping {prompt_file} due to errors.")
                continue

            sample_dir, validation_dir, test_dir = setup_folders(model, prompt_number)

            for dataset_name, dataset_file in DATASETS.items():
                dataset_path = os.path.join(DATASET_DIR, dataset_file)
                safe_model_name = sanitize_model_name(model)
                output_filename = f"{safe_model_name}_prompt{prompt_number}_{dataset_name}.json"

                if dataset_name == "sample":
                    output_path = os.path.join(sample_dir, output_filename)
                elif dataset_name == "validation":
                    output_path = os.path.join(validation_dir, output_filename)
                else:
                    output_path = os.path.join(test_dir, output_filename)

                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                results = {}
                total_interventions = len(data)

                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_key = {
                        executor.submit(process_intervention, model, prompt_text, key, line): key
                        for key, line in data.items()
                    }
                    with tqdm(total=total_interventions, desc=f"Generating CQs ({model} - Prompt {prompt_number} - {dataset_name})") as pbar:
                        for future in concurrent.futures.as_completed(future_to_key):
                            key = future_to_key[future]
                            try:
                                processed_key, processed_line = future.result()
                                results[processed_key] = processed_line
                            except Exception as e:
                                logger.error(f"Error processing intervention {key}: {str(e)}")
                            pbar.update(1)

                with open(output_path, "w", encoding="utf-8") as o:
                    json.dump(results, o, indent=4)
                logger.info(f"Generated CQs saved at: {output_path}")
                generated_outputs.append((dataset_path, output_path))

    logger.info("All CQ generation is complete.")
    reference_file = os.path.join(RESULTS_DIR, "generated_files.json")
    with open(reference_file, "w", encoding="utf-8") as f:
        json.dump(generated_outputs, f, indent=4)
    logger.info(f"Generated file paths saved in: {reference_file}")

if __name__ == "__main__":
    main()
