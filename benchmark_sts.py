import gzip
import json
import os
import numpy as np
import scipy.stats
import torch
import transformers
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import ai_edge_litert.interpreter as litert
from tqdm import tqdm

def load_stsb_data():
    print("Downloading STS Benchmark test set...")
    # MTEB stores data in test.jsonl.gz
    file_path = hf_hub_download(repo_id="mteb/stsbenchmark-sts", filename="test.jsonl.gz", repo_type="dataset")
    
    samples = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # MTEB format: "sentence1", "sentence2", "score"
            samples.append({
                "s1": data["sentence1"],
                "s2": data["sentence2"],
                "score": float(data["score"])
            })
    print(f"Loaded {len(samples)} pairs.")
    return samples

def get_tflite_embedding(interpreter, tokenizer, text, input_details, output_details):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=2048, truncation=True)
    
    # Set inputs
    if len(input_details) >= 2:
        interpreter.set_tensor(input_details[0]['index'], inputs["input_ids"].numpy().astype(input_details[0]['dtype']))
        interpreter.set_tensor(input_details[1]['index'], inputs["attention_mask"].numpy().astype(input_details[1]['dtype']))
    else:
        interpreter.set_tensor(input_details[0]['index'], inputs["input_ids"].numpy().astype(input_details[0]['dtype']))
    
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output

def main():
    # Default to the best model found
    tflite_model_path = "embedding_gemma_no_normalize_q8.tflite"
    checkpoint_dir = "embeddinggemma-300m"
    limit = 100
    dims = [768, 512, 256]
    
    # Check if model exists
    if not os.path.exists(tflite_model_path):
        print(f"Error: Model file {tflite_model_path} not found.")
        return

    # 1. Load Data
    try:
        samples = load_stsb_data()
        if limit and limit < len(samples):
            print(f"Limiting benchmark to first {limit} samples.")
            samples = samples[:limit]
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    # 2. Setup TFLite
    print(f"Loading TFLite model: {tflite_model_path}")
    interpreter = litert.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
    
    # 3. Setup Baseline (HF)
    print(f"Loading HF model: {checkpoint_dir}")
    hf_model = SentenceTransformer(checkpoint_dir, device="cpu")
    
    # 4. Run Benchmark
    print(f"Running inference on {len(samples)} pairs with dims {dims}...")
    
    results_tflite = {d: [] for d in dims}
    results_hf = {d: [] for d in dims}
    ground_truths = []
    
    for sample in tqdm(samples):
        s1, s2, score = sample['s1'], sample['s2'], sample['score']
        ground_truths.append(score)
        
        # TFLite Inference (Raw)
        prefix = "task: sentence similarity | query: "
        raw_tflite_1 = get_tflite_embedding(interpreter, tokenizer, prefix + s1, input_details, output_details)
        raw_tflite_2 = get_tflite_embedding(interpreter, tokenizer, prefix + s2, input_details, output_details)
        
        # HF Inference (Raw/Default)
        raw_hf_1 = hf_model.encode(s1, prompt_name="STS", convert_to_numpy=True)
        raw_hf_2 = hf_model.encode(s2, prompt_name="STS", convert_to_numpy=True)
        
        for d in dims:
            # TFLite Slicing & Normalization
            v1 = raw_tflite_1[:d]
            v2 = raw_tflite_2[:d]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-9 and norm2 > 1e-9:
                sim = np.dot(v1, v2) / (norm1 * norm2)
            else:
                sim = 0.0
            results_tflite[d].append(sim)
            
            # HF Slicing & Normalization
            h1 = raw_hf_1[:d]
            h2 = raw_hf_2[:d]
            hn1 = np.linalg.norm(h1)
            hn2 = np.linalg.norm(h2)
            if hn1 > 1e-9 and hn2 > 1e-9:
                sim_h = np.dot(h1, h2) / (hn1 * hn2)
            else:
                sim_h = 0.0
            results_hf[d].append(sim_h)
        
    # 5. Calculate Spearman Correlation
    print("\nResults (Spearman Correlation x100):")
    print(f"{'Dim':<5} | {'HF Model':<10} | {'TFLite Model':<12} | {'Diff':<8}")
    print("-" * 45)
    
    for d in dims:
        spearman_tflite, _ = scipy.stats.spearmanr(results_tflite[d], ground_truths)
        spearman_hf, _ = scipy.stats.spearmanr(results_hf[d], ground_truths)
        diff = abs(spearman_hf - spearman_tflite)
        print(f"{d:<5} | {spearman_hf*100:<10.2f} | {spearman_tflite*100:<12.2f} | {diff:.4f}")

if __name__ == "__main__":
    main()
