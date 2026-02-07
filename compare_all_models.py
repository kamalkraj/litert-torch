import os
import glob
import numpy as np
import torch
import transformers
import ai_edge_litert.interpreter as litert
from litert_torch.generative.examples.embedding_gemma import embedding_gemma
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

def main():
    checkpoint_dir = "embeddinggemma-300m"
    
    sentences = [
        "What are the symptoms of malaria?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "To be or not to be, that is the question.",
        "Quantum computing utilizes qubits to perform calculations.",
        "The weather in London is often rainy.",
        "Python is a versatile programming language.",
        "Economic inflation affects the purchasing power of currency.",
        "Photosynthesis is the process by which plants make food.",
        "The history of Rome spans over two and a half thousand years."
    ]
    
    # 1. Load Tokenizer & Baseline Models
    print(f"Loading tokenizer and baseline models from {checkpoint_dir}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
    
    # Hugging Face (SentenceTransformer)
    hf_model = SentenceTransformer(checkpoint_dir, device="cpu")
    
    # Custom PyTorch (Raw)
    torch_model = embedding_gemma.build_model(checkpoint_dir)
    torch_model.eval()
    
    # 2. Find TFLite models
    tflite_files = sorted(glob.glob("*.tflite"))
    print(f"\nFound {len(tflite_files)} TFLite models: {tflite_files}\n")
    
    model_metrics = {model: {'cos_sim_sum': 0.0, 'diff_raw_sum': 0.0, 'count': 0, 'normalized': False, 'norm_val': 0.0} for model in tflite_files}
    
    print(f"Running comparison on {len(sentences)} sentences...")
    
    for text in sentences:
        # Precompute Baselines
        hf_output = hf_model.encode(text, convert_to_numpy=True)
        if hf_output.ndim == 1: hf_output = hf_output.reshape(1, -1)
        
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=2048, truncation=True)
        tokens_pt = inputs["input_ids"]
        mask_pt = inputs["attention_mask"]
        
        with torch.no_grad():
            torch_output_raw = torch_model(tokens_pt, attention_mask=mask_pt).numpy()
            
        for model_path in tflite_files:
            try:
                interpreter = litert.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                if len(input_details) >= 2:
                    interpreter.set_tensor(input_details[0]['index'], tokens_pt.numpy().astype(input_details[0]['dtype']))
                    interpreter.set_tensor(input_details[1]['index'], mask_pt.numpy().astype(input_details[1]['dtype']))
                else:
                    interpreter.set_tensor(input_details[0]['index'], tokens_pt.numpy().astype(input_details[0]['dtype']))
                    
                interpreter.invoke()
                tflite_output = interpreter.get_tensor(output_details[0]['index'])
                
                tflite_norm = np.linalg.norm(tflite_output[0])
                is_normalized = abs(tflite_norm - 1.0) < 0.01
                
                model_metrics[model_path]['normalized'] = is_normalized
                model_metrics[model_path]['norm_val'] = tflite_norm 
                
                if is_normalized:
                    tflite_output_norm = tflite_output
                else:
                    tflite_output_norm = tflite_output / tflite_norm
                    
                # Cosine Similarity vs HF
                cos_sim = np.dot(hf_output[0], tflite_output_norm[0])
                model_metrics[model_path]['cos_sim_sum'] += cos_sim
                
                # Diff vs PyTorch Raw (only if not normalized)
                if not is_normalized:
                    diff = np.max(np.abs(torch_output_raw - tflite_output))
                    model_metrics[model_path]['diff_raw_sum'] += diff
                
                model_metrics[model_path]['count'] += 1
                
            except Exception as e:
                pass 

    # 3. Print Aggregated Results
    results = []
    for model_path in tflite_files:
        metrics = model_metrics[model_path]
        count = metrics['count']
        if count > 0:
            avg_cos = metrics['cos_sim_sum'] / count
            avg_diff = metrics['diff_raw_sum'] / count if not metrics['normalized'] else "N/A (Norm)"
            norm_status = "Yes" if metrics['normalized'] else f"No (~{metrics['norm_val']:.1f})"
            
            results.append([
                model_path,
                norm_status,
                f"{avg_cos:.6f}",
                f"{avg_diff if isinstance(avg_diff, str) else f'{avg_diff:.4f}'}"
            ])
        else:
             results.append([model_path, "Error", "-", "-"])

    headers = ["Model File", "Normalized?", "Avg Cos Sim (vs HF)", "Avg Max Diff (vs PT Raw)"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
