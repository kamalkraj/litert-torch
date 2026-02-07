import os
import numpy as np
import transformers
import ai_edge_litert.interpreter as litert

# Reuse the template logic from your current setup
PREFIX = "task: sentence similarity | query: "

class EmbeddingGemmaTFLite:
    def __init__(self, model_path, tokenizer_path):
        print(f"Loading TFLite model: {model_path}")
        self.interpreter = litert.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    def encode(self, text):
        full_text = PREFIX + text
        inputs = self.tokenizer(full_text, return_tensors="pt", padding="max_length", max_length=2048, truncation=True)
        
        # Handle single or dual input signatures
        if len(self.input_details) >= 2:
            self.interpreter.set_tensor(self.input_details[0]['index'], inputs["input_ids"].numpy().astype(self.input_details[0]['dtype']))
            self.interpreter.set_tensor(self.input_details[1]['index'], inputs["attention_mask"].numpy().astype(self.input_details[1]['dtype']))
        else:
            self.interpreter.set_tensor(self.input_details[0]['index'], inputs["input_ids"].numpy().astype(self.input_details[0]['dtype']))

        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # L2 Normalize as the model was exported without the normalization layer
        norm = np.linalg.norm(output)
        if norm > 1e-9:
            output = output / norm
        return output

def main():
    model_path = "embedding_gemma_no_normalize_q8.tflite"
    tokenizer_path = "embeddinggemma-300m" if os.path.exists("embeddinggemma-300m") else "google/embeddinggemma-300m"
    
    sentences = ["The cat sits outside", "A man is playing a guitar"]
    
    try:
        model = EmbeddingGemmaTFLite(model_path, tokenizer_path)
        
        for text in sentences:
            embedding = model.encode(text)
            print(f"\nInput: {text}")
            print(f"Embedding Shape: {embedding.shape}")
            print(f"First 10 values: {embedding[:10]}")
            print(f"Last 10 values:  {embedding[-10:]}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
