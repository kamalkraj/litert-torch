import os
import numpy as np
import transformers
import ai_edge_litert.interpreter as litert

# Prompt templates for different tasks
PROMPT_TEMPLATES = {
    "query": "task: search result | query: ",
    "document": "title: {title} | text: ",
    "BitextMining": "task: search result | query: ",
    "Clustering": "task: clustering | query: ",
    "Classification": "task: classification | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "Reranking": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-query": "task: search result | query: ",
    "Retrieval-document": "title: {title} | text: ",
    "STS": "task: sentence similarity | query: ",
    "Summarization": "task: summarization | query: "
}

class EmbeddingGemmaTFLite:
    def __init__(self, model_path, tokenizer_path):
        """Initializes the TFLite interpreter and tokenizer."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found at {model_path}")
            
        print(f"Loading TFLite model: {model_path}")
        self.interpreter = litert.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    def encode(self, text, task_type=None, title="none", embedding_dim=None):
        """Encodes text into an embedding vector.
        
        Args:
            text: The input text.
            task_type: Optional task name from PROMPT_TEMPLATES.
            title: Title for document tasks (defaults to "none").
            embedding_dim: Optional dimension to truncate to (e.g. 512, 256).
        """
        prompt_prefix = ""
        if task_type:
            if task_type not in PROMPT_TEMPLATES:
                raise ValueError(f"Unknown task type: '{task_type}'. Available: {list(PROMPT_TEMPLATES.keys())}")
            
            template = PROMPT_TEMPLATES[task_type]
            if "{title}" in template:
                prompt_prefix = template.format(title=title)
            else:
                prompt_prefix = template
        
        full_text = prompt_prefix + text
        
        # Tokenize and pad/truncate to 2048 (model's expected sequence length)
        inputs = self.tokenizer(
            full_text, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=2048, 
            truncation=True
        )
        
        # Set inputs for TFLite model
        if len(self.input_details) >= 2:
            self.interpreter.set_tensor(self.input_details[0]['index'], inputs["input_ids"].numpy().astype(self.input_details[0]['dtype']))
            self.interpreter.set_tensor(self.input_details[1]['index'], inputs["attention_mask"].numpy().astype(self.input_details[1]['dtype']))
        else:
            self.interpreter.set_tensor(self.input_details[0]['index'], inputs["input_ids"].numpy().astype(self.input_details[0]['dtype']))

        # Run inference
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Truncate if specified (Matryoshka learning)
        if embedding_dim is not None:
            output = output[:embedding_dim]

        # L2 Normalize the final vector
        norm = np.linalg.norm(output)
        if norm > 1e-9:
            return output / norm
        return output

def main():
    # Configuration
    model_path = "embedding_gemma_no_normalize_q8.tflite"
    tokenizer_path = "embeddinggemma-300m"
    
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "google/embeddinggemma-300m"

    try:
        model = EmbeddingGemmaTFLite(model_path, tokenizer_path)
        
        print("\n--- Example 1: Semantic Textual Similarity (STS) with truncation ---")
        text1 = "The cat sits outside"
        text2 = "A man is playing a guitar"
        
        # Full 768 dimension
        emb1 = model.encode(text1, task_type="STS")
        # Truncated to 256 dimension
        emb2_truncated = model.encode(text2, task_type="STS", embedding_dim=256)
        
        print(f"Sentence 1 (full) dim: {len(emb1)}")
        print(f"Sentence 2 (truncated) dim: {len(emb2_truncated)}")
        
        print("\n--- Example 2: Retrieval with Title ---")
        query = "What is the capital of France?"
        doc_text = "Paris is the capital and most populous city of France."
        doc_title = "City of Paris"
        
        query_emb = model.encode(query, task_type="Retrieval-query")
        doc_emb = model.encode(doc_text, task_type="Retrieval-document", title=doc_title)
        
        retrieval_score = np.dot(query_emb, doc_emb)
        print(f"Query: {query}")
        print(f"Document Title: {doc_title}")
        print(f"Retrieval Score: {retrieval_score:.4f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()