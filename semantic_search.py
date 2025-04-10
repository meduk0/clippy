import os
import json # manipulate JSON data
import numpy as np
import torch
import faiss # fast
import re
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Any # for code readability
import pickle # Serializes and deserializes Python objects (used to save/load embeddings).

class SemanticSearch:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the semantic search engine.

        Args:
            model_name: The name of the transformer model to use for embeddings
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Initialize variables for storing paragraphs and index
        self.paragraphs = []
        self.index = None
        self.embeddings = None

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling to get sentence embeddings.

        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask from tokenization

        Returns:
            Sentence embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        # Tokenize and prepare inputs
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=512,
                                       return_tensors='pt').to(self.device)

        # Get model output
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Mean pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.cpu().numpy()

    def batch_get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Get embeddings for a list of texts using batching.

        Args:
            texts: List of input texts
            batch_size: Size of batches for processing

        Returns:
            Array of embeddings
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize and prepare inputs
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True,
                                           max_length=512, return_tensors='pt').to(self.device)

            # Get model output
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Mean pooling
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            all_embeddings.append(sentence_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def load_paragraphs_from_json(self, json_file: str) -> None:
        """
        Load paragraphs from a JSON file.

        Args:
            json_file: Path to the JSON file containing paragraphs
        """
        print(f"Loading paragraphs from {json_file}")
        self.paragraphs = []

        with open(json_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    # Parse the JSON object
                    item = json.loads(line)

                    # Extract title and text
                    title = item.get("title", "")
                    text = item.get("text", "")

                    if text:
                        self.paragraphs.append({
                            "title": title,
                            "text": text,
                            "full_item": item
                        })
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line as JSON: {line[:100]}...")
                except Exception as e:
                    print(f"Error processing line: {str(e)}")

        print(f"Loaded {len(self.paragraphs)} paragraphs")

    def build_index(self, force_rebuild: bool = False, index_file: str = "faiss_index.bin",
                    embeddings_file: str = "embeddings.pkl") -> None:
        """
        Build a FAISS index from paragraphs.

        Args:
            force_rebuild: Whether to force rebuilding the index
            index_file: File to save/load the FAISS index
            embeddings_file: File to save/load the embeddings
        """
        if not self.paragraphs:
            raise ValueError("No paragraphs loaded. Call load_paragraphs_from_json first.")

        # Check if index already exists
        if not force_rebuild and os.path.exists(index_file) and os.path.exists(embeddings_file):
            print(f"Loading existing index from {index_file}")
            self.index = faiss.read_index(index_file)

            print(f"Loading existing embeddings from {embeddings_file}")
            with open(embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)

            return

        print("Building index...")

        # Extract text from paragraphs
        texts = [p["text"] for p in self.paragraphs]

        # Get embeddings in batches
        self.embeddings = self.batch_get_embeddings(texts)

        # Build FAISS index
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)

        # Use GPU for FAISS if available
        if self.device == "cuda":
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # Add embeddings to index
        self.index.add(self.embeddings)

        # Save index and embeddings
        faiss.write_index(faiss.index_gpu_to_cpu(self.index) if self.device == "cuda" else self.index, index_file)

        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)

        print(f"Index built and saved to {index_file}")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for paragraphs similar to the query.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of top matching paragraphs with scores
        """
        if not self.index:
            raise ValueError("Index not built. Call build_index first.")

        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Search in the index
        scores, indices = self.index.search(query_embedding, top_k)

        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.paragraphs):
                results.append({
                    "rank": i + 1,
                    "similarity_score": float(score),
                    "title": self.paragraphs[idx]["title"],
                    "text": self.paragraphs[idx]["text"],
                    "full_item": self.paragraphs[idx]["full_item"]
                })

        return results

def main():
    parser = argparse.ArgumentParser(description="Semantic search for paragraphs in JSON files")
    parser.add_argument("--json_file", required=True, help="Path to JSON file containing paragraphs")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuilding the index")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top results to return")
    parser.add_argument("--index_file", default="faiss_index.bin", help="Path to index file")
    parser.add_argument("--embeddings_file", default="embeddings.pkl", help="Path to embeddings file")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use (default: auto)")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Transformer model to use for embeddings")

    args = parser.parse_args()

    # Initialize search engine
    searcher = SemanticSearch(model_name=args.model, device=args.device)

    # Load paragraphs
    searcher.load_paragraphs_from_json(args.json_file)

    # Build index
    searcher.build_index(
        force_rebuild=args.force_rebuild,
        index_file=args.index_file,
        embeddings_file=args.embeddings_file
    )

    # If query is provided, search
    if args.query:
        results = searcher.search(args.query, top_k=args.top_k)
        print(f"\nTop {len(results)} results for query: '{args.query}'")
        for result in results:
            print(f"\n{'-' * 80}")
            print(f"Rank: {result['rank']}, Score: {result['similarity_score']:.4f}")
            print(f"Title: {result['title']}")
            print(f"Text: {result['text'][:200]}...")
    else:
        print("\nNo query provided. Use --query to search the index.")
        while True:
            query = input("\nEnter a search query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break

            results = searcher.search(query, top_k=args.top_k)
            print(f"\nTop {len(results)} results for query: '{query}'")
            for result in results:
                print(f"\n{'-' * 80}")
                print(f"Rank: {result['rank']}, Score: {result['similarity_score']:.4f}")
                print(f"Title: {result['title']}")
                print(f"Text: {result['text'][:200]}...")

if __name__ == "__main__":
    main()
