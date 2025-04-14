import numpy as np
from PIL import Image
from clipextract import CLIPFeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import os
import json
# Import the semantic search module
from semantic_search import SemanticSearch
app = Flask(__name__)
# Ensure upload directory exists
os.makedirs("static/uploaded", exist_ok=True)
# Initialize the feature extractor for images
fe = CLIPFeatureExtractor()
# Initialize semantic search for text
semantic_searcher = SemanticSearch()
# Set the JSON file path - update this to your actual path
json_file_path ="static/wikipedia_paragraphs_train.json"
# Load paragraphs and build index (will load from cache if available)
semantic_searcher.load_paragraphs_from_json(json_file_path)
semantic_searcher.build_index()
# Read image features and store the paths for processing
features = []
img_paths = []
for feature_path in Path("./static/clipfeature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)
@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request is for text search or image upload
        if 'query_text' in request.form and request.form['query_text'].strip():
            # Text-based search
            query_text = request.form['query_text']
            # Get both image search and semantic paragraph search results
            # 1. CLIP image search
            query = fe.extract_text_features(query_text)
            similarities = np.dot(features, query)
            ids = np.argsort(-similarities)[:50]  # Top 50 results
            image_scores = [(similarities[id], os.path.relpath(img_paths[id], "static")) for id in ids]
            # 2. Semantic paragraph search
            semantic_results = semantic_searcher.search(query_text, top_k=10)

            # Convert semantic results to proper format for the template
            formatted_semantic_results = []
            for result in semantic_results:
                formatted_semantic_results.append({
                    "rank": result['rank'],
                    "score": result['similarity_score'],
                    "title": result['title'],
                    "text": result['text']
                })

            return render_template('index3.html',
                                  query_text=query_text,
                                  scores=image_scores,
                                  semantic_results=formatted_semantic_results)
        elif 'query_img' in request.files:
            # Image-based search
            file = request.files['query_img']
            # Save query image
            img = Image.open(file.stream)  # PIL image
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
            img.save(uploaded_img_path)
            # Run search
            query = fe.extract(img)
            # Compute cosine similarity (using dot product of normalized vectors)
            similarities = np.dot(features, query)
            # Sort by highest similarity (most similar first)
            ids = np.argsort(-similarities)[:30]  # Top 30 results, negative for descending order
            scores = [(similarities[id], os.path.relpath(img_paths[id], "static")) for id in ids]
            return render_template('index3.html',
                                  query_path=uploaded_img_path,
                                  scores=scores)
    # GET request - just show the form
    return render_template('index3.html')
if __name__ == "__main__":
    print("Starting CLIP and semantic search server...")
    app.run("0.0.0.0", threaded=True)  # allow multiple threads
