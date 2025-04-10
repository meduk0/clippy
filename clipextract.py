import open_clip # load clip model and its tokenizers
from PIL import Image # add image processing capabilities (PILLOW)
from pathlib import Path
import numpy as np # For numerical arrays (used to store features).
import torch # for model inference
from typing import Union, List

class CLIPFeatureExtractor:
    def __init__(self, model_name="ViT-B-32-quickgelu", pretrained="openai"):
        # Load OpenCLIP model and preprocessing
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained) # load the model and its pipeline
        self.tokenizer = open_clip.get_tokenizer(model_name) #  load the tokenizer for text input

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pass the model either to the gpu or the cpu
        self.model = self.model.to(self.device)
        self.model.eval() # Switch the model to inference mode

        print(f"CLIP model '{model_name}' loaded on: {self.device}")

    def extract_image_features(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")
        image_tensor = self.preprocess(img).unsqueeze(0).to(self.device) # Applies CLIP preprocessing
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor) # encode the image into a vector (result is image_tensor)
        feature = image_features.cpu().numpy()[0] # normalize the vector (will be used for cosine similarity)
        return feature / np.linalg.norm(feature)

    def extract_text_features(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            text = [text] # check for the string
        tokens = self.tokenizer(text).to(self.device) # extract the token from the txt querry
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        features = text_features.cpu().numpy()
        if len(features) == 1:
            return features[0] / np.linalg.norm(features[0])
        else:
            return np.array([f / np.linalg.norm(f) for f in features])

    def extract(self, img: Image.Image) -> np.ndarray: # a simple wrapper for the image extraction
        return self.extract_image_features(img)



if __name__ == '__main__':
    import os
    os.makedirs("static/clipfeature", exist_ok=True)

    fe = CLIPFeatureExtractor()

    for img_path in sorted(Path("static/img").glob("*.jpg")):
        print(f"Processing {img_path}")
        try:
            feature = fe.extract(Image.open(img_path))
            feature_path = Path("static/clipfeature") / (img_path.stem + ".npy")
            np.save(feature_path, feature)
            print(f"Saved feature to {feature_path}")

        except Exception as e:
            print(f"Skipping {img_path.name}: {e}")
