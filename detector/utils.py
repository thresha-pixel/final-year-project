# detector/utils.py
import os
import numpy as np
import cv2  # Use OpenCV like in your notebook for consistency
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as BaseDepthwiseConv2D

class CustomDepthwiseConv2D(BaseDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

MODEL_PATH = r"C:\Users\thres\OneDrive\Documents\Desktop\my_projects\Frontend image_forgery\mobilenet_project.h5"

_model = None

def load_custom_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):


            
            raise FileNotFoundError(f"Model not found → {MODEL_PATH}")
        print(f"Loading model: {MODEL_PATH}")
        _model = load_model(
            MODEL_PATH,
            custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
            compile=False
        )
        print("Model loaded OK")
    return _model

def predict_image(img_path):
    model = load_custom_model()

    # Load with cv2 like in notebook (BGR order)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]  # [p0, p1]

    # Mapping from your notebook: Fake = p0, Real = p1
    confidence = {
        "Fake": round(float(preds[0]) * 100, 2),
        "Real": round(float(preds[1]) * 100, 2)
    }

    # Decide verdict like notebook
    if confidence["Real"] > confidence["Fake"]:
        result = "Real"
        conf = confidence["Real"]
    else:
        result = "Fake"
        conf = confidence["Fake"]

    # Debug prints (remove after testing)
    print("\n" + "="*50)
    print("Image:", os.path.basename(img_path))
    print("Raw probs [Fake, Real]:", [confidence["Fake"], confidence["Real"]])
    print("Predicted:", result, f"({conf}%)")
    print("="*50 + "\n")

    return {
        "result": result,
        "confidence": conf,
        "details": confidence
    }