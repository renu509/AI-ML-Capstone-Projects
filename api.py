
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image, UnidentifiedImageError
from pathlib import Path
from io import BytesIO


# =====================================
# CONFIGURATION
# =====================================

app = FastAPI(
    title="MedicalAI API",
    description="Educational chest X-ray classification API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_PATH = Path("models/densenet121_xray.pth")
DEVICE = torch.device("cpu")
IMAGE_SIZE = 160

# =====================================
# LOAD TRAINED MODEL
# =====================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    classes = checkpoint["classes"]

    model = models.densenet121(weights=None)

    model.classifier = nn.Linear(
        model.classifier.in_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model, classes


model, classes = load_model()


# =====================================
# IMAGE PREPROCESSING
# =====================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================
# HOME ENDPOINT
# =====================================

@app.get("/")
def home():

    return {
        "message": "Welcome to MedicalAI API",
        "status": "running",
        "model": "DenseNet121"
    }


# =====================================
# HEALTH CHECK
# =====================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# =====================================
# IMAGE PREDICTION ENDPOINT
# =====================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_types = [
        "image/jpeg",
        "image/png"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():

            output = model(input_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )

        predicted_class = classes[
            predicted_index.item()
        ]

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": round(
                confidence.item() * 100, 2
            ),
            "model": "DenseNet121",
            "notice": (
                "Experimental educational output only. "
                "Not for clinical diagnosis or treatment."
            )
        }

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    except HTTPException:
        raise

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Image processing failed."
        )
