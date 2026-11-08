from PIL import Image
from torchvision import models, transforms
from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

# Settings
IMAGE_SIZE = 160
MODEL_PATH = Path("models/densenet121_xray.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# Load the trained model
def load_model():

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


# Predict an X-ray image
def predict_image(image_path):

    model, classes = load_model()

    image = Image.open(image_path).convert("RGB")

    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = (
            probabilities.max(dim=1)
        )

    predicted_class = classes[predicted_index.item()]
    confidence_value = confidence.item() * 100

    print("\n----- MedicalAI Prediction -----")
    print("Predicted class:", predicted_class)
    print(f"Model confidence: {confidence_value:.2f}%")
    print("--------------------------------")

    return predicted_class, confidence_value


if __name__ == "__main__":

    print("MedicalAI Prediction Module")
    print("Model file exists:", MODEL_PATH.exists())

    image_path = input(
        "\nEnter the X-ray image path: "
    ).strip().strip('"')

    if Path(image_path).exists():

        try:
            predict_image(image_path)

        except Exception as error:
            print("Prediction error:", error)

    else:
        print("Image file not found. Check the path.")
