
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms


# Configuration
IMAGE_SIZE = 160
MODEL_PATH = Path("models/densenet121_xray.pth")
OUTPUT_DIR = Path("outputs")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def generate_gradcam(image_path):

    model, classes = load_model()

    activations = []
    gradients = []

    # Capture feature maps
    def forward_hook(module, input, output):
        activations.append(output)

    # Capture gradients
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.features.denseblock4

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        original_image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = image_transform(
            original_image
        ).unsqueeze(0).to(DEVICE)

        model.zero_grad()

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_index = output.argmax(dim=1).item()
        predicted_class = classes[predicted_index]

        # Backpropagate predicted class score
        score = output[0, predicted_index]
        score.backward()

        feature_maps = activations[0]
        feature_gradients = gradients[0]

        # Calculate Grad-CAM weights
        weights = feature_gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * feature_maps
        ).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize heatmap
        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min() + 1e-8
        )

        confidence = (
            probabilities[0, predicted_index].item()
            * 100
        )

        # Create visualization
        original_image = original_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        original_array = np.array(
            original_image
        )

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(original_array)
        plt.title("Original X-ray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(original_array)
        plt.imshow(
            cam,
            cmap="jet",
            alpha=0.45
        )
        plt.title("Grad-CAM Heatmap")
        plt.axis("off")

        plt.tight_layout()

        output_path = OUTPUT_DIR / "gradcam_result.png"

        plt.savefig(
            output_path,
            bbox_inches="tight",
            dpi=150
        )

        plt.close()

        print("\n----- Grad-CAM Result -----")
        print("Predicted class:", predicted_class)
        print(f"Model confidence: {confidence:.2f}%")
        print("Heatmap saved:", output_path)
        print("---------------------------")

    finally:
        forward_handle.remove()
        backward_handle.remove()


if __name__ == "__main__":

    image_path = input(
        "Enter the X-ray image path: "
    ).strip().strip('"')

    if Path(image_path).is_file():

        try:
            generate_gradcam(image_path)

        except Exception as error:
            print("Grad-CAM error:", error)

    else:
        print("Image file not found.")
