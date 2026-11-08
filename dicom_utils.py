
import numpy as np
import pydicom

from PIL import Image


def load_dicom_image(file_path):

    # Read DICOM file
    dicom_data = pydicom.dcmread(file_path)

    # Extract image pixels
    image_array = dicom_data.pixel_array.astype(
        np.float32
    )

    # Apply rescale parameters when available
    slope = float(
        getattr(dicom_data, "RescaleSlope", 1)
    )

    intercept = float(
        getattr(dicom_data, "RescaleIntercept", 0)
    )

    image_array = image_array * slope + intercept

    # Apply MONOCHROME1 inversion
    if getattr(dicom_data, "PhotometricInterpretation", "") == "MONOCHROME1":
        image_array = image_array.max() - image_array

    # Normalize pixel values to 0–255
    image_array -= image_array.min()

    max_value = image_array.max()

    if max_value > 0:
        image_array = image_array / max_value * 255

    image_array = image_array.astype(np.uint8)

    # Convert to PIL image
    image = Image.fromarray(image_array).convert("RGB")

    return image
