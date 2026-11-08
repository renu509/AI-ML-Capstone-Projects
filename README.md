
# MedicalAI — AI-Assisted Chest X-Ray Classification

MedicalAI is an educational AI project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a fine-tuned DenseNet121 model.

The project combines deep learning, image preprocessing, visual explanations, a web interface, a REST API, Docker deployment, and a doctor-review workflow.

> **Important:** This is an experimental educational prototype. It is not clinically validated and must not be used for diagnosis, treatment, or medical decision-making.

## Features

- Chest X-ray classification using DenseNet121
- PyTorch model inference
- Image upload through a Streamlit interface
- FastAPI REST API for predictions
- Grad-CAM heatmaps to visualize influential image regions
- Basic DICOM image preprocessing utility
- SQLite-based reviewer feedback and review status
- Docker and Docker Compose deployment
- Automated API tests using pytest

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch, Torchvision |
| Model | DenseNet121 |
| Image processing | Pillow, pydicom |
| Web interface | Streamlit |
| REST API | FastAPI, Uvicorn |
| Database | SQLite |
| Testing | pytest |
| Containerization | Docker, Docker Compose |

## Project Architecture

```text
Chest X-ray Image
       |
       v
Image Preprocessing
       |
       v
DenseNet121 Model
       |
       v
NORMAL / PNEUMONIA Prediction
       |
       +------> Confidence Score
       |
       +------> Grad-CAM Heatmap
       |
       +------> Doctor Review Workflow
       |
       v
Streamlit UI / FastAPI
```

## Project Structure

```text
MedicalAI/
├── app.py
├── api.py
├── train.py
├── predict.py
├── gradcam.py
├── dicom_utils.py
├── review_db.py
├── models/
│   └── densenet121_xray.pth
├── tests/
│   └── test_api.py
├── utils/
├── outputs/
├── docs/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Local Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd MedicalAI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Ensure the trained model checkpoint is available at:

```text
models/densenet121_xray.pth
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

### 5. Run the FastAPI service

```bash
python -m uvicorn api:app --reload
```

Open the API documentation:

http://localhost:8000/docs

## API Usage

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Predict from an Image

```http
POST /predict
```

Upload a JPG or PNG image using the `file` form field.

Example response:

```json
{
  "filename": "sample_xray.jpeg",
  "predicted_class": "PNEUMONIA",
  "confidence": 89.39,
  "model": "DenseNet121",
  "notice": "Experimental educational output only. Not for clinical diagnosis or treatment."
}
```

The confidence value represents the model's output probability. It does not establish diagnostic certainty.

## Docker Deployment

Build and start the application:

```bash
docker compose up --build -d
```

Check the service:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs --tail=30
```

Stop the service:

```bash
docker compose down
```

API documentation:

http://localhost:8000/docs

## Automated Testing

Run the test suite:

```bash
pytest -v
```

The current API tests cover:

- Home endpoint
- Health endpoint
- Successful image prediction
- Invalid file type
- Empty file upload

## Model Evaluation

The model was trained using a subset of the Kaggle Chest X-Ray Images (Pneumonia) dataset.

The current experimental evaluation showed limited performance, particularly in recognizing NORMAL images. The model tends to predict PNEUMONIA and requires further training and evaluation.

Grad-CAM visualizations show regions that influenced the model output. They are not proof of disease presence or medical evidence.

## Limitations

- Educational prototype, not a clinical product
- Binary classification only: NORMAL and PNEUMONIA
- Not designed for CT or MRI classification
- No implemented 3D segmentation
- Limited DICOM support; not validated across all DICOM formats
- Model performance requires improvement and independent validation
- Prediction confidence is not equivalent to clinical certainty
- Doctor-review workflow is a prototype and does not replace clinical review

## Future Improvements

- Improve model training and class-specific evaluation
- Add stronger data validation and model monitoring
- Expand DICOM compatibility and testing
- Improve API logging and error handling
- Add authentication and access control
- Expand automated testing and deployment checks
- Explore segmentation as a separate future feature

## Acknowledgements

- PyTorch and Torchvision
- FastAPI
- Streamlit
- Kaggle Chest X-Ray Images (Pneumonia) dataset

## Disclaimer

MedicalAI is intended solely for educational and research purposes. It is not a medical device and must not be used to diagnose, treat, or prevent any disease.
