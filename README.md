# Mechanical Parts Classifier 🛠️

A Deep Learning solution for the **SolidWorks AI Hackathon** to detect and count mechanical parts from top-down assembly images. This project utilizes a **ResNet18** model trained from scratch to classify image quadrants into 5 categories: `bolt`, `nut`, `washer`, `locatingpin`, and `empty`.

## 📌 Project Overview
The pipeline consists of three main stages:
1.  **Data Preprocessing**: Original high-res images are split into 4 quadrants. Labels are assigned based on bounding box centers using `train_bboxes.csv`.
2.  **Model Training**: A ResNet18 architecture is trained from scratch on the processed quadrant dataset.
3.  **Inference & Counting**: The trained model predicts the class of each quadrant in test images and aggregates the counts for a final CSV submission.

## 📂 Project Structure
```
├── models/                  # Saved model weights
│   └── resnet18_scratch_100.pth
├── processed_data/          # Generated training data (quadrants)
├── train/                   # Raw training images
├── test/                    # Test images for inference
├── inference.py               # Main inference script (generates submission.csv)
├── preprocess_data.py       # Splits images and generates labeled dataset
├── train.py                 # Trains the ResNet18 model
├── train_bboxes.csv         # Bounding box annotations
└── requirements.txt         # (Optional) List of dependencies
```

## ⚙️ Setup & Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch torchvision pandas opencv-python tqdm pillow
```

## 🚀 Usage

### 1. Data Preprocessing
Prepare the training data by splitting images into labeled quadrants.
```bash
python preprocess_data.py
```
*Output: Populates `processed_data/` with labeled sub-folders.*

### 2. Training
Train the ResNet18 model on the processed dataset.
```bash
python train.py
```
*Output: Saves the best model to `models/resnet18_scratch_100.pth`.*

### 3. Inference
Run the model on the `test/` directory to generate the submission file.
```bash
python inference.py
```
*Output: Generates `submission_plainuj.csv` with the count of parts per image.*

## 📊 Classes
- **Bolt**
- **Nut**
- **Washer**
- **Locating Pin**
- **Empty**

## 💻 Tech Stack
- **PyTorch**: Model training and inference.
- **OpenCV & PIL**: Image processing.
- **Pandas**: Data handling and CSV generation.
