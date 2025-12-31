import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# =====================
# CONFIG
# =====================
TEST_DIR = "test"                         # test images folder
MODEL_PATH = "models/resnet18_scratch_100.pth"
OUTPUT_CSV = "submission.csv"

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MUST match training ImageFolder class order
CLASSES = ['bolt', 'empty', 'locatingpin', 'nut', 'washer']

# =====================
# MODEL LOADING
# =====================
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# =====================
# TRANSFORMS (same as training)
# =====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# =====================
# QUADRANT SPLIT
# =====================
def split_quadrants(img):
    w, h = img.size
    mx, my = w // 2, h // 2
    return [
        img.crop((0, 0, mx, my)),       # TL
        img.crop((mx, 0, w, my)),       # TR
        img.crop((0, my, mx, h)),       # BL
        img.crop((mx, my, w, h))        # BR
    ]

# =====================
# MAIN INFERENCE
# =====================
def main():
    model = load_model()
    image_files = sorted([
        f for f in os.listdir(TEST_DIR)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])

    results = []

    print(f"🚀 Running inference on {len(image_files)} images...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(TEST_DIR, img_name)
        img = Image.open(img_path).convert("RGB")

        quadrants = split_quadrants(img)

        counts = {
            "bolt": 0,
            "locatingpin": 0,
            "nut": 0,
            "washer": 0
        }

        for q in quadrants:
            x = transform(q).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(x)
                pred_idx = torch.argmax(logits, dim=1).item()

            pred_class = CLASSES[pred_idx]

            if pred_class != "empty":
                counts[pred_class] += 1

        results.append({
            "image_name": img_name,
            "bolt": counts["bolt"],
            "locatingpin": counts["locatingpin"],
            "nut": counts["nut"],
            "washer": counts["washer"]
        })

    df = pd.DataFrame(results)
    df = df[["image_name", "bolt", "locatingpin", "nut", "washer"]]
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Submission saved as {OUTPUT_CSV}")

# =====================
if __name__ == "__main__":
    main()
