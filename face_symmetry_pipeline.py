import os
import random
import time
import bz2
import urllib.request
import cv2
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, levene
from torchvision.datasets import CelebA
from torchvision import transforms
from tqdm import tqdm

from thispersondoesnotexist import get_online_person, save_picture

NUM_IMAGES = 100
OUTPUT_DIR = "data"
AI_DIR = os.path.join(OUTPUT_DIR, "ai_faces")
HUMAN_DIR = os.path.join(OUTPUT_DIR, "human_faces")  # fallback if CelebA sampling needed
CUSTOM_HUMAN_DIR = "/Users/abhinavgunturi/engr_project/data/celeba"
MANIFEST_CSV = os.path.join(OUTPUT_DIR, "image_manifest.csv")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "image_manifest_with_symmetry.csv")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(AI_DIR, exist_ok=True)
os.makedirs(HUMAN_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CUSTOM_HUMAN_DIR, exist_ok=True)

MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(MODEL_PATH):
    print("Dlib model not found. Downloading...")
    compressed_path = MODEL_PATH + ".bz2"
    urllib.request.urlretrieve(MODEL_URL, compressed_path)
    print("Download complete. Extracting...")
    with bz2.BZ2File(compressed_path, 'rb') as f_in:
        with open(MODEL_PATH, 'wb') as f_out:
            f_out.write(f_in.read())
    os.remove(compressed_path)
    print("Model extracted successfully.")
else:
    print("Dlib shape predictor model found.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

def detect_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    return [(shape.part(i).x, shape.part(i).y) for i in range(68)]

def compute_symmetry(landmarks, width):
    if landmarks is None:
        return np.nan
    xs = np.array([x for x, y in landmarks])
    mirrored = width - xs
    dev = np.abs(xs - mirrored) / width
    return 1 - np.mean(dev)

existing_ai = sorted([f for f in os.listdir(AI_DIR) if f.endswith(".jpg")])
num_existing_ai = len(existing_ai)
records = []

print(f"Existing AI images: {num_existing_ai}/{NUM_IMAGES}")
if num_existing_ai < NUM_IMAGES:
    print("Downloading missing AI-generated images...")
    for i in tqdm(range(num_existing_ai, NUM_IMAGES), desc="AI faces"):
        try:
            img_bytes = get_online_person()
            fname = f"ai_{i+1:03d}.jpg"
            path = os.path.join(AI_DIR, fname)
            save_picture(img_bytes, path)
            records.append({"id": f"ai_{i+1:03d}", "path": path, "source": "ai"})
            time.sleep(1)
        except Exception as e:
            print(f"Failed {i}: {e}")
else:
    print("All AI images already exist.")

for i, fname in enumerate(existing_ai, 1):
    path = os.path.join(AI_DIR, fname)
    records.append({"id": f"ai_{i:03d}", "path": path, "source": "ai"})

existing_human = sorted([f for f in os.listdir(CUSTOM_HUMAN_DIR) if f.endswith(".jpg")])
num_existing_human = len(existing_human)
print(f"Existing human images: {num_existing_human}/{NUM_IMAGES}")

if num_existing_human >= NUM_IMAGES:
    print("Using existing human images from custom CelebA folder.")
    for i, fname in enumerate(existing_human[:NUM_IMAGES], 1):
        path = os.path.join(CUSTOM_HUMAN_DIR, fname)
        records.append({"id": f"human_{i:03d}", "path": path, "source": "human"})
else:
    remaining = NUM_IMAGES - num_existing_human
    print(f"Sampling {remaining} missing human faces from CelebA dataset...")
    celeba = CelebA(root=OUTPUT_DIR, split="test", download=True)
    indices = random.sample(range(len(celeba)), remaining)
    for i, idx in enumerate(tqdm(indices, desc="CelebA faces"), start=num_existing_human):
        img, _ = celeba[idx]
        img_pil = transforms.ToPILImage()(img)
        fname = f"human_{i+1:03d}.jpg"
        path = os.path.join(HUMAN_DIR, fname)
        img_pil.save(path)
        records.append({"id": f"human_{i+1:03d}", "path": path, "source": "human"})

df = pd.DataFrame(records)
df.to_csv(MANIFEST_CSV, index=False)
print("Saved manifest to:", MANIFEST_CSV)

if os.path.exists(RESULTS_CSV):
    df_results = pd.read_csv(RESULTS_CSV)
else:
    df_results = pd.DataFrame(columns=df.columns.tolist() + ["symmetry_score"])

sym_scores = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Compute symmetry"):
    if row["id"] in df_results["id"].values and "symmetry_score" in df_results.columns and not pd.isna(df_results.loc[df_results["id"] == row["id"], "symmetry_score"]).all():
        sym_scores.append(df_results.loc[df_results["id"] == row["id"], "symmetry_score"].values[0])
        continue

    lm = detect_landmarks(row["path"])
    img = cv2.imread(row["path"])
    if img is None:
        sym = np.nan
    else:
        h, w = img.shape[:2]
        sym = compute_symmetry(lm, w)
    sym_scores.append(sym)

df["symmetry_score"] = sym_scores
df.to_csv(RESULTS_CSV, index=False)
print("Saved results with symmetry to:", RESULTS_CSV)

df_clean = df.dropna(subset=["symmetry_score"])
ai = df_clean[df_clean["source"] == "ai"]["symmetry_score"]
human = df_clean[df_clean["source"] == "human"]["symmetry_score"]

summary = pd.DataFrame({
    "mean": [ai.mean(), human.mean()],
    "std": [ai.std(), human.std()],
    "count": [len(ai), len(human)]
}, index=["AI (TPDNE)", "Human (CelebA)"])

print("\nSummary statistics:")
print(summary)

t_stat, p_val = ttest_ind(ai, human, equal_var=False)
lev_stat, lev_p = levene(ai, human)

print(f"\nT-test (difference of means): t = {t_stat:.4f}, p = {p_val:.5f}")
print(f"Levene’s test (difference in variance): stat = {lev_stat:.4f}, p = {lev_p:.5f}")

plt.figure(figsize=(8,5))
sns.kdeplot(ai, label="AI (TPDNE)", fill=True, bw_adjust=0.5)
sns.kdeplot(human, label="Human (CelebA)", fill=True, bw_adjust=0.5)
plt.xlabel("Symmetry Score")
plt.ylabel("Density")
plt.title("Kernel Density of Facial Symmetry")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "symmetry_kde.png"))
plt.close()

plt.figure(figsize=(6,4))
plt.bar(summary.index, summary["mean"], yerr=summary["std"], capsize=5)
plt.ylabel("Symmetry Score")
plt.title("Mean ± Std of Symmetry by Source")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "symmetry_bar.png"))
plt.close()

print("Plots saved to", FIGURES_DIR)
