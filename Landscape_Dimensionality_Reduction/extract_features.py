#BUTTERFLY FEATURE EXTRACTION
import os
import re
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from tqdm import tqdm
import cv2

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CSV_PATH      = os.path.join(BASE_DIR, "Training_set.csv")
IMAGES_FOLDER = os.path.join(BASE_DIR, "train")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

OUTPUT_FEATURES     = os.path.join(OUTPUT_FOLDER, "resnet_features.npy")
OUTPUT_LABELS       = os.path.join(OUTPUT_FOLDER, "resnet_labels.npy")
OUTPUT_CLASSES      = os.path.join(OUTPUT_FOLDER, "class_names.npy")
OUTPUT_IMAGE_IDS    = os.path.join(OUTPUT_FOLDER, "image_ids.npy")
OUTPUT_IMAGE_COLORS = os.path.join(OUTPUT_FOLDER, "image_colors.npy")

# ------------------------------------------------------------
# IMAGE TRANSFORMS
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------------------
# LOAD RESNET50
# ------------------------------------------------------------
print("Loading pretrained ResNet50 model...")

resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(DEVICE)
resnet.eval()

# ------------------------------------------------------------
# COLOR EXTRACTION
# ------------------------------------------------------------
def compute_image_lab(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Warning: cannot read", image_path)
        return np.array([50, 128, 128])

    h, w, _ = img.shape
    crop = img[h//4:3*h//4, w//4:3*w//4]
    crop = cv2.resize(crop, (64, 64))
    lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0].mean()
    A = lab[:, :, 1].mean()
    B = lab[:, :, 2].mean()
    return np.array([L, A, B])

# ------------------------------------------------------------
# FEATURE + COLOR EXTRACTION
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print(f"Found {len(df)} images")


class_names    = sorted(df["label"].unique().tolist())
class_to_index = {c: i for i, c in enumerate(class_names)}

features     = []
labels       = []
image_ids    = []
image_colors = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
    filename   = row["filename"]
    image_path = os.path.join(IMAGES_FOLDER, filename)

    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        continue

    try:
        image        = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = resnet(input_tensor)
            output = output.view(1, -1)
            output = output / output.norm(dim=1, keepdim=True)
            output = output.squeeze().cpu().numpy()

        color_lab = compute_image_lab(image_path)

        features.append(output)
        labels.append(class_to_index[row["label"]])
        image_ids.append(filename)
        image_colors.append(color_lab)

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

features     = np.stack(features)
labels       = np.array(labels)
image_colors = np.array(image_colors)

print("\nFeature shape :", features.shape)
print("Image IDs     :", len(image_ids))
print("Image colors  :", image_colors.shape)

assert features.shape[0] == len(image_ids),       "Mismatch: features vs image_ids"
assert features.shape[0] == image_colors.shape[0], "Mismatch: features vs image_colors"
assert features.shape[0] == len(labels),          "Mismatch: features vs labels"

# ------------------------------------------------------------
# IMAGE ID VALIDATION
# ------------------------------------------------------------
print("\n" + "="*55)
print("IMAGE ID VALIDATION")
print("="*55)

validation_passed = True

null_ids = [x for x in image_ids if x is None or str(x).strip() == ""]
print(f"[1] Null/empty image IDs  : {len(null_ids)}")
if null_ids:
    print(f"    Examples: {null_ids[:5]}")
    validation_passed = False

unique_ids = set(image_ids)
dup_count  = len(image_ids) - len(unique_ids)
print(f"[2] Duplicate image IDs   : {dup_count}")
if dup_count > 0:
    seen = {}
    for img_id in image_ids:
        seen[img_id] = seen.get(img_id, 0) + 1
    dups = {k: v for k, v in seen.items() if v > 1}
    for img_id, cnt in list(dups.items())[:5]:
        print(f"    '{img_id}' appears {cnt} times")
    validation_passed = False

folder_files = set()
if os.path.isdir(IMAGES_FOLDER):
    for f in os.listdir(IMAGES_FOLDER):
        ext = os.path.splitext(f)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            folder_files.add(f)

missing_on_disk = [x for x in image_ids if str(x) not in folder_files]
print(f"[3] IDs missing on disk   : {len(missing_on_disk)}")
if missing_on_disk:
    print(f"    Examples: {missing_on_disk[:5]}")
    validation_passed = False

unassigned_files = folder_files - unique_ids
print(f"[4] Files on disk with no ID assigned : {len(unassigned_files)}")
if unassigned_files:
    print(f"    Examples: {sorted(unassigned_files)[:5]}")
    print(f"    (warning only - likely skipped due to read errors)")

bad_format = [
    x for x in image_ids
    if not re.match(r'.+\.(jpg|jpeg|png|bmp|tif|tiff|webp)$', str(x), re.IGNORECASE)
]
print(f"[5] IDs with bad format   : {len(bad_format)}")
if bad_format:
    print(f"    Examples: {bad_format[:5]}")
    validation_passed = False

print(f"[6] Input CSV rows        : {len(df)}")
print(f"    image_ids collected   : {len(image_ids)}")
print(f"    Count match           : {len(image_ids) == len(df)}")
if len(image_ids) != len(df):
    validation_passed = False

print("="*55)
if validation_passed:
    print("All checks passed - safe to save")
else:
    print("Validation FAILED - fix errors above before saving")
print("="*55)

if not validation_passed:
    raise RuntimeError(
        "Image ID validation failed. "
        "Fix the errors above before running dimensional reduction."
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
np.save(OUTPUT_FEATURES,     features)
np.save(OUTPUT_LABELS,       labels)
np.save(OUTPUT_CLASSES,      np.array(class_names))
np.save(OUTPUT_IMAGE_IDS,    np.array(image_ids))
np.save(OUTPUT_IMAGE_COLORS, image_colors)

print("\nSaved:")
print("  resnet_features.npy")
print("  resnet_labels.npy")
print("  class_names.npy")
print("  image_ids.npy")
print("  image_colors.npy")