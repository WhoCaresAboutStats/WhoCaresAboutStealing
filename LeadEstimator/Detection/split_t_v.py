import os
import random
import shutil

images_dir = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\V2\images\train"
labels_dir = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\V2\labels\train"

train_img_dir = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\V2\images\train"
val_img_dir = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\V2\images\val"
unlabeled_img_dir = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\Deprecated\Unlabeled\images"
train_lbl_dir = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\V2\labels\train"
val_lbl_dir = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\V2\labels\val"
unlabeled_lbl_dir = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\Deprecated\Unlabeled\lbl"

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)
os.makedirs(unlabeled_img_dir, exist_ok=True)
os.makedirs(unlabeled_lbl_dir, exist_ok=True)

images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# Separate images with and without labels
labeled_images = []
unlabeled_images = []

for img in images:
    label_name = img.rsplit(".", 1)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_name)

    if os.path.exists(label_path):
        labeled_images.append(img)
    else:
        unlabeled_images.append(img)

# Move unlabeled images
for img in unlabeled_images:
    img_src = os.path.join(images_dir, img)
    shutil.move(img_src, os.path.join(unlabeled_img_dir, img))

print(f"Moved {len(unlabeled_images)} unlabeled images.")

# Shuffle labeled images
random.shuffle(labeled_images)

# 80/20 split
split_idx = int(len(labeled_images) * 0.8)
train_images = labeled_images[:split_idx]
val_images = labeled_images[split_idx:]

def move_files(img_list, img_dest, lbl_dest):
    for img in img_list:
        img_src = os.path.join(images_dir, img)
        lbl_src = os.path.join(labels_dir, img.rsplit(".", 1)[0] + ".txt")

        shutil.move(img_src, os.path.join(img_dest, img))
        shutil.move(lbl_src, os.path.join(lbl_dest, os.path.basename(lbl_src)))

# Move validation images/labels
move_files(val_images, val_img_dir, val_lbl_dir)

print("Dataset split complete!")