import os
import shutil

def save_data(data, src_img_dir, save_dir):
    
    os.makedirs(save_dir, exist_ok=True)

    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    for image_path, yolo_labels in data:

        shutil.copy(
            os.path.join(src_img_dir, image_path),
            os.path.join(save_dir, "images")
        )

        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0]

        with open(os.path.join(save_dir, "labels", f"{image_name}.txt"), "w") as f:
            for label in yolo_labels:
                f.write(f"{label}\n")