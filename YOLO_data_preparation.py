from utils.data_saving import save_data
from utils.xml_extractor import extract_data_from_xml
from utils.YOLO_converter import convert_to_yolo_format

import os
from sklearn.model_selection import train_test_split
import yaml

SEED = 0
SAVE_YOLO_DATA_DIR = "Datasets/yolo_data"
DATASET_DIR = "Datasets/SceneTrialTrain"

words_xml_path = os.path.join(DATASET_DIR, "words.xml")
image_paths, image_sizes, image_labels, bounding_boxes = extract_data_from_xml(
    words_xml_path
)

class_labels = ["text"]

yolo_data = convert_to_yolo_format(image_paths, image_sizes, bounding_boxes)

val_size = 0.2
test_size = 0.125
is_shuffle = True
train_data, test_data = train_test_split(
    yolo_data,
    test_size=val_size,
    random_state=SEED,
    shuffle=is_shuffle,
)
test_data, val_data = train_test_split(
    test_data,
    test_size=test_size,
    random_state=SEED,
    shuffle=is_shuffle,
)

os.makedirs(SAVE_YOLO_DATA_DIR, exist_ok=True)
save_train_dir = os.path.join(SAVE_YOLO_DATA_DIR, "train")
save_val_dir = os.path.join(SAVE_YOLO_DATA_DIR, "val")
save_test_dir = os.path.join(SAVE_YOLO_DATA_DIR, "test")

save_data(train_data, DATASET_DIR, save_train_dir)
save_data(test_data, DATASET_DIR, save_val_dir)
save_data(val_data, DATASET_DIR, save_test_dir)

data_yaml = {
    "path": "./Datasets/yolo_data",
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": 1,
    "names": class_labels,
}

yolo_yaml_path = os.path.join(SAVE_YOLO_DATA_DIR, "data.yaml")

with open(yolo_yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)