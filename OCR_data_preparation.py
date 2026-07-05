from utils.bbs_splitting import split_bounding_boxes
from utils.xml_extractor import extract_data_from_xml

import os

from sklearn.model_selection import train_test_split
import json

from utils.data_splitting_save import save_split

SAVE_DIR = "Datasets/ocr_dataset"
DATASET_DIR = "Datasets/SceneTrialTrain"
SEED = 0
VAL_SIZE = 0.1
TEST_SIZE = 0.1
IS_SHUFFLE = True

img_paths, img_sizes, img_labels, bboxes = extract_data_from_xml(os.path.join(DATASET_DIR, "words.xml"))
img_paths = [os.path.join(DATASET_DIR, path) for path in img_paths]

split_bounding_boxes(img_paths, img_labels, bboxes, SAVE_DIR)

img_ocr_paths = []
ocr_labels = []

with open(os.path.join(SAVE_DIR, "labels.txt"), "r") as f:
    for label in f:
        ocr_labels.append(label.strip().split("\t")[1])
        img_ocr_paths.append(label.strip().split("\t")[0])

# ============ PREPARE VOCABULARY ==============
letters = [char.split(".")[0].lower() for char in ocr_labels]
letters = "".join(letters)
letters = sorted(list(set(letters)))

chars = "".join(letters)

blank_char = "-"
chars += blank_char
vocab_size = len(chars)

char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

X_train, X_val, y_train, y_val = train_test_split(
    img_ocr_paths,
    ocr_labels,
    test_size=VAL_SIZE,
    random_state=SEED,
    shuffle=IS_SHUFFLE
)

X_train, X_test, y_train, y_test = train_test_split(
    X_train,
    y_train,
    test_size=TEST_SIZE,
    random_state=SEED,
    shuffle=IS_SHUFFLE
)

vocab = {
    "chars": chars,
    "char_to_idx": char_to_idx,
    "idx_to_char": idx_to_char,
    "blank_char": blank_char
}

with open(os.path.join(SAVE_DIR, "vocab.json"), "w") as f:
    json.dump(vocab, f, indent=4)

save_split(
    os.path.join(SAVE_DIR, "train_labels.txt"),
    X_train,
    y_train
)

save_split(
    os.path.join(SAVE_DIR, "val_labels.txt"),
    X_val,
    y_val
)

save_split(
    os.path.join(SAVE_DIR, "test_labels.txt"),
    X_test,
    y_test
)