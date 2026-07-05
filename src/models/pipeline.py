from PIL import Image
import matplotlib.pyplot as plt
import torch

class Pipeline:
    def __init__(self, detector, recognizer):
        self.detector = detector
        self.recognizer = recognizer

    def process(self, img):
        # Step 1: Detect text regions in the image
        bboxes, classes, names, confs = self.detector.detect(img)

        predictions = []

        # Step 2: Recognize text in each detected region
        for bbox, cls, conf in zip(bboxes, classes, confs):
            x1, y1, x2, y2 = bbox
            confidence = conf
            name = names[int(cls)]

            # Extract the detected object and crop it
            cropped_image = img.crop((x1, y1, x2, y2))

            transcribed_text = self.recognizer.recognize(
                cropped_image
            )

            predictions.append(
                {
                    "bbox": bbox,
                    "class_name": name,
                    "confidence": confidence,
                    "text": transcribed_text
                }
            )

        return predictions
