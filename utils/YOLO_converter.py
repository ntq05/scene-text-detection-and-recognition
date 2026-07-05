def convert_to_yolo_format(image_paths, image_sizes, bounding_boxes):
    yolo_data = []

    for image_path, image_size, bboxes, in zip(image_paths, image_sizes, bounding_boxes):
        image_width, image_height = image_size

        yolo_labels = []

        for bbox in bboxes:
            x, y, w, h = bbox

            center_x = (x + w / 2) / image_width
            center_y = (y + h / 2) / image_height

            normalized_width = w / image_width
            normalized_height = h / image_height

            class_id = 0

            yolo_label = f"{class_id} {center_x:.6f} {center_y:.6f} {normalized_width:.6f} {normalized_height:.6f}"
            yolo_labels.append(yolo_label)

        yolo_data.append((image_path, yolo_labels))

    return yolo_data