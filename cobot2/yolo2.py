import os

from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO


PACKAGE_NAME = "cobot2"
YOLO_MODEL_FILENAME = "all_best_v0.0.1.pt"


class YoloModel:
    def __init__(self, model_filename=YOLO_MODEL_FILENAME):
        package_path = get_package_share_directory(PACKAGE_NAME)
        model_path = os.path.join(package_path, "resource", model_filename)
        self.model = YOLO(model_path)

    def get_detections(self, image, confidence_threshold=0.5, iou_threshold=0.5):
        if image is None:
            return []

        result = self.model(
            image,
            conf=confidence_threshold,
            iou=iou_threshold,
            agnostic_nms=True,
            verbose=False,
        )[0]

        detections = []
        for box, score, class_id in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            class_id = int(class_id)
            detections.append(
                {
                    "box": [float(value) for value in box],
                    "class": result.names.get(class_id, str(class_id)),
                    "class_id": class_id,
                    "score": float(score),
                }
            )

        return detections
