import os

from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO


PACKAGE_NAME = "cobot2"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

YOLO_MODEL_FILENAME = "box_can_pla_best.pt"
YOLO_MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", YOLO_MODEL_FILENAME)


class YoloModel:
    def __init__(self):
        # YOLO_MODEL_FILENAME = "box_can_pla_best.pt"
        # YOLO_MODEL_FILENAME = "box_can_cup_0430.pt"
        YOLO_MODEL_FILENAME = "all_best_v0.0.1.pt"
        YOLO_MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", YOLO_MODEL_FILENAME)
        self.model = YOLO(YOLO_MODEL_PATH)

    def get_detections(self, img, confidence_threshold=0.5, iou_threshold=0.5):
        if img is None:
            return []
        result = self.model(
            img,
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


class YoloeModel:
    def __init__(self):
        self.model = YOLO("yoloe-26n-seg.pt")
        self.model.set_classes(["box", "can", "plastic_cup"])


    def get_detections(self, img, confidence_threshold=0.5, iou_threshold=0.5):
        if img is None:
            return []
        result = self.model(
            img,
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
