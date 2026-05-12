import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from cobot2.yolo2 import YoloModel


COLOR_IMAGE_TOPIC = "/camera/camera/color/image_raw"
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50
EDGE_MARGIN = 2
NORMAL_BOX_COLOR = (0, 255, 0)
CLIPPED_BOX_COLOR = (0, 0, 255)


def is_box_on_image_edge(box, width, height, edge_margin=EDGE_MARGIN):
    x1, y1, x2, y2 = box
    return (
        x1 <= edge_margin
        or y1 <= edge_margin
        or x2 >= width - 1 - edge_margin
        or y2 >= height - 1 - edge_margin
    )


def draw_detection(image, detection, color):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, detection["box"])
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    label = f"{detection['class']} {detection['score'] * 100:.1f}%"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


class YoloPredictCameraNode(Node):
    def __init__(self):
        super().__init__("yolo_predict_camera_node")
        self.bridge = CvBridge()
        self.model = YoloModel()

        cv2.namedWindow("YOLO Predict", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Predict", 960, 720)

        self.create_subscription(
            Image,
            COLOR_IMAGE_TOPIC,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"실시간 예측 시작: {COLOR_IMAGE_TOPIC} 구독 중 (종료: Q 키)"
        )

    def get_split_detections(self, frame):
        height, width = frame.shape[:2]
        detections = self.model.get_detections(
            frame,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
        )

        edge_detections = []
        inner_detections = []
        for detection in detections:
            if is_box_on_image_edge(detection["box"], width, height):
                edge_detections.append(detection)
            else:
                inner_detections.append(detection)

        return edge_detections, inner_detections

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"이미지 변환 실패: {exc}")
            return

        edge_detections, inner_detections = self.get_split_detections(frame)
        for detection in inner_detections:
            draw_detection(frame, detection, NORMAL_BOX_COLOR)
        for detection in edge_detections:
            draw_detection(frame, detection, CLIPPED_BOX_COLOR)

        cv2.imshow("YOLO Predict", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = YoloPredictCameraNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        print("예측 종료")


if __name__ == "__main__":
    main()
