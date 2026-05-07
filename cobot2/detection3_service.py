import threading

import cv2
import rclpy
from cv_bridge import CvBridge
from od_msg.srv import SrvDetections, SrvImage
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from cobot2.yolo2 import YoloModel


DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_EDGE_MARGIN = 2
SERVICE_TIMEOUT_SEC = 10.0
NORMAL_BOX_COLOR = (0, 255, 0)
CLIPPED_BOX_COLOR = (0, 0, 255)
DEBUG_IMAGE_PUBLISH_PERIOD_SEC = 0.2


class ObjectDetectionServiceNode(Node):
    def __init__(self):
        super().__init__("object_detection_service_node")
        self.bridge = CvBridge()
        self.model = YoloModel()
        self.callback_group = ReentrantCallbackGroup()
        self.update_images_client = self.create_client(
            Trigger,
            "update_images",
            callback_group=self.callback_group,
        )
        self.color_image_client = self.create_client(
            SrvImage,
            "get_color_image",
            callback_group=self.callback_group,
        )
        detection_image_qos = QoSProfile(depth=1)
        detection_image_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.detection_image_pub = self.create_publisher(
            Image,
            "detection_result_image",
            detection_image_qos,
        )
        self.latest_detection_image_msg = None
        self.debug_image_timer = self.create_timer(
            DEBUG_IMAGE_PUBLISH_PERIOD_SEC,
            self._publish_latest_detection_image,
            callback_group=self.callback_group,
        )
        self.create_service(
            SrvDetections,
            "detect_objects",
            self.handle_detect_objects,
            callback_group=self.callback_group,
        )
        self.create_service(
            SrvDetections,
            "detect_all_objects",
            self.handle_detect_all_objects,
            callback_group=self.callback_group,
        )
        self.create_service(
            SrvDetections,
            "detect_center_object",
            self.handle_detect_center_object,
            callback_group=self.callback_group,
        )
        self.get_logger().info("ObjectDetectionServiceNode initialized.")

    def _wait_for_future(self, future, timeout_sec):
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())
        return done_event.wait(timeout=timeout_sec)

    def handle_detect_objects(self, request, response):
        self.get_logger().info("detect_objects request received.")
        frame = self._get_updated_color_frame()
        if frame is None:
            self.get_logger().warn("Cannot detect objects: color frame is empty.")
            response.boxes = []
            response.class_ids = []
            response.scores = []
            return response

        edge_detections, inner_detections = self.get_split_detections(
            frame,
            confidence_threshold=request.confidence_threshold
            or DEFAULT_CONFIDENCE_THRESHOLD,
            iou_threshold=request.iou_threshold or DEFAULT_IOU_THRESHOLD,
            edge_margin=DEFAULT_EDGE_MARGIN,
        )
        self._log_detection_counts(inner_detections, edge_detections)
        self._publish_detection_image(frame, edge_detections, inner_detections)

        response.boxes = self._flatten_boxes(inner_detections)
        response.class_ids = [detection["class_id"] for detection in inner_detections]
        response.scores = [detection["score"] for detection in inner_detections]
        self.get_logger().info(
            f"Responding with {len(inner_detections)} inner objects."
        )
        return response

    def handle_detect_all_objects(self, request, response):
        self.get_logger().info("detect_all_objects request received.")
        frame = self._get_updated_color_frame()
        if frame is None:
            self.get_logger().warn("Cannot detect all objects: color frame is empty.")
            response.boxes = []
            response.class_ids = []
            response.scores = []
            return response

        detections = self.model.get_detections(
            frame,
            confidence_threshold=request.confidence_threshold
            or DEFAULT_CONFIDENCE_THRESHOLD,
            iou_threshold=request.iou_threshold or DEFAULT_IOU_THRESHOLD,
        )
        self.get_logger().info(
            f"all objects by class_id: "
            f"{self._format_counts(self._count_by_class_id(detections))}"
        )
        self._publish_detection_image(frame, [], detections)

        response.boxes = self._flatten_boxes(detections)
        response.class_ids = [detection["class_id"] for detection in detections]
        response.scores = [detection["score"] for detection in detections]
        self.get_logger().info(
            f"Responding with {len(detections)} total objects."
        )
        return response

    def handle_detect_center_object(self, request, response):
        self.get_logger().info("detect_center_object request received.")
        frame = self._get_updated_color_frame()
        if frame is None:
            self.get_logger().warn("Cannot detect center object: color frame is empty.")
            response.boxes = []
            response.class_ids = []
            response.scores = []
            return response

        detections = self.model.get_detections(
            frame,
            confidence_threshold=request.confidence_threshold
            or DEFAULT_CONFIDENCE_THRESHOLD,
            iou_threshold=request.iou_threshold or DEFAULT_IOU_THRESHOLD,
        )
        self.get_logger().info(
            f"center detection candidates by class_id: "
            f"{self._format_counts(self._count_by_class_id(detections))}"
        )

        center_detection = self._select_center_detection(frame, detections)
        center_detections = [center_detection] if center_detection is not None else []
        self._publish_detection_image(frame, [], center_detections)

        response.boxes = self._flatten_boxes(center_detections)
        response.class_ids = [
            detection["class_id"] for detection in center_detections
        ]
        response.scores = [detection["score"] for detection in center_detections]
        self.get_logger().info(
            f"Responding with {len(center_detections)} center object."
        )
        return response

    def get_split_detections(
        self,
        frame,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold=DEFAULT_IOU_THRESHOLD,
        edge_margin=DEFAULT_EDGE_MARGIN,
    ):
        height, width = frame.shape[:2]
        detections = self.model.get_detections(
            frame,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )

        edge_detections = []
        inner_detections = []
        for detection in detections:
            if self._is_box_on_image_edge(detection["box"], width, height, edge_margin):
                edge_detections.append(detection)
            else:
                inner_detections.append(detection)

        return edge_detections, inner_detections

    def _get_updated_color_frame(self):
        self.get_logger().info("Updating RealSense image cache...")
        if not self._call_update_images():
            return None

        self.get_logger().info("Requesting updated color image...")
        color_msg = self._request_color_image()
        if color_msg is None:
            return None
        stamp = color_msg.header.stamp
        self.get_logger().info(
            f"Using color image stamp={stamp.sec}.{stamp.nanosec:09d}"
        )
        return self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")

    def _call_update_images(self):
        while not self.update_images_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for update_images service...")

        future = self.update_images_client.call_async(Trigger.Request())
        if not self._wait_for_future(future, SERVICE_TIMEOUT_SEC):
            future.cancel()
            self.get_logger().error(
                f"Timed out waiting for update_images after {SERVICE_TIMEOUT_SEC:.1f}s."
            )
            return False
        if future.result() is None:
            self.get_logger().error("Failed to call update_images service.")
            return False

        response = future.result()
        if not response.success:
            self.get_logger().warn(response.message)
            return False

        return True

    def _request_color_image(self):
        while not self.color_image_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for get_color_image service...")

        future = self.color_image_client.call_async(SrvImage.Request())
        if not self._wait_for_future(future, SERVICE_TIMEOUT_SEC):
            future.cancel()
            self.get_logger().error(
                f"Timed out waiting for get_color_image after {SERVICE_TIMEOUT_SEC:.1f}s."
            )
            return None
        if future.result() is None:
            self.get_logger().error("Failed to call get_color_image service.")
            return None

        response = future.result()
        if not response.success:
            self.get_logger().warn(response.message)
            return None

        return response.image

    def _is_box_on_image_edge(self, box, width, height, edge_margin):
        x1, y1, x2, y2 = box
        return (
            x1 <= edge_margin
            or y1 <= edge_margin
            or x2 >= width - 1 - edge_margin
            or y2 >= height - 1 - edge_margin
        )

    def _select_center_detection(self, frame, detections):
        if not detections:
            return None

        height, width = frame.shape[:2]
        image_center_x = width / 2.0
        image_center_y = height / 2.0
        return min(
            detections,
            key=lambda detection: self._get_squared_center_distance(
                detection["box"],
                image_center_x,
                image_center_y,
            ),
        )

    def _get_squared_center_distance(self, box, image_center_x, image_center_y):
        x1, y1, x2, y2 = box
        box_center_x = (x1 + x2) / 2.0
        box_center_y = (y1 + y2) / 2.0
        dx = box_center_x - image_center_x
        dy = box_center_y - image_center_y
        return dx * dx + dy * dy

    def _publish_detection_image(self, frame, edge_detections, inner_detections):
        annotated = frame.copy()
        for detection in inner_detections:
            self._draw_detection(annotated, detection, color=NORMAL_BOX_COLOR)
        for detection in edge_detections:
            self._draw_detection(annotated, detection, color=CLIPPED_BOX_COLOR)

        msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "detection_result_image"
        self.latest_detection_image_msg = msg
        self.detection_image_pub.publish(msg)

    def _publish_latest_detection_image(self):
        if self.latest_detection_image_msg is not None:
            self.latest_detection_image_msg.header.stamp = self.get_clock().now().to_msg()
            self.detection_image_pub.publish(self.latest_detection_image_msg)

    def _draw_detection(self, image, detection, color):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(int, detection["box"])
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{detection['class']}: {detection['score']:.2f}"
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

    def _flatten_boxes(self, detections):
        boxes = []
        for detection in detections:
            boxes.extend(float(value) for value in detection["box"])
        return boxes

    def _log_detection_counts(self, inner_detections, edge_detections):
        normal_counts = self._count_by_class_id(inner_detections)
        clipped_counts = self._count_by_class_id(edge_detections)
        self.get_logger().info(
            f"normal objects by class_id: {self._format_counts(normal_counts)}"
        )
        self.get_logger().info(
            f"clipped objects by class_id: {self._format_counts(clipped_counts)}"
        )

    def _count_by_class_id(self, detections):
        counts = {}
        for detection in detections:
            class_id = detection["class_id"]
            counts[class_id] = counts.get(class_id, 0) + 1
        return counts

    def _format_counts(self, counts):
        if not counts:
            return "none"
        return ", ".join(
            f"{class_id}: {count}" for class_id, count in sorted(counts.items())
        )

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionServiceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
