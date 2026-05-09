from cobot2.yolo2 import YoloModel
import cv2
from cv_bridge import CvBridge
from dsr_msgs2.srv import GetCurrentPosx
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from od_msg.srv import SrvBasePositions
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from ament_index_python.packages import get_package_share_directory
import os
import threading
from cobot2.realsense3 import RealsenseFrameNode
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_EDGE_MARGIN = 2 # pixels
SERVICE_TIMEOUT_SEC = 15.0
FRAME_TIMEOUT_SEC = 3.0
GET_CURRENT_POSX_SERVICE = "/dsr01/aux_control/get_current_posx"
NORMAL_BOX_COLOR = (0, 255, 0)
CLIPPED_BOX_COLOR = (0, 0, 255)
NEAREST_DEPTH_SEARCH_RADIUS = 30
PCA_BACKGROUND_HUE_MIN = 20
PCA_BACKGROUND_HUE_MAX = 40
PCA_BACKGROUND_SATURATION_MIN = 0
PCA_BACKGROUND_SATURATION_MAX = 255
PCA_BACKGROUND_VALUE_MIN = 0
PCA_BACKGROUND_VALUE_MAX = 255
PCA_MASK_MORPH_KERNEL_SIZE = 5
PCA_MIN_POINTS = 10
PCA_LINE_SAMPLE_STEP_PX = 5
PCA_MIN_3D_POINTS = 2
DEBUG_IMAGE_DIR = "/tmp/cobot2_detect_cal_pos_debug"
DEBUG_MAX_DEPTH_LABELS = 12

class DetectCalPosService(Node):
    def __init__(self):
        super().__init__("detect_cal_pos_service_node")
        self.frame_node = RealsenseFrameNode()
        self.bridge = CvBridge()
        self.model = YoloModel()
        self.callback_group = ReentrantCallbackGroup()
        self.current_posx_client = self.create_client(
            GetCurrentPosx,
            GET_CURRENT_POSX_SERVICE,
            callback_group=self.callback_group,
        )
        self.detection_image_pub = self.create_publisher(
            Image,
            "detection_result_image",
            10,
        )
        self.latest_detection_image_msg = None
        self.debug_image_timer = self.create_timer(
            0.2,
            self._publish_latest_detection_image,
            callback_group=self.callback_group,
        )
        self.create_service(
            SrvBasePositions,
            'get_base_positions', #,"inner_objects_points",
            self.handle_inner_objects_points,
            callback_group=self.callback_group,
        )
        self.create_service(
            SrvBasePositions,
            "center_of_center_points",
            self.handle_center_of_center_points,
            callback_group=self.callback_group,
        )
        self.create_service(
            SrvBasePositions,
            'get_center_base_positions',#"center_object_points",
            self.handle_center_object_points,
            callback_group=self.callback_group,
        )
        self.gripper2cam_path = os.path.join(
            get_package_share_directory("cobot2"),
            "resource",
            "T_gripper2camera.npy",
        )
        self.get_logger().info("DetectCalPosServiceNode initialized.")

    def _wait_for_future(self, future, timeout_sec):
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())
        return done_event.wait(timeout=timeout_sec)

    def get_frames_once(self, timeout_sec=FRAME_TIMEOUT_SEC):
        self.frame_node.reset_frames()
        deadline = self.get_clock().now().nanoseconds + int(timeout_sec * 1e9)

        while rclpy.ok() and not self.frame_node.has_frames():
            rclpy.spin_once(self.frame_node, timeout_sec=0.1)
            if self.get_clock().now().nanoseconds >= deadline:
                self.get_logger().warn(
                    "Timed out waiting for color image, depth image, and camera info."
                )
                return None, None, None

        return self.frame_node.get_frames()

    def handle_inner_objects_points(self, request, response):
        self.get_logger().info("inner_objects_points request received.")
        frame, depth_image, camera_info = self.get_frames_once()
        if frame is None or depth_image is None or camera_info is None:
            self._set_empty_base_position_response(
                response,
                "Failed to receive color image, depth image, or camera info.",
            )
            return response

        _, detections = self.detect_from_color_image(color_image=frame)

        edge_detections, inner_detections = self.split_inner_detections(
            frame,
            detections,
            edge_margin=DEFAULT_EDGE_MARGIN,
        )
        self._log_detection_counts(inner_detections, edge_detections)
        self._publish_detection_image(frame, edge_detections, inner_detections)

        robot_posx = self.request_robot_posx()
        if robot_posx is None:
            self._set_empty_base_position_response(
                response,
                "Failed to receive current robot posx.",
            )
            return response

        base_positions = []
        for detection in inner_detections:
            cx, cy = self._get_box_center(detection["box"])
            xyz = self.get_xyz_from_pixel(
                cx,
                cy,
                depth_image=depth_image,
                camera_info=camera_info,
                robot_posx=robot_posx,
            )
            if xyz is None:
                continue

            x, y, z = xyz
            base_positions.append(
                {
                    "box": detection["box"],
                    "class_id": detection["class_id"],
                    "x": x,
                    "y": y,
                    "z": z,
                    "rx": 0.0,
                    "ry": 180.0,
                    "rz": 0.0,
                }
            )

        self._set_base_position_response(
            response,
            base_positions,
            f"Calculated {len(base_positions)} inner object positions.",
        )
        return response

    def handle_center_of_center_points(self, request, response):
        self.get_logger().info("center_of_center_points request received.")
        frame, depth_image, camera_info = self.get_frames_once()
        if frame is None or depth_image is None or camera_info is None:
            self._set_empty_base_position_response(
                response,
                "Failed to receive color image, depth image, or camera info.",
            )
            return response

        _, detections = self.detect_from_color_image(color_image=frame)
        self._publish_detection_image(frame, [], detections)
        if not detections:
            self._set_empty_base_position_response(
                response,
                "No objects detected.",
            )
            return response

        center_xs = []
        center_ys = []
        for detection in detections:
            cx, cy = self._get_box_center(detection["box"])
            center_xs.append(cx)
            center_ys.append(cy)

        avg_x = int(round(float(np.mean(center_xs))))
        avg_y = int(round(float(np.mean(center_ys))))

        robot_posx = self.request_robot_posx()
        if robot_posx is None:
            self._set_empty_base_position_response(
                response,
                "Failed to receive current robot posx.",
            )
            return response

        xyz = self.get_xyz_from_pixel(
            avg_x,
            avg_y,
            depth_image=depth_image,
            camera_info=camera_info,
            robot_posx=robot_posx,
        )
        if xyz is None:
            self._set_empty_base_position_response(
                response,
                "Failed to calculate center-of-centers xyz.",
            )
            return response

        x, y, z = xyz
        position = {
            "box": [float(avg_x), float(avg_y), float(avg_x), float(avg_y)],
            "class_id": -1,
            "x": x,
            "y": y,
            "z": z,
            "rx": 0.0,
            "ry": 180.0,
            "rz": 0.0,
        }
        self._set_base_position_response(
            response,
            [position],
            f"Calculated center of {len(detections)} object centers.",
        )
        return response

    def handle_center_object_points(self, request, response):
        self.get_logger().info("center_object_points request received.")
        frame, depth_image, camera_info = self.get_frames_once()
        if frame is None or depth_image is None or camera_info is None:
            self._set_empty_base_position_response(
                response,
                "Failed to receive color image, depth image, or camera info.",
            )
            return response

        _, detections = self.detect_from_color_image(color_image=frame)
        center_detection = self.select_center_detection(frame, detections)
        center_detections = [center_detection] if center_detection is not None else []
        self._publish_detection_image(frame, [], center_detections)
        if center_detection is None:
            self._set_empty_base_position_response(
                response,
                "No center object detected.",
            )
            return response

        cx, cy = self._get_box_center(center_detection["box"])
        robot_posx = self.request_robot_posx()
        if robot_posx is None:
            self._set_empty_base_position_response(
                response,
                "Failed to receive current robot posx.",
            )
            return response

        xyz = self.get_xyz_from_pixel(
            cx,
            cy,
            depth_image=depth_image,
            camera_info=camera_info,
            robot_posx=robot_posx,
        )
        if xyz is None:
            self._set_empty_base_position_response(
                response,
                "Failed to calculate center object xyz.",
            )
            return response

        rx, ry, rz = self.get_rxyz_from_box(
            center_detection["box"],
            color_image=frame,
            depth_image=depth_image,
            camera_info=camera_info,
            base_xyz=xyz,
        )
        x, y, z = xyz
        position = {
            "box": center_detection["box"],
            "class_id": center_detection["class_id"],
            "x": x,
            "y": y,
            "z": z,
            "rx": rx,
            "ry": ry,
            "rz": rz,
        }
        self._set_base_position_response(
            response,
            [position],
            "Calculated center object position.",
        )
        return response

    def _handle_center_detection(self, response):
        frame, detections = self.detect_from_color_image()
        if frame is None:
            self._set_empty_base_position_response(response, "Color image is empty.")
            return response

        center_detection = self.select_center_detection(frame, detections)
        center_detections = [center_detection] if center_detection is not None else []
        self._publish_detection_image(frame, [], center_detections)

        response.boxes = self._flatten_boxes(center_detections)
        response.class_ids = [detection["class_id"] for detection in center_detections]
        response.xs = []
        response.ys = []
        response.zs = []
        response.rxs = []
        response.rys = []
        response.rzs = []
        response.success = bool(center_detections)
        response.message = f"Detected {len(center_detections)} center object."
        return response

    def detect_from_color_image(
        self,
        color_image=None,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold=DEFAULT_IOU_THRESHOLD,
    ):
        if color_image is None:
            color_image, _, _ = self.get_frames_once()
        if color_image is None:
            self.get_logger().warn("Cannot detect objects: color image is empty.")
            return None, []

        detections = self.model.get_detections(
            color_image,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        self.get_logger().info(f"Detected {len(detections)} objects.")
        return color_image, detections

    def _set_base_position_response(self, response, positions, message):
        response.boxes = self._flatten_boxes(positions)
        response.class_ids = [position["class_id"] for position in positions]
        response.xs = [position["x"] for position in positions]
        response.ys = [position["y"] for position in positions]
        response.zs = [position["z"] for position in positions]
        response.rxs = [position["rx"] for position in positions]
        response.rys = [position["ry"] for position in positions]
        response.rzs = [position["rz"] for position in positions]
        response.success = bool(positions)
        response.message = message

    def _set_empty_base_position_response(self, response, message):
        response.boxes = []
        response.class_ids = []
        response.xs = []
        response.ys = []
        response.zs = []
        response.rxs = []
        response.rys = []
        response.rzs = []
        response.success = False
        response.message = message

    def split_inner_detections(
        self,
        frame,
        detections,
        edge_margin=DEFAULT_EDGE_MARGIN,
    ):
        height, width = frame.shape[:2]
        edge_detections = []
        inner_detections = []
        for detection in detections:
            if self._is_box_on_image_edge(detection["box"], width, height, edge_margin):
                edge_detections.append(detection)
            else:
                inner_detections.append(detection)

        return edge_detections, inner_detections

    def select_center_detection(self, frame, detections):
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

    def _is_box_on_image_edge(self, box, width, height, edge_margin):
        x1, y1, x2, y2 = box
        return (
            x1 <= edge_margin
            or y1 <= edge_margin
            or x2 >= width - 1 - edge_margin
            or y2 >= height - 1 - edge_margin
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

    def get_xyz_from_pixel(
        self,
        x,
        y,
        depth_image=None,
        camera_info=None,
        robot_posx=None,
    ):
        if depth_image is None or camera_info is None:
            _, depth_image, camera_info = self.get_frames_once()
        if depth_image is None or camera_info is None:
            return None

        px = int(x)
        py = int(y)
        depth = self._get_depth_or_nearest(depth_image, px, py)
        if depth is None:
            self.get_logger().warn(f"Invalid depth near ({px}, {py}).")
            return None

        camera_coord = self._pixel_to_camera_coords(px, py, depth, camera_info)
        if camera_coord is None:
            return None

        if robot_posx is None:
            robot_posx = self.request_robot_posx()
        if robot_posx is None:
            return None

        base_coord = self.transform_to_base(
            camera_coord,
            self.gripper2cam_path,
            robot_posx,
        )
        xyz = tuple(float(value) for value in base_coord[:3])
        self.get_logger().info(
            "pixel=(%d, %d), depth=%.3f, base_xyz=[%.3f, %.3f, %.3f]"
            % (int(x), int(y), depth, xyz[0], xyz[1], xyz[2])
        )
        return xyz

    def _get_box_center(self, box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _get_depth_or_nearest(self, depth_image, x, y):
        height, width = depth_image.shape[:2]
        if x < 0 or y < 0 or x >= width or y >= height:
            self.get_logger().warn(f"Coordinates ({x}, {y}) out of range.")
            return None

        depth = float(depth_image[y, x])
        if self._is_valid_position_depth(depth):
            return depth

        x1 = max(0, x - NEAREST_DEPTH_SEARCH_RADIUS)
        y1 = max(0, y - NEAREST_DEPTH_SEARCH_RADIUS)
        x2 = min(width, x + NEAREST_DEPTH_SEARCH_RADIUS + 1)
        y2 = min(height, y + NEAREST_DEPTH_SEARCH_RADIUS + 1)

        roi = depth_image[y1:y2, x1:x2]
        valid_mask = np.isfinite(roi) & (roi > 0)
        if not np.any(valid_mask):
            return None

        valid_ys, valid_xs = np.where(valid_mask)
        image_xs = valid_xs + x1
        image_ys = valid_ys + y1
        squared_distances = (image_xs - x) ** 2 + (image_ys - y) ** 2
        nearest_index = int(np.argmin(squared_distances))
        nearest_x = int(image_xs[nearest_index])
        nearest_y = int(image_ys[nearest_index])
        nearest_depth = float(depth_image[nearest_y, nearest_x])

        self.get_logger().warn(
            "Depth at (%d, %d) was invalid: %.3f. "
            "Using nearest valid depth %.3f at (%d, %d)."
            % (x, y, depth, nearest_depth, nearest_x, nearest_y)
        )
        return nearest_depth

    def _is_valid_position_depth(self, depth):
        return depth is not None and np.isfinite(depth) and depth > 0.0

    def _pixel_to_camera_coords(self, x, y, depth, camera_info):
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        ppx = camera_info.k[2]
        ppy = camera_info.k[5]

        if fx == 0.0 or fy == 0.0:
            self.get_logger().warn(
                f"Invalid camera intrinsics: fx={fx}, fy={fy}"
            )
            return None

        return [
            float((x - ppx) * depth / fx),
            float((y - ppy) * depth / fy),
            float(depth),
        ]

    def request_robot_posx(self):
        while not self.current_posx_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for {GET_CURRENT_POSX_SERVICE} service...")

        request = GetCurrentPosx.Request()
        request.ref = 0

        self.get_logger().info(f"Calling {GET_CURRENT_POSX_SERVICE} service...")
        future = self.current_posx_client.call_async(request)

        if not self._wait_for_future(future, SERVICE_TIMEOUT_SEC):
            future.cancel()
            self.get_logger().error(
                f"Timed out waiting for {GET_CURRENT_POSX_SERVICE} "
                f"after {SERVICE_TIMEOUT_SEC:.1f}s."
            )
            return None

        if future.result() is None:
            self.get_logger().error(f"Failed to call {GET_CURRENT_POSX_SERVICE}.")
            return None

        response = future.result()
        if not response.success:
            self.get_logger().warn(f"{GET_CURRENT_POSX_SERVICE} returned success=False.")
            return None

        if not response.task_pos_info:
            self.get_logger().warn(f"{GET_CURRENT_POSX_SERVICE} returned no pose data.")
            return None

        robot_posx = list(response.task_pos_info[0].data[:6])
        if len(robot_posx) < 6:
            self.get_logger().warn(
                f"{GET_CURRENT_POSX_SERVICE} returned invalid pose: {robot_posx}"
            )
            return None

        return robot_posx

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        rotation = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [x, y, z]
        return transform

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        gripper2cam = np.load(gripper2cam_path)
        camera_coord = np.append(np.array(camera_coords), 1.0)

        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)
        base2camera = base2gripper @ gripper2cam
        base_coord = base2camera @ camera_coord

        return base_coord[:3]

    def get_rxyz_from_angles(
        self,
        horizontal_angle=None,
        vertical_angle=None,
        default_rxyz=(0.0, 180.0, 0.0),
    ):
        rx, ry, rz = default_rxyz
        if horizontal_angle is None and vertical_angle is None:
            return float(rx), float(ry), float(rz)

        yzx_euler = self._get_center_yzx_euler(
            horizontal_angle or 0.0,
            vertical_angle or 0.0,
        )
        target_rx, target_ry, target_rz = Rotation.from_euler(
            "YZX",
            yzx_euler,
            degrees=True,
        ).as_euler(
            "ZYZ",
            degrees=True,
        )
        return float(target_rx), float(target_ry), float(target_rz)

    def _get_center_yzx_euler(self, horizontal_angle, vertical_angle):
        horizontal_angle, vertical_angle = self._fold_gripper_symmetric_angles(
            horizontal_angle,
            vertical_angle,
        )
        return [179.0, horizontal_angle, vertical_angle]

    def _fold_gripper_symmetric_angles(self, horizontal_angle, vertical_angle):
        horizontal_angle = ((horizontal_angle + 180.0) % 360.0) - 180.0
        if horizontal_angle >= 90.0:
            return horizontal_angle - 180.0, -vertical_angle
        if horizontal_angle < -90.0:
            return horizontal_angle + 180.0, -vertical_angle
        return horizontal_angle, vertical_angle

    def get_rxyz_from_box(
        self,
        box,
        color_image=None,
        depth_image=None,
        camera_info=None,
        base_xyz=None,
    ):
        if color_image is None or depth_image is None or camera_info is None:
            color_image, depth_image, camera_info = self.get_frames_once()
        if color_image is None or depth_image is None or camera_info is None:
            self.get_logger().warn("Cannot calculate rxyz: RealSense frames are empty.")
            return self.get_rxyz_from_angles()

        pca_direction = self._calculate_pca_direction_from_box(color_image, box)
        if pca_direction is None:
            self.get_logger().warn("Cannot calculate rxyz: PCA direction is empty.")
            return self.get_rxyz_from_angles()

        horizontal_angle = self._calculate_horizontal_angle_from_pca_direction(
            pca_direction
        )
        vertical_angle = self._calculate_vertical_angle_from_pca_direction(
            pca_direction,
            depth_image,
            camera_info,
        )
        rx, ry, rz = self.get_rxyz_from_angles(horizontal_angle, vertical_angle)
        debug_pose = None
        if base_xyz is not None:
            x, y, z = base_xyz
            debug_pose = [x, y, z, rx, ry, rz]
        self._save_orientation_debug_image(
            color_image,
            depth_image,
            pca_direction,
            horizontal_angle,
            vertical_angle,
            debug_pose=debug_pose,
        )
        return rx, ry, rz

    def _calculate_pca_direction_from_box(self, image, box):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, min(int(x1), width - 1))
        x2 = max(0, min(int(x2), width))
        y1 = max(0, min(int(y1), height - 1))
        y2 = max(0, min(int(y2), height))
        if x2 <= x1 or y2 <= y1:
            return None

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        binary_mask = self._make_box_binary_mask(roi)
        points = np.column_stack(np.where(binary_mask > 0))
        if len(points) < PCA_MIN_POINTS:
            return None

        mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=None)
        center = mean[0]
        direction = eigenvectors[0]

        cx = int(center[1]) + x1
        cy = int(center[0]) + y1
        vx = float(direction[1])
        vy = float(direction[0])
        return x1, y1, x2, y2, cx, cy, vx, vy, binary_mask

    def _make_box_binary_mask(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_background = np.array(
            [
                PCA_BACKGROUND_HUE_MIN,
                PCA_BACKGROUND_SATURATION_MIN,
                PCA_BACKGROUND_VALUE_MIN,
            ],
            dtype=np.uint8,
        )
        upper_background = np.array(
            [
                PCA_BACKGROUND_HUE_MAX,
                PCA_BACKGROUND_SATURATION_MAX,
                PCA_BACKGROUND_VALUE_MAX,
            ],
            dtype=np.uint8,
        )
        background_mask = cv2.inRange(
            hsv_image,
            lower_background,
            upper_background,
        )
        object_mask = cv2.bitwise_not(background_mask)
        kernel = np.ones(
            (PCA_MASK_MORPH_KERNEL_SIZE, PCA_MASK_MORPH_KERNEL_SIZE),
            np.uint8,
        )
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)

    def _calculate_horizontal_angle_from_pca_direction(self, pca_direction):
        _, _, _, _, _, _, vx, vy, _ = pca_direction
        return float(np.degrees(np.arctan2(vy, vx)) + 90.0)

    def _calculate_vertical_angle_from_pca_direction(
        self,
        pca_direction,
        depth_image,
        camera_info,
    ):
        x1, y1, x2, y2, cx, cy, vx, vy, binary_mask = pca_direction
        pixel_points = self._sample_pca_line_pixels(
            binary_mask,
            x1,
            y1,
            x2,
            y2,
            cx,
            cy,
            vx,
            vy,
        )

        camera_points = []
        for px, py in pixel_points:
            depth = self._get_depth_or_nearest(depth_image, px, py)
            if depth is None:
                continue
            camera_coord = self._pixel_to_camera_coords(px, py, depth, camera_info)
            if camera_coord is not None:
                camera_points.append(camera_coord)

        if len(camera_points) < PCA_MIN_3D_POINTS:
            self.get_logger().warn(
                f"Not enough valid 3D PCA samples: {len(camera_points)}"
            )
            return None

        points = np.array(camera_points, dtype=np.float32)
        _, eigenvectors = cv2.PCACompute(points, mean=None)
        direction = eigenvectors[0]
        reference = points[-1] - points[0]
        if np.dot(direction, reference) < 0:
            direction = -direction

        lateral_length = float(np.linalg.norm(direction[:2]))
        if lateral_length <= 1e-6:
            return None

        return float(np.degrees(np.arctan2(direction[2], lateral_length)))

    def _save_orientation_debug_image(
        self,
        color_image,
        depth_image,
        pca_direction,
        horizontal_angle,
        vertical_angle,
        debug_pose=None,
    ):
        x1, y1, x2, y2, cx, cy, vx, vy, binary_mask = pca_direction
        pixel_points = self._sample_pca_line_pixels(
            binary_mask,
            x1,
            y1,
            x2,
            y2,
            cx,
            cy,
            vx,
            vy,
        )

        annotated = color_image.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self._draw_pca_line(annotated, cx, cy, vx, vy, (0, 255, 255))

        valid_depths = []
        for px, py in pixel_points:
            depth = self._get_raw_depth(depth_image, px, py)
            is_valid = self._is_valid_position_depth(depth)
            color = (255, 0, 255) if is_valid else (0, 0, 255)
            cv2.circle(annotated, (px, py), 4, color, -1)
            if is_valid:
                valid_depths.append((px, py, depth))

        center_depth = self._get_raw_depth(depth_image, cx, cy)

        self._draw_depth_labels(annotated, valid_depths)
        self._draw_orientation_text(
            annotated,
            horizontal_angle,
            vertical_angle,
            len(valid_depths),
            center_depth,
            debug_pose,
        )

        mask_panel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        self._draw_pca_line(mask_panel, cx - x1, cy - y1, vx, vy, (0, 255, 255))
        for px, py in pixel_points:
            roi_x = px - x1
            roi_y = py - y1
            depth = self._get_raw_depth(depth_image, px, py)
            color = (255, 0, 255) if self._is_valid_position_depth(depth) else (0, 0, 255)
            cv2.circle(mask_panel, (roi_x, roi_y), 3, color, -1)

        mask_panel = self._resize_debug_panel(mask_panel, annotated.shape[0])
        debug_image = np.hstack([annotated, mask_panel])

        os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
        stamp = self.get_clock().now().nanoseconds
        debug_path = os.path.join(DEBUG_IMAGE_DIR, f"orientation_debug_{stamp}.png")
        latest_path = os.path.join(DEBUG_IMAGE_DIR, "latest_orientation_debug.png")
        cv2.imwrite(debug_path, debug_image)
        cv2.imwrite(latest_path, debug_image)
        self.get_logger().info(f"Saved orientation debug image: {debug_path}")

    def _draw_pca_line(self, image, cx, cy, vx, vy, color):
        height, width = image.shape[:2]
        length = int(np.hypot(width, height) / 2)
        x1 = int(round(cx - vx * length))
        y1 = int(round(cy - vy * length))
        x2 = int(round(cx + vx * length))
        y2 = int(round(cy + vy * length))
        cv2.line(image, (x1, y1), (x2, y2), color, 2)
        cv2.circle(image, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    def _draw_depth_labels(self, image, valid_depths):
        if not valid_depths:
            return

        step = max(1, len(valid_depths) // DEBUG_MAX_DEPTH_LABELS)
        for index, (px, py, depth) in enumerate(valid_depths):
            if index % step != 0:
                continue

            cv2.putText(
                image,
                f"{depth:.3f}",
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"{depth:.3f}",
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    def _draw_orientation_text(
        self,
        image,
        horizontal_angle,
        vertical_angle,
        valid_depth_count,
        center_depth=None,
        debug_pose=None,
    ):
        horizontal_text = "horizontal: n/a"
        if horizontal_angle is not None:
            horizontal_text = f"horizontal: {horizontal_angle:.2f} deg"

        vertical_text = "vertical: n/a"
        if vertical_angle is not None:
            vertical_text = f"vertical: {vertical_angle:.2f} deg"

        lines = [
            horizontal_text,
            vertical_text,
            f"valid depth samples: {valid_depth_count}",
        ]
        if center_depth is not None:
            lines.append(f"center depth: {center_depth:.3f}")
        if debug_pose is not None:
            x, y, z, rx, ry, rz = debug_pose
            lines.extend(
                [
                    f"pos xyz: {x:.2f}, {y:.2f}, {z:.2f}",
                    f"pos rxyz: {rx:.2f}, {ry:.2f}, {rz:.2f}",
                ]
            )
        x = 12
        y = 28
        for line in lines:
            cv2.putText(
                image,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 30

    def _resize_debug_panel(self, image, target_height):
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return image

        scale = target_height / float(height)
        target_width = max(1, int(round(width * scale)))
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    def _get_raw_depth(self, depth_image, x, y):
        height, width = depth_image.shape[:2]
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        return float(depth_image[y, x])

    def _sample_pca_line_pixels(
        self,
        binary_mask,
        x1,
        y1,
        x2,
        y2,
        cx,
        cy,
        vx,
        vy,
    ):
        max_length = int(np.hypot(x2 - x1, y2 - y1) / 2)
        samples = []
        seen = set()
        for offset in range(-max_length, max_length + 1, PCA_LINE_SAMPLE_STEP_PX):
            px = int(round(cx + vx * offset))
            py = int(round(cy + vy * offset))
            if px < x1 or px >= x2 or py < y1 or py >= y2:
                continue

            roi_x = px - x1
            roi_y = py - y1
            if binary_mask[roi_y, roi_x] == 0:
                continue

            key = (px, py)
            if key in seen:
                continue

            seen.add(key)
            samples.append(key)

        return samples

    def _publish_latest_detection_image(self):
        if self.latest_detection_image_msg is not None:
            self.latest_detection_image_msg.header.stamp = self.get_clock().now().to_msg()
            self.detection_image_pub.publish(self.latest_detection_image_msg)

    

def main(args=None):
    rclpy.init(args=args)
    node = DetectCalPosService()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.frame_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
