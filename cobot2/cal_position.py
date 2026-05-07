import os
import threading

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from dsr_msgs2.srv import GetCurrentPosx
from od_msg.srv import (
    SrvBasePositions,
    SrvCameraInfo,
    SrvDetections,
    SrvImage,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation


DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.5
SERVICE_TIMEOUT_SEC = 15.0
GET_CURRENT_POSX_SERVICE = "/dsr01/aux_control/get_current_posx"
PCA_THRESHOLD_VALUE = 50
PCA_MIN_POINTS = 10
PCA_LINE_SAMPLE_STEP_PX = 5
PCA_MIN_3D_POINTS = 4
DEBUG_IMAGE_DIR = "/tmp/cobot2_cal_position_debug"
DEBUG_MAX_DEPTH_LABELS = 12
DEPTH_OUTLIER_ABS_THRESHOLD = 0.03
DEPTH_OUTLIER_MAD_SCALE = 3.0


class CalPositionNode(Node):
    def __init__(self):
        super().__init__("cal_position_node")
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()
        self.detect_client = self.create_client(
            SrvDetections,
            "detect_objects",
            callback_group=self.callback_group,
        )
        self.center_detect_client = self.create_client(
            SrvDetections,
            "detect_center_object",
            callback_group=self.callback_group,
        )
        self.depth_image_client = self.create_client(
            SrvImage,
            "get_depth_image",
            callback_group=self.callback_group,
        )
        self.color_image_client = self.create_client(
            SrvImage,
            "get_color_image",
            callback_group=self.callback_group,
        )
        self.camera_info_client = self.create_client(
            SrvCameraInfo,
            "get_camera_info",
            callback_group=self.callback_group,
        )
        self.current_posx_client = self.create_client(
            GetCurrentPosx,
            GET_CURRENT_POSX_SERVICE,
            callback_group=self.callback_group,
        )
        self.create_service(
            SrvBasePositions,
            "get_base_positions",
            self.handle_get_base_positions,
            callback_group=self.callback_group,
        )
        self.create_service(
            SrvBasePositions,
            "get_center_base_positions",
            self.handle_get_center_base_positions,
            callback_group=self.callback_group,
        )
        self.gripper2cam_path = os.path.join(
            get_package_share_directory("cobot2"),
            "resource",
            "T_gripper2camera.npy",
        )

    def _wait_for_future(self, future, timeout_sec):
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())
        return done_event.wait(timeout=timeout_sec)

    def handle_get_base_positions(self, request, response):
        self.get_logger().info("get_base_positions request received.")
        base_positions = self.calculate_base_positions()
        response.boxes = self._flatten_boxes(base_positions)
        response.class_ids = [position["class_id"] for position in base_positions]
        response.xs = [position["x"] for position in base_positions]
        response.ys = [position["y"] for position in base_positions]
        response.zs = [position["z"] for position in base_positions]
        response.rxs = [position["rx"] for position in base_positions]
        response.rys = [position["ry"] for position in base_positions]
        response.rzs = [position["rz"] for position in base_positions]
        response.success = bool(base_positions)
        response.message = f"Calculated {len(base_positions)} base positions."
        return response

    def handle_get_center_base_positions(self, request, response):
        self.get_logger().info("get_center_base_positions request received.")
        base_positions = self.calculate_center_base_positions()
        response.boxes = self._flatten_boxes(base_positions)
        response.class_ids = [position["class_id"] for position in base_positions]
        response.xs = [position["x"] for position in base_positions]
        response.ys = [position["y"] for position in base_positions]
        response.zs = [position["z"] for position in base_positions]
        response.rxs = [position["rx"] for position in base_positions]
        response.rys = [position["ry"] for position in base_positions]
        response.rzs = [position["rz"] for position in base_positions]
        response.success = bool(base_positions)
        response.message = (
            f"Calculated {len(base_positions)} center base positions."
        )
        return response

    def request_detections(
        self,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold=DEFAULT_IOU_THRESHOLD,
        detect_client=None,
        request_type=SrvDetections,
        service_name="detect_objects",
    ):
        if detect_client is None:
            detect_client = self.detect_client

        while not detect_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for {service_name} service...")

        request = request_type.Request()
        request.confidence_threshold = float(confidence_threshold)
        request.iou_threshold = float(iou_threshold)

        self.get_logger().info(f"Calling {service_name} service...")
        future = detect_client.call_async(request)

        if not self._wait_for_future(future, SERVICE_TIMEOUT_SEC):
            future.cancel()
            self.get_logger().error(
                f"Timed out waiting for {service_name} after {SERVICE_TIMEOUT_SEC:.1f}s."
            )
            return []

        if future.result() is None:
            self.get_logger().error(f"Failed to call {service_name} service.")
            return []

        detections = self._parse_detection_response(future.result())
        self.get_logger().info(f"Received {len(detections)} detections.")
        for detection in detections:
            self.get_logger().info(
                "box=%s, class_id=%d, score=%.3f"
                % (
                    detection["box"],
                    detection["class_id"],
                    detection["score"],
                )
            )
        return detections

    def request_depth_image(self):
        while not self.depth_image_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for get_depth_image service...")

        self.get_logger().info("Calling get_depth_image service...")
        future = self.depth_image_client.call_async(SrvImage.Request())

        if not self._wait_for_future(future, SERVICE_TIMEOUT_SEC):
            future.cancel()
            self.get_logger().error(
                f"Timed out waiting for get_depth_image after {SERVICE_TIMEOUT_SEC:.1f}s."
            )
            return None

        if future.result() is None:
            self.get_logger().error("Failed to call get_depth_image service.")
            return None

        response = future.result()
        if not response.success:
            self.get_logger().warn(response.message)
            return None

        depth_image = self.bridge.imgmsg_to_cv2(
            response.image,
            desired_encoding="passthrough",
        )
        self.get_logger().info(
            f"Received depth image: shape={depth_image.shape}, dtype={depth_image.dtype}"
        )
        return depth_image

    def request_color_image(self):
        while not self.color_image_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for get_color_image service...")

        self.get_logger().info("Calling get_color_image service...")
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

        color_image = self.bridge.imgmsg_to_cv2(
            response.image,
            desired_encoding="bgr8",
        )
        self.get_logger().info(
            f"Received color image: shape={color_image.shape}, dtype={color_image.dtype}"
        )
        return color_image

    def request_camera_info(self):
        while not self.camera_info_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for get_camera_info service...")

        self.get_logger().info("Calling get_camera_info service...")
        future = self.camera_info_client.call_async(SrvCameraInfo.Request())

        if not self._wait_for_future(future, SERVICE_TIMEOUT_SEC):
            future.cancel()
            self.get_logger().error(
                f"Timed out waiting for get_camera_info after {SERVICE_TIMEOUT_SEC:.1f}s."
            )
            return None

        if future.result() is None:
            self.get_logger().error("Failed to call get_camera_info service.")
            return None

        response = future.result()
        if not response.success:
            self.get_logger().warn(response.message)
            return None

        return response.camera_info

    def calculate_camera_positions(self, detections=None):
        return self._calculate_camera_positions(detections=detections)

    def calculate_center_camera_positions(self, detections):
        return self._calculate_camera_positions(
            detections=detections,
            calculate_orientation=True,
        )

    def _calculate_camera_positions(self, detections=None, calculate_orientation=False):
        if detections is None:
            detections = self.request_detections()
        depth_image = self.request_depth_image()
        camera_info = self.request_camera_info()

        if depth_image is None or camera_info is None:
            return []

        color_image = None
        if calculate_orientation:
            color_image = self.request_color_image()

        positions = []
        for detection in detections:
            cx, cy = self._get_box_center(detection["box"])
            depth = self._get_depth(depth_image, cx, cy)
            if not self._is_valid_depth(depth):
                self.get_logger().warn(f"Invalid depth at ({cx}, {cy}): {depth}")
                continue

            camera_position = self._pixel_to_camera_coords(cx, cy, depth, camera_info)
            position = {
                "box": detection["box"],
                "class_id": detection["class_id"],
                "x": camera_position[0],
                "y": camera_position[1],
                "z": camera_position[2],
            }
            if calculate_orientation and color_image is not None:
                self._attach_orientation_angles(
                    position,
                    color_image,
                    depth_image,
                    camera_info,
                    detection["box"],
                )
            positions.append(position)
            self.get_logger().info(
                "class_id=%d, x=%.3f, y=%.3f, z=%.3f"
                % (
                    detection["class_id"],
                    camera_position[0],
                    camera_position[1],
                    camera_position[2],
                )
            )
            if "vertical_angle" in position:
                self.get_logger().info(
                    "class_id=%d, vertical_angle=%.3f deg"
                    % (detection["class_id"], position["vertical_angle"])
                )
            if "angle" in position:
                self.get_logger().info(
                    "class_id=%d, horizontal_angle=%.3f deg"
                    % (detection["class_id"], position["angle"])
                )

        return positions

    def calculate_base_positions(self):
        camera_positions = self.calculate_camera_positions()
        return self._calculate_base_positions_from_camera_positions(camera_positions)

    def calculate_center_base_positions(self):
        detections = self.request_detections(
            detect_client=self.center_detect_client,
            service_name="detect_center_object",
        )
        camera_positions = self.calculate_center_camera_positions(detections)
        return self._calculate_base_positions_from_camera_positions(camera_positions)

    def _calculate_base_positions_from_camera_positions(self, camera_positions):
        if not camera_positions:
            return []

        robot_posx = self.request_robot_posx()
        if robot_posx is None:
            return []
        self.get_logger().info(f"Current robot posx: {robot_posx}")
        rx, ry, rz = (0, 180, 0)

        base_positions = []
        for camera_position in camera_positions:
            base_coord = self.transform_to_base(
                [
                    camera_position["x"],
                    camera_position["y"],
                    camera_position["z"],
                ],
                self.gripper2cam_path,
                robot_posx,
            )
            target_rx, target_ry, target_rz = self._get_target_zyz_euler(
                rx,
                ry,
                rz,
                camera_position,
            )
            base_positions.append(
                {
                    "box": camera_position["box"],
                    "class_id": camera_position["class_id"],
                    "x": float(base_coord[0]),
                    "y": float(base_coord[1]),
                    "z": float(base_coord[2]),
                    "rx": float(target_rx),
                    "ry": float(target_ry),
                    "rz": float(target_rz),
                }
            )
            self.get_logger().info(
                "class_id=%d, base=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]"
                % (
                    camera_position["class_id"],
                    base_coord[0],
                    base_coord[1],
                    base_coord[2],
                    target_rx,
                    target_ry,
                    target_rz,
                )
            )

        return base_positions

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

    def _get_target_zyz_euler(self, rx, ry, rz, camera_position):
        horizontal_angle = camera_position.get("angle")
        vertical_angle = camera_position.get("vertical_angle")
        if horizontal_angle is None and vertical_angle is None:
            return rx, ry, rz

        yzx_euler = self._get_center_yzx_euler(
            horizontal_angle or 0.0,
            vertical_angle or 0.0,
        )
        return Rotation.from_euler("YZX", yzx_euler, degrees=True).as_euler(
            "ZYZ",
            degrees=True,
        )

    def _get_center_yzx_euler(self, horizontal_angle, vertical_angle): # horizontal_angle 좌우각도, vertical_angle 상하각도
        return [179, horizontal_angle, vertical_angle]

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1)

        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)
        base2cam = base2gripper @ gripper2cam
        base_coord = base2cam @ coord

        return base_coord[:3]

    def _get_box_center(self, box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _get_depth(self, depth_image, x, y):
        height, width = depth_image.shape[:2]
        if x < 0 or y < 0 or x >= width or y >= height:
            self.get_logger().warn(f"Coordinates ({x}, {y}) out of range.")
            return None
        return float(depth_image[y, x])

    def _pixel_to_camera_coords(self, x, y, z, camera_info):
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        ppx = camera_info.k[2]
        ppy = camera_info.k[5]
        return [
            float((x - ppx) * z / fx),
            float((y - ppy) * z / fy),
            float(z),
        ]

    def _attach_orientation_angles(
        self,
        position,
        color_image,
        depth_image,
        camera_info,
        box,
    ):
        pca_direction = self._calculate_pca_direction(color_image, box)
        if pca_direction is None:
            return

        horizontal_angle = self._calculate_horizontal_angle_from_pca_direction(
            pca_direction
        )
        if horizontal_angle is not None:
            position["angle"] = horizontal_angle

        vertical_angle = self._calculate_vertical_angle_from_pca_direction(
            pca_direction,
            depth_image,
            camera_info,
        )
        if vertical_angle is not None:
            position["vertical_angle"] = vertical_angle

        self._save_orientation_debug_image(
            color_image,
            depth_image,
            pca_direction,
            horizontal_angle,
            vertical_angle,
        )

    def _calculate_horizontal_angle_from_pca_direction(self, pca_direction):
        _, _, _, _, _, _, vx, vy, _ = pca_direction
        return float(np.degrees(np.arctan2(vy, vx)) + 90.0)

    def _calculate_vertical_angle_from_pca_direction(
        self,
        pca_direction,
        depth_image,
        camera_info,
    ):
        x1, y1, x2, y2, cx, cy, vx, vy, mask = pca_direction
        pixel_points = self._sample_pca_line_pixels(
            mask,
            x1,
            y1,
            x2,
            y2,
            cx,
            cy,
            vx,
            vy,
        )
        depth_samples = []
        for px, py in pixel_points:
            depth = self._get_depth(depth_image, px, py)
            if not self._is_valid_depth(depth):
                continue
            depth_samples.append((px, py, depth))

        filtered_depths, _ = self._filter_depth_outliers(depth_samples)

        camera_points = []
        for px, py, depth in filtered_depths:
            camera_points.append(
                self._pixel_to_camera_coords(px, py, depth, camera_info)
            )

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
    ):
        x1, y1, x2, y2, cx, cy, vx, vy, mask = pca_direction
        pixel_points = self._sample_pca_line_pixels(
            mask,
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
            depth = self._get_depth(depth_image, px, py)
            is_valid = self._is_valid_depth(depth)
            color = (255, 0, 255) if is_valid else (0, 0, 255)
            cv2.circle(annotated, (px, py), 4, color, -1)
            if is_valid:
                valid_depths.append((px, py, depth))

        filtered_depths, outlier_depths = self._filter_depth_outliers(valid_depths)
        for px, py, _ in outlier_depths:
            cv2.circle(annotated, (px, py), 6, (0, 165, 255), 2)

        self._draw_depth_labels(annotated, filtered_depths)
        self._draw_orientation_text(
            annotated,
            horizontal_angle,
            vertical_angle,
            len(filtered_depths),
            len(outlier_depths),
        )

        mask_panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self._draw_pca_line(mask_panel, cx - x1, cy - y1, vx, vy, (0, 255, 255))
        for px, py in pixel_points:
            roi_x = px - x1
            roi_y = py - y1
            depth = self._get_depth(depth_image, px, py)
            color = (255, 0, 255) if self._is_valid_depth(depth) else (0, 0, 255)
            cv2.circle(mask_panel, (roi_x, roi_y), 3, color, -1)
        for px, py, _ in outlier_depths:
            cv2.circle(mask_panel, (px - x1, py - y1), 5, (0, 165, 255), 1)

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
        outlier_depth_count,
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
            f"depth outliers: {outlier_depth_count}",
        ]
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
        target_width = max(1, int(width * scale))
        return cv2.resize(image, (target_width, target_height))

    def _filter_depth_outliers(self, depth_samples):
        if len(depth_samples) < PCA_MIN_3D_POINTS:
            return depth_samples, []

        depths = np.array(
            [sample[2] for sample in depth_samples],
            dtype=np.float32,
        )
        median = float(np.median(depths))
        absolute_deviations = np.abs(depths - median)
        mad = float(np.median(absolute_deviations))
        threshold = DEPTH_OUTLIER_ABS_THRESHOLD
        if mad > 1e-9:
            threshold = max(threshold, DEPTH_OUTLIER_MAD_SCALE * 1.4826 * mad)

        filtered = []
        outliers = []
        for sample, deviation in zip(depth_samples, absolute_deviations):
            if deviation <= threshold:
                filtered.append(sample)
            else:
                outliers.append(sample)

        return filtered, outliers

    def _calculate_pca_direction(self, image, box):
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

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(
            gray,
            PCA_THRESHOLD_VALUE,
            255,
            cv2.THRESH_BINARY_INV,
        )

        points = np.column_stack(np.where(mask > 0))
        if len(points) < PCA_MIN_POINTS:
            return None

        mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=None)
        center = mean[0]
        direction = eigenvectors[0]

        cx = int(center[1]) + x1
        cy = int(center[0]) + y1
        vx, vy = float(direction[1]), float(direction[0])
        return x1, y1, x2, y2, cx, cy, vx, vy, mask

    def _sample_pca_line_pixels(
        self,
        mask,
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
            if mask[roi_y, roi_x] == 0:
                continue

            key = (px, py)
            if key in seen:
                continue

            seen.add(key)
            samples.append(key)

        return samples

    def _is_valid_depth(self, depth):
        return depth is not None and np.isfinite(depth) and depth > 0

    def _flatten_boxes(self, positions):
        boxes = []
        for position in positions:
            boxes.extend(float(value) for value in position["box"])
        return boxes

    def _parse_detection_response(self, response):
        boxes = list(response.boxes)
        class_ids = list(response.class_ids)
        scores = list(response.scores)

        if len(boxes) % 4 != 0:
            self.get_logger().warn(
                f"Invalid boxes length: {len(boxes)}. Expected a multiple of 4."
            )

        box_count = len(boxes) // 4
        detection_count = min(box_count, len(class_ids), len(scores))
        detections = []

        for index in range(detection_count):
            box_start = index * 4
            detections.append(
                {
                    "box": boxes[box_start : box_start + 4],
                    "class_id": int(class_ids[index]),
                    "score": float(scores[index]),
                }
            )

        return detections


def main(args=None):
    rclpy.init(args=args)
    node = CalPositionNode()
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
