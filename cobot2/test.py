import cv2
import numpy as np
import os
import rclpy
import statistics
import time

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


GRID = 50
MIN_LENGTH_GRID = 3.5
MAX_LENGTH_GRID = 8.0
MIN_THICK_GRID = 0.10
MAX_THICK_GRID = 2.3
MIN_ASPECT_RATIO = 1.0
THRESHOLD_VALUE = 125
MAX_LINE_COUNT = 4
FRAME_TIMEOUT_SEC = 2.0
DEFAULT_SAMPLE_DURATION_SEC = 3.0
DEBUG_IMAGE_DIR = "/tmp/cobot2_line_count_debug"
ROI_X1 = 600
ROI_Y1 = 250
ROI_X2 = 900
ROI_Y2 = 600
COLOR_IMAGE_TOPIC = "/camera/camera/color/image_raw"


class ColorFrameNode(Node):
    def __init__(self):
        super().__init__("line_count_color_frame_node")
        self.bridge = CvBridge()
        self.color_msg = None
        self.create_subscription(
            Image,
            COLOR_IMAGE_TOPIC,
            self.color_callback,
            qos_profile_sensor_data,
        )

    def color_callback(self, msg):
        self.color_msg = msg


def preprocess_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred,
        THRESHOLD_VALUE,
        255,
        cv2.THRESH_BINARY_INV,
    )
    kernel = np.ones((3, 3), np.uint8)
    clean_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    roi_mask = np.zeros_like(clean_image)
    cv2.rectangle(
        roi_mask,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
        255,
        -1,
    )
    return cv2.bitwise_and(clean_image, roi_mask)


def count_line_candidates(clean_image):
    contours, _ = cv2.findContours(
        clean_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue

        _, _, width, height = cv2.boundingRect(cnt)
        if width < height:
            continue

        length_grid = width / GRID
        thickness_grid = height / GRID
        aspect_ratio = width / (height + 1e-6)

        if not (MIN_LENGTH_GRID <= length_grid <= MAX_LENGTH_GRID):
            continue
        if not (MIN_THICK_GRID <= thickness_grid <= MAX_THICK_GRID):
            continue
        if aspect_ratio < MIN_ASPECT_RATIO:
            continue

        count += 1

    return count


def count_lines_from_image(image):
    clean_image = preprocess_image(image)
    line_count = count_line_candidates(clean_image)
    return max(0, min(MAX_LINE_COUNT, line_count))


def save_debug_image(image, line_count, output_dir=DEBUG_IMAGE_DIR):
    os.makedirs(output_dir, exist_ok=True)

    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_image = image.copy()
    else:
        gray_image = image
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    clean_image = preprocess_image(image)
    clean_vis = cv2.cvtColor(clean_image, cv2.COLOR_GRAY2BGR)
    detection_vis = np.zeros_like(color_image)
    detection_vis[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = color_image[
        ROI_Y1:ROI_Y2,
        ROI_X1:ROI_X2,
    ]
    cv2.rectangle(
        detection_vis,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
        (0, 255, 255),
        2,
    )

    contours, _ = cv2.findContours(
        clean_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue

        x, y, width, height = cv2.boundingRect(cnt)
        if width < height:
            continue

        length_grid = width / GRID
        thickness_grid = height / GRID
        aspect_ratio = width / (height + 1e-6)

        if not (MIN_LENGTH_GRID <= length_grid <= MAX_LENGTH_GRID):
            continue
        if not (MIN_THICK_GRID <= thickness_grid <= MAX_THICK_GRID):
            continue
        if aspect_ratio < MIN_ASPECT_RATIO:
            continue

        cv2.rectangle(clean_vis, (x, y), (x + width, y + height), (0, 0, 255), 2)

        center_y = y + height // 2
        center_x = x + width // 2
        cv2.line(clean_vis, (x, center_y), (x + width, center_y), (255, 0, 0), 2)
        cv2.circle(clean_vis, (center_x, center_y), 4, (0, 255, 255), -1)
        cv2.putText(
            clean_vis,
            f"L:{length_grid:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            clean_vis,
            f"T:{thickness_grid:.2f}",
            (x, y - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

    water_level = calculate_water_level(line_count)
    bar_x, bar_y, bar_w, bar_h = 580, 100, 30, 300
    cv2.rectangle(
        detection_vis,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        (60, 60, 60),
        -1,
    )
    fill_h = int((water_level / 100) * bar_h)
    cv2.rectangle(
        detection_vis,
        (bar_x, bar_y + bar_h - fill_h),
        (bar_x + bar_w, bar_y + bar_h),
        (255, 100, 0),
        -1,
    )
    cv2.putText(
        detection_vis,
        f"WATER LEVEL: {water_level}%",
        (320, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        detection_vis,
        f"Tape Count: {line_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(output_dir, f"line_count_raw_{timestamp}.jpg")
    clean_path = os.path.join(output_dir, f"line_count_clean_analysis_{timestamp}.jpg")
    detection_path = os.path.join(
        output_dir,
        f"line_count_detection_water_level_{timestamp}.jpg",
    )
    latest_clean_path = os.path.join(output_dir, "latest_clean_analysis.jpg")
    latest_detection_path = os.path.join(
        output_dir,
        "latest_detection_water_level.jpg",
    )

    write_results = [
        cv2.imwrite(raw_path, color_image),
        cv2.imwrite(clean_path, clean_vis),
        cv2.imwrite(detection_path, detection_vis),
        cv2.imwrite(latest_clean_path, clean_vis),
        cv2.imwrite(latest_detection_path, detection_vis),
    ]
    if not all(write_results):
        print(f"Failed to write one or more debug images to {output_dir}")

    return detection_path


def get_realsense_color_image(timeout_sec=FRAME_TIMEOUT_SEC):
    frame_node = ColorFrameNode()
    deadline = frame_node.get_clock().now().nanoseconds + int(timeout_sec * 1e9)

    while rclpy.ok() and frame_node.color_msg is None:
        rclpy.spin_once(frame_node, timeout_sec=0.1)
        if frame_node.get_clock().now().nanoseconds >= deadline:
            frame_node.destroy_node()
            return None

    color_image = frame_node.bridge.imgmsg_to_cv2(
        frame_node.color_msg,
        desired_encoding="bgr8",
    )
    frame_node.destroy_node()
    return color_image


def get_realsense_line_count(
    timeout_sec=FRAME_TIMEOUT_SEC,
    sample_duration_sec=DEFAULT_SAMPLE_DURATION_SEC,
    save_debug=False,
):
    frame_node = ColorFrameNode()
    first_frame_deadline = (
        frame_node.get_clock().now().nanoseconds
        + int(timeout_sec * 1e9)
    )
    sample_deadline = None
    line_counts = []
    latest_color_image = None

    while rclpy.ok():
        rclpy.spin_once(frame_node, timeout_sec=0.1)
        now_ns = frame_node.get_clock().now().nanoseconds

        if frame_node.color_msg is None:
            if now_ns >= first_frame_deadline:
                frame_node.destroy_node()
                if save_debug:
                    print(
                        "No RealSense color frame received. "
                        "Debug image was not saved."
                    )
                return 0
            continue

        if sample_deadline is None:
            sample_deadline = now_ns + int(sample_duration_sec * 1e9)

        color_image = frame_node.bridge.imgmsg_to_cv2(
            frame_node.color_msg,
            desired_encoding="bgr8",
        )
        latest_color_image = color_image
        line_counts.append(count_lines_from_image(color_image))
        frame_node.color_msg = None

        if now_ns >= sample_deadline:
            break

    frame_node.destroy_node()

    if not line_counts:
        return 0

    line_count = int(statistics.median(line_counts))
    if save_debug:
        debug_path = save_debug_image(latest_color_image, line_count)
        print(
            f"Saved line count debug image: {debug_path}, "
            f"output_dir={DEBUG_IMAGE_DIR}, "
            f"samples={line_counts}"
        )

    return line_count


def calculate_water_level(tape_count):
    return max(0, 100 - (tape_count * 25))


def main():
    rclpy.init()
    try:
        line_count = get_realsense_line_count(save_debug=True)
        water_level = calculate_water_level(line_count)
        print(f"median_line_count={line_count}, water_level={water_level}%")
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        print("Program End")


if __name__ == "__main__":
    main()
