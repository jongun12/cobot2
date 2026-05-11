import cv2
import numpy as np
import rclpy

from cobot2.realsense3 import RealsenseFrameNode


GRID = 50
MIN_LENGTH_GRID = 1.6
MAX_LENGTH_GRID = 3.5
MIN_THICK_GRID = 0.20
MAX_THICK_GRID = 0.50
MIN_ASPECT_RATIO = 1.0
THRESHOLD_VALUE = 105
MAX_LINE_COUNT = 4
FRAME_TIMEOUT_SEC = 3.0


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
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


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


def get_realsense_color_image(timeout_sec=FRAME_TIMEOUT_SEC):
    frame_node = RealsenseFrameNode("line_count_realsense_frame_node")
    deadline = frame_node.get_clock().now().nanoseconds + int(timeout_sec * 1e9)

    while rclpy.ok() and not frame_node.has_color_frame():
        rclpy.spin_once(frame_node, timeout_sec=0.1)
        if frame_node.get_clock().now().nanoseconds >= deadline:
            frame_node.destroy_node()
            return None

    color_image = frame_node.get_color_image()
    frame_node.destroy_node()
    return color_image


def get_realsense_line_count(timeout_sec=FRAME_TIMEOUT_SEC):
    color_image = get_realsense_color_image(timeout_sec=timeout_sec)
    if color_image is None:
        return 0

    return count_lines_from_image(color_image)


def calculate_water_level(tape_count):
    return max(0, 100 - (tape_count * 25))


def main():
    rclpy.init()
    frame_node = RealsenseFrameNode("line_count_realsense_frame_node")
    try:
        while rclpy.ok():
            rclpy.spin_once(frame_node, timeout_sec=0.1)
            color_image = frame_node.get_color_image()
            if color_image is None:
                continue

            line_count = count_lines_from_image(color_image)
            water_level = calculate_water_level(line_count)

            print(f"line_count={line_count}, water_level={water_level}%")

    except KeyboardInterrupt:
        pass
    finally:
        frame_node.destroy_node()
        rclpy.shutdown()
        print("Program End")


if __name__ == "__main__":
    main()
