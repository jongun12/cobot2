import cv2
import numpy as np
import pyrealsense2 as rs


GRID = 50
MIN_LENGTH_GRID = 1.6
MAX_LENGTH_GRID = 3.5
MIN_THICK_GRID = 0.20
MAX_THICK_GRID = 0.50
MIN_ASPECT_RATIO = 1.0
THRESHOLD_VALUE = 105
MAX_LINE_COUNT = 4


def preprocess_ir(ir_image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(ir_image)
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


def count_lines_from_ir_image(ir_image):
    clean_image = preprocess_ir(ir_image)
    line_count = count_line_candidates(clean_image)
    return max(0, min(MAX_LINE_COUNT, line_count))


def get_realsense_ir_image(timeout_ms=2000, warmup_frames=5):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.emitter_enabled, 0)

    try:
        ir_frame = None
        for _ in range(warmup_frames + 1):
            frames = pipeline.wait_for_frames(timeout_ms)
            ir_frame = frames.get_infrared_frame(1)

        if not ir_frame:
            return None

        return np.asanyarray(ir_frame.get_data()).copy()
    finally:
        pipeline.stop()


def get_realsense_line_count(timeout_ms=2000, warmup_frames=5):
    ir_image = get_realsense_ir_image(
        timeout_ms=timeout_ms,
        warmup_frames=warmup_frames,
    )
    if ir_image is None:
        return 0

    return count_lines_from_ir_image(ir_image)


def calculate_water_level(tape_count):
    return max(0, 100 - (tape_count * 25))


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.emitter_enabled, 0)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                continue

            ir_image = np.asanyarray(ir_frame.get_data())
            line_count = count_lines_from_ir_image(ir_image)
            water_level = calculate_water_level(line_count)

            print(f"line_count={line_count}, water_level={water_level}%")

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        print("Program End")


if __name__ == "__main__":
    main()
