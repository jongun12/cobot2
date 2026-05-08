import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import torch
import pyrealsense2 as rs

from ultralytics import YOLO
from recycle_robot_interface.msg import ObjectInfo
from ament_index_python.packages import get_package_share_directory

import os


class YoloRS(Node):

    def __init__(self):

        super().__init__('yolo_rs')

        # =================================================
        # YOLO
        # =================================================

        pkg_path = get_package_share_directory('recycle_robot')

        model_path = os.path.join(
            pkg_path,
            'resource',
            'best.pt'
        )

        self.model = YOLO(model_path)

        if torch.cuda.is_available():
            self.model.to('cuda')
            print("CUDA ENABLED")

        # =================================================
        # ROS Publisher
        # =================================================

        self.publisher = self.create_publisher(
            ObjectInfo,
            'dsr01/object_info',
            10
        )

        # =================================================
        # RealSense
        # =================================================

        self.pipeline = rs.pipeline()

        config = rs.config()

        config.enable_stream(
            rs.stream.color,
            640,
            480,
            rs.format.bgr8,
            30
        )

        config.enable_stream(
            rs.stream.depth,
            640,
            480,
            rs.format.z16,
            30
        )

        profile = self.pipeline.start(config)

        self.align = rs.align(rs.stream.color)

        # depth intrinsics
        depth_stream = profile.get_stream(rs.stream.depth)

        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        self.create_timer(0.03, self.loop)

        print("🚀 YOLO + RealSense START")

    # =====================================================
    # Main Loop
    # =====================================================

    def loop(self):

        frames = self.pipeline.wait_for_frames()

        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        img = np.asanyarray(color_frame.get_data())

        # =================================================
        # YOLO
        # =================================================

        results = self.model(
            img,
            imgsz=192,
            conf=0.5,
            verbose=False
        )[0]

        low_conf_centers = []

        for b in results.boxes:

            x1, y1, x2, y2 = map(int, b.xyxy[0])

            conf = float(b.conf[0])

            cls_id = int(b.cls[0])

            label = results.names[cls_id]

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                img,
                (cx, cy),
                4,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            # =============================================
            # LOW CONFIDENCE OBJECTS
            # =============================================

            if conf < 0.9:
                low_conf_centers.append((cx, cy))

        # =================================================
        # Average Center
        # =================================================

        if len(low_conf_centers) > 0:

            avg_x = int(np.mean([p[0] for p in low_conf_centers]))
            avg_y = int(np.mean([p[1] for p in low_conf_centers]))

            # =============================================
            # Depth
            # =============================================

            depth = depth_frame.get_distance(avg_x, avg_y)

            if depth <= 0:
                return

            # =============================================
            # Pixel -> Camera XYZ
            # =============================================

            point_3d = rs.rs2_deproject_pixel_to_point(
                self.depth_intrinsics,
                [avg_x, avg_y],
                depth
            )

            # meter -> mm
            cam_x = point_3d[0] * 1000.0
            cam_y = point_3d[1] * 1000.0
            cam_z = point_3d[2] * 1000.0

            print("\n========================")
            print("CAMERA XYZ (mm)")
            print(cam_x, cam_y, cam_z)
            print("========================")

            # =============================================
            # Draw
            # =============================================

            cv2.circle(
                img,
                (avg_x, avg_y),
                10,
                (255, 0, 0),
                -1
            )

            # =============================================
            # Publish
            # =============================================

            msg = ObjectInfo()

            msg.x = float(cam_x)
            msg.y = float(cam_y)
            msg.z = float(cam_z)

            self.publisher.publish(msg)

        cv2.imshow("YOLO", img)

        cv2.waitKey(1)

    def destroy_node(self):

        self.pipeline.stop()

        cv2.destroyAllWindows()

        super().destroy_node()


def main():

    rclpy.init()

    node = YoloRS()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()