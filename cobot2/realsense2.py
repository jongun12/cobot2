import copy

import rclpy
from od_msg.srv import SrvCameraInfo, SrvImage
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger


class RealsenseServiceNode(Node):
    def __init__(self):
        super().__init__("realsense_service_node")

        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_camera_info = None

        self.cached_color_image = None
        self.cached_depth_image = None
        self.cached_camera_info = None

        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.color_callback,
            10,
        )
        self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_callback,
            10,
        )
        self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_info_callback,
            10,
        )

        self.create_service(Trigger, "update_images", self.handle_update_images)
        self.create_service(SrvImage, "get_color_image", self.handle_get_color_image)
        self.create_service(SrvImage, "get_depth_image", self.handle_get_depth_image)
        self.create_service(SrvCameraInfo, "get_camera_info", self.handle_get_camera_info)

        self.get_logger().info("RealsenseServiceNode initialized.")

    def color_callback(self, msg):
        self.latest_color_image = msg

    def depth_callback(self, msg):
        self.latest_depth_image = msg

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def handle_update_images(self, request, response):
        if not self._has_latest_data():
            response.success = False
            response.message = "Failed to receive color image, depth image, or camera info."
            return response

        self.cached_color_image = copy.deepcopy(self.latest_color_image)
        self.cached_depth_image = copy.deepcopy(self.latest_depth_image)
        self.cached_camera_info = copy.deepcopy(self.latest_camera_info)

        response.success = True
        response.message = "Updated color image, depth image, and camera info."
        return response

    def handle_get_color_image(self, request, response):
        if self.cached_color_image is None:
            response.success = False
            response.message = "No cached color image. Call update_images first."
            return response

        response.success = True
        response.message = "OK"
        response.image = self.cached_color_image
        return response

    def handle_get_depth_image(self, request, response):
        if self.cached_depth_image is None:
            response.success = False
            response.message = "No cached depth image. Call update_images first."
            return response

        response.success = True
        response.message = "OK"
        response.image = self.cached_depth_image
        return response

    def handle_get_camera_info(self, request, response):
        if self.cached_camera_info is None:
            response.success = False
            response.message = "No cached camera info. Call update_images first."
            return response

        response.success = True
        response.message = "OK"
        response.camera_info = self.cached_camera_info
        return response

    def _has_latest_data(self):
        return (
            self.latest_color_image is not None
            and self.latest_depth_image is not None
            and self.latest_camera_info is not None
        )


def main(args=None):
    rclpy.init(args=args)
    node = RealsenseServiceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
