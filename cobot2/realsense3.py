from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


COLOR_IMAGE_TOPIC = "/camera/camera/color/image_raw"
DEPTH_IMAGE_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"


class RealsenseFrameNode(Node):
    def __init__(self, node_name="realsense_frame_node"):
        super().__init__(node_name)
        self.bridge = CvBridge()

        self.color_msg = None
        self.depth_msg = None
        self.camera_info = None

        self.create_subscription(
            Image,
            COLOR_IMAGE_TOPIC,
            self.color_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            DEPTH_IMAGE_TOPIC,
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

    def color_callback(self, msg):
        self.color_msg = msg

    def depth_callback(self, msg):
        self.depth_msg = msg

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def has_frames(self):
        return (
            self.color_msg is not None
            and self.depth_msg is not None
            and self.camera_info is not None
        )

    def has_color_frame(self):
        return self.color_msg is not None

    def get_color_image(self):
        if not self.has_color_frame():
            return None

        return self.bridge.imgmsg_to_cv2(
            self.color_msg,
            desired_encoding="bgr8",
        )

    def get_frames(self):
        if not self.has_frames():
            return None, None, None

        color_image = self.bridge.imgmsg_to_cv2(
            self.color_msg,
            desired_encoding="bgr8",
        )
        depth_image = self.bridge.imgmsg_to_cv2(
            self.depth_msg,
            desired_encoding="passthrough",
        )
        return color_image, depth_image, self.camera_info

    def reset_frames(self):
        self.color_msg = None
        self.depth_msg = None
        self.camera_info = None
