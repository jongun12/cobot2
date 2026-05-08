import os
import time
import threading
import numpy as np

import rclpy
from rclpy.executors import SingleThreadedExecutor

import DR_init

from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

from ament_index_python.packages import get_package_share_directory

from pick_and_place_text.onrobot import RG
from od_msg.srv import SrvCameraInfo, SrvDetections, SrvImage
from std_srvs.srv import Trigger

VELOCITY, ACC = 60, 60
JReady = [0,0,90,0,90,0]
SERVICE_TIMEOUT_SEC = 10.0
DETECTION_CONFIDENCE_THRESHOLD = 0.5
DETECTION_IOU_THRESHOLD = 0.5

# =====================================================
# Robot Setting
# =====================================================

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# =====================================================
# Gripper
# =====================================================

GRIPPER_NAME = "rg2"

TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"

gripper = RG(
    GRIPPER_NAME,
    TOOLCHANGER_IP,
    TOOLCHANGER_PORT
)

# =====================================================
# Calibration Matrix Path
# =====================================================

package_path = get_package_share_directory(
    "pick_and_place_text"
)

T_PATH = os.path.join(
    package_path,
    "resource",
    "T_gripper2camera.npy"
)

# =====================================================
# Global Variables
# =====================================================

target_pose = None
is_busy = False
latest_detections = []
bridge = CvBridge()


def wait_for_future(future, timeout_sec):

    done_event = threading.Event()
    future.add_done_callback(lambda _: done_event.set())
    return done_event.wait(timeout=timeout_sec)


def parse_detection_response(response):

    boxes = list(response.boxes)
    class_ids = list(response.class_ids)
    scores = list(response.scores)

    if len(boxes) % 4 != 0:
        print(
            f"⚠️ Invalid boxes length: {len(boxes)}. Expected a multiple of 4."
        )

    detection_count = min(len(boxes) // 4, len(class_ids), len(scores))
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


def calculate_detection_center(detections):

    if not detections:
        return None

    sum_x1 = 0.0
    sum_y1 = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        sum_x1 += float(x1)
        sum_y1 += float(y1)
        sum_x2 += float(x2)
        sum_y2 += float(y2)

    count = float(len(detections))
    center_x = (sum_x1 + sum_x2) / (2.0 * count)
    center_y = (sum_y1 + sum_y2) / (2.0 * count)
    return [center_x, center_y]


def wait_for_service_response(comm_node, client, request, timeout_sec, service_name):

    while not client.wait_for_service(timeout_sec=1.0):
        comm_node.get_logger().info(f"Waiting for {service_name} service...")

    future = client.call_async(request)

    if not wait_for_future(future, timeout_sec):
        future.cancel()
        comm_node.get_logger().error(
            f"Timed out waiting for {service_name} after {timeout_sec:.1f}s."
        )
        return None

    if future.result() is None:
        comm_node.get_logger().error(f"Failed to call {service_name} service.")
        return None

    return future.result()


def request_depth_image(comm_node, update_images_client, depth_image_client):

    update_response = wait_for_service_response(
        comm_node,
        update_images_client,
        Trigger.Request(),
        SERVICE_TIMEOUT_SEC,
        "update_images",
    )
    if update_response is None or not update_response.success:
        if update_response is not None:
            comm_node.get_logger().warn(update_response.message)
        return None

    depth_response = wait_for_service_response(
        comm_node,
        depth_image_client,
        SrvImage.Request(),
        SERVICE_TIMEOUT_SEC,
        "get_depth_image",
    )
    if depth_response is None or not depth_response.success:
        if depth_response is not None:
            comm_node.get_logger().warn(depth_response.message)
        return None

    return bridge.imgmsg_to_cv2(depth_response.image, desired_encoding="passthrough")


def request_camera_info(comm_node, camera_info_client):

    response = wait_for_service_response(
        comm_node,
        camera_info_client,
        SrvCameraInfo.Request(),
        SERVICE_TIMEOUT_SEC,
        "get_camera_info",
    )
    if response is None or not response.success:
        if response is not None:
            comm_node.get_logger().warn(response.message)
        return None

    return response.camera_info


def get_box_center(box):

    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_depth_value(depth_image, x, y):

    height, width = depth_image.shape[:2]
    if x < 0 or y < 0 or x >= width or y >= height:
        return None
    return float(depth_image[y, x])


def pixel_to_camera_coords(x, y, z, camera_info):

    fx = camera_info.k[0]
    fy = camera_info.k[4]
    ppx = camera_info.k[2]
    ppy = camera_info.k[5]
    return [
        float((x - ppx) * z / fx),
        float((y - ppy) * z / fy),
        float(z),
    ]


def calculate_target_base_xyz(comm_node, detections, update_images_client, depth_image_client, camera_info_client):

    if not detections:
        return None, None, None

    center_x, center_y = calculate_detection_center(detections)
    center_x = int(round(center_x))
    center_y = int(round(center_y))

    depth_image = request_depth_image(comm_node, update_images_client, depth_image_client)
    if depth_image is None:
        return (center_x, center_y), None, None

    camera_info = request_camera_info(comm_node, camera_info_client)
    if camera_info is None:
        return (center_x, center_y), None, None

    depth = get_depth_value(depth_image, center_x, center_y)
    if depth is None or not np.isfinite(depth) or depth <= 0:
        comm_node.get_logger().warn(
            f"Invalid depth at center ({center_x}, {center_y}): {depth}"
        )
        return (center_x, center_y), None, None

    camera_xyz = pixel_to_camera_coords(center_x, center_y, depth, camera_info)

    from DSR_ROBOT2 import get_current_posx

    robot_posx = get_current_posx()[0]
    base_xyz = transform_to_base(camera_xyz, robot_posx)
    return (center_x, center_y), camera_xyz, base_xyz


def request_all_object_detections(comm_node, detect_client):

    while not detect_client.wait_for_service(timeout_sec=1.0):
        comm_node.get_logger().info("Waiting for detect_all_objects service...")

    request = SrvDetections.Request()
    request.confidence_threshold = float(DETECTION_CONFIDENCE_THRESHOLD)
    request.iou_threshold = float(DETECTION_IOU_THRESHOLD)

    comm_node.get_logger().info("Calling detect_all_objects service...")
    future = detect_client.call_async(request)

    if not wait_for_future(future, SERVICE_TIMEOUT_SEC):
        future.cancel()
        comm_node.get_logger().error(
            f"Timed out waiting for detect_all_objects after {SERVICE_TIMEOUT_SEC:.1f}s."
        )
        return []

    if future.result() is None:
        comm_node.get_logger().error("Failed to call detect_all_objects service.")
        return []

    detections = parse_detection_response(future.result())
    comm_node.get_logger().info(f"Received {len(detections)} detections.")
    return detections


def fetch_and_print_detections(comm_node, detect_client):

    global latest_detections

    latest_detections = request_all_object_detections(comm_node, detect_client)

    if not latest_detections:
        print("\n📦 No bounding boxes detected.")
        return

    center, camera_xyz, base_xyz = calculate_target_base_xyz(
        comm_node,
        latest_detections,
        comm_node.update_images_client,
        comm_node.depth_image_client,
        comm_node.camera_info_client,
    )

    print("\n==============================")
    print("📦 DETECTED BOUNDING BOXES")
    for detection in latest_detections:
        print(
            f"box={detection['box']}, class_id={detection['class_id']}, score={detection['score']:.3f}"
        )
    if center is not None:
        print(f"🎯 CENTER PIXEL = [{center[0]}, {center[1]}]")
    if camera_xyz is not None:
        print(
            f"📷 CAMERA XYZ = [{camera_xyz[0]:.2f}, {camera_xyz[1]:.2f}, {camera_xyz[2]:.2f}]"
        )
    if base_xyz is not None:
        print(
            f"🤖 BASE XYZ = [{base_xyz[0]:.2f}, {base_xyz[1]:.2f}, {base_xyz[2]:.2f}]"
        )
        try:
            perform_movec(base_xyz)
        except Exception as e:
            print("\n❌ ROBOT MOVE ERROR")
            print(e)
    print("==============================")

# =====================================================
# Robot Initialize
# =====================================================

def initialize_robot():

    from DSR_ROBOT2 import (
        set_tool,
        set_tcp
    )

    set_tool("Tool Weight")
    set_tcp("GripperDA")

# =====================================================
# Pose Matrix
# =====================================================

def get_robot_pose_matrix(
    x,
    y,
    z,
    rx,
    ry,
    rz
):

    """
    Doosan posx -> homogeneous transform
    """

    R = Rotation.from_euler(
        "ZYZ",
        [rx, ry, rz],
        degrees=True
    ).as_matrix()

    T = np.eye(4)

    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T

# =====================================================
# Camera -> Base Transform
# =====================================================

def transform_to_base(
    camera_coords,
    robot_pos
):

    """
    camera -> base transform
    """

    # -----------------------------------------
    # Load calibration matrix
    # -----------------------------------------

    T_gripper2camera = np.load(T_PATH)

    # -----------------------------------------
    # Current robot TCP pose
    # -----------------------------------------

    x, y, z, rx, ry, rz = robot_pos

    T_base2gripper = get_robot_pose_matrix(
        x,
        y,
        z,
        rx,
        ry,
        rz
    )

    # -----------------------------------------
    # base -> camera
    # -----------------------------------------

    T_base2camera = (
        T_base2gripper @
        T_gripper2camera
    )

    # -----------------------------------------
    # Camera point
    # -----------------------------------------

    cam_point = np.array([
        camera_coords[0],
        camera_coords[1],
        camera_coords[2],
        1.0
    ])

    # -----------------------------------------
    # Transform
    # -----------------------------------------

    base_point = (
        T_base2camera @
        cam_point
    )

    return base_point[:3]

# =====================================================
# YOLO Callback
# =====================================================

def object_callback(msg):

    global target_pose

    target_pose = [
        msg.x,
        msg.y,
        msg.z
    ]

    print("\n==============================")
    print("📦 YOLO CAMERA COORD")
    print(target_pose)
    print("==============================")

# =====================================================
# Pick + movec
# =====================================================

def perform_movec(base_xyz):

    from DSR_ROBOT2 import (
        movej,
        movel,
        movec,
        posx,
        get_current_posx,
        DR_BASE
    )

    bx, by, bz = base_xyz
    pos = get_current_posx()[0]

    print("\n==============================")
    print("🌍 BASE XYZ")
    print(base_xyz)
    print("==============================")

    # =================================================
    # Safety
    # =================================================

    bz = max(bz, 5.0)

    safe_z = bz + 120

    # =================================================
    # Orientation
    # =================================================
    safe_z = 300

    RX = pos[3]
    RY = pos[4]
    RZ = pos[5]

    # =================================================
    # Positions
    # =================================================

    # Above target
    P1 = posx(
        bx,
        by,
        safe_z,
        RX,
        RY,
        RZ
    )

    # Pick point
    P2 = posx(
        bx,
        by,
        100,
        RX,
        RY,
        RZ
    )

    # movec midpoint
    P3 = posx(
        bx + 100,
        by,
        100,
        RX,
        RY,
        RZ
    )

    # movec endpoint
    P4 = posx(
        bx,
        by + 100,
        100,
        RX,
        RY,
        RZ
    )

    P5 = posx(
        bx - 100,
        by,
        100,
        RX,
        RY,
        RZ
    )

    # movec endpoint
    P6 = posx(
        bx,
        by - 100,
        100,
        RX,
        RY,
        RZ
    )

    print("\n🚀 START TASK")

    # =================================================
    # Move Above
    # =================================================

    movel(
        P1,
        vel=50,
        acc=50
    )
    gripper.close_gripper()

    time.sleep(1.0)

    print("🤏 GRIPPER CLOSED")
    # =================================================
    # Move Down
    # =================================================

    movel(
        P2,
        vel=20,
        acc=20
    )

    print("📍 ARRIVED TARGET")

    # =================================================
    # Grip Close
    # =================================================

    


    # =================================================
    # Current Position Debug
    # =================================================

    current = get_current_posx()[0]

    print("\nCURRENT POS:")
    print(current)

    print("\nP3:")
    print(P3)

    print("\nP4:")
    print(P4)

    # =================================================
    # movec
    # =================================================

    print("\n🌀 START MOVEC")

    movec(
        P3,
        P4,
        vel=80,
        acc=80,
        ref=DR_BASE
    )

    print("✅ MOVEC DONE")

    print("\n🌀 START MOVEC")

    movec(
        P5,
        P6,
        vel=80,
        acc=80,
        ref=DR_BASE
    )

    print("✅ MOVEC DONE")

    # =================================================
    # Return Home
    # =================================================

    movej(
        JReady,
        vel=VELOCITY,
        acc=ACC
    )

    print("🏠 RETURN HOME")

    # =================================================
    # Open Gripper
    # =================================================

    # gripper.open_gripper()

    print("✅ TASK COMPLETE")

# =====================================================
# Main Task Loop
# =====================================================

def task_loop():

    global target_pose
    global is_busy

    from DSR_ROBOT2 import (
        movej,
        get_current_posx
    )

    # =================================================
    # Initial Position
    # =================================================

    movej(
        JReady,
        vel=VELOCITY,
        acc=ACC
    )

    # gripper.open_gripper()

    # =================================================
    # Loop
    # =================================================

    while rclpy.ok():

        if target_pose is not None and not is_busy:

            is_busy = True

            try:

                # -----------------------------------------
                # Camera Coord
                # -----------------------------------------

                cam_coords = target_pose

                print("\n📸 CAMERA XYZ:")
                print(cam_coords)

                # -----------------------------------------
                # Current Robot Pose
                # -----------------------------------------

                robot_posx = get_current_posx()[0]

                print("\n🤖 CURRENT ROBOT POSX:")
                print(robot_posx)

                # -----------------------------------------
                # Camera -> Base
                # -----------------------------------------

                base_xyz = transform_to_base(
                    cam_coords,
                    robot_posx
                )

                print("\n🌍 TRANSFORMED BASE XYZ:")
                print(base_xyz)

                # -----------------------------------------
                # Execute Task
                # -----------------------------------------

                perform_movec(base_xyz)

            except Exception as e:

                print("\n❌ ERROR OCCURRED")
                print(e)

            # -----------------------------------------
            # Reset
            # -----------------------------------------

            target_pose = None
            is_busy = False

            # gripper.open_gripper()

        time.sleep(0.1)

# =====================================================
# Main
# =====================================================

def main(args=None):

    rclpy.init(args=args)

    # =================================================
    # Robot Node
    # =================================================

    robot_node = rclpy.create_node(
        "robot_control_node",
        namespace=ROBOT_ID
    )

    DR_init.__dsr__node = robot_node

    initialize_robot()

    # =================================================
    # Communication Node
    # =================================================

    comm_node = rclpy.create_node(
        "communication_node",
        namespace=ROBOT_ID
    )

    detect_client = comm_node.create_client(
        SrvDetections,
        "/detect_all_objects"
    )
    comm_node.update_images_client = comm_node.create_client(
        Trigger,
        "/update_images"
    )
    comm_node.depth_image_client = comm_node.create_client(
        SrvImage,
        "/get_depth_image"
    )
    comm_node.camera_info_client = comm_node.create_client(
        SrvCameraInfo,
        "/get_camera_info"
    )


    # =================================================
    # Executor
    # =================================================

    executor = SingleThreadedExecutor()

    executor.add_node(comm_node)

    threading.Thread(
        target=fetch_and_print_detections,
        args=(comm_node, detect_client),
        daemon=True
    ).start()

    # =================================================
    # Task Thread
    # =================================================

    threading.Thread(
        target=task_loop,
        daemon=True
    ).start()

    print("\n====================================")
    print("🚀 YOLO → BASE → PICK → MOVEC START")
    print("====================================")

    try:

        executor.spin()

    finally:

        robot_node.destroy_node()

        comm_node.destroy_node()

        rclpy.shutdown()

# =====================================================

if __name__ == "__main__":

    main()