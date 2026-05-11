from cobot2.onrobot import RG
import DR_init
import rclpy
from od_msg.srv import SrvBasePositions
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import threading
import time
from cobot2.test import get_realsense_line_count
from cobot2.test_retain import run_cluster_check_once
from std_msgs.msg import Int32
from std_srvs.srv import Trigger
from dsr_msgs2.srv import MoveStop

SERVICE_TIMEOUT_SEC = 15.0
TRASH_FULL_CHECK_PERIOD_SEC = 2.0
EMERGENCY_STOP_CHECK_PERIOD_SEC = 0.1
VOICE_PAUSE_CHECK_PERIOD_SEC = 0.1
PICK_Z_MIN = 50.0
MIN_GRIPPED_WIDTH_MM = 15.0

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
BUCKET_POS = [319.4, -211.9, 450.7, 155.1, 179.8, 155.4]
P0 = [3.31, 0.1, 76.66, 0, 102.75, 7.79]

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"
gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)

class RobotMoveNode(Node):
    def __init__(self):
        super().__init__("robot_move3")
        self.voice_callback_group = ReentrantCallbackGroup()
        self.base_positions_client = self.create_client(
            SrvBasePositions,
            "get_base_positions",
        )
        self.center_base_positions_client = self.create_client(
            SrvBasePositions,
            "get_center_base_positions",
        )
        self.center_of_center_client = self.create_client(
            SrvBasePositions,
            "center_of_center_points",
        )
        self.db_publisher = self.create_publisher(Int32, 'trash_count', 10)
        self.task_complete_publisher = self.create_publisher(Int32, 'task_complete', 10)
        self.flag_client = self.create_client(
            Trigger,
            'is_trash_full',
        )
        self.move_stop_client = self.create_client(
            MoveStop,
            f'/{ROBOT_ID}/motion/move_stop',
            callback_group=self.voice_callback_group,
        )
        self.start_requested = False
        self.emergency_stopped = False
        self.voice_paused = False
        self.voice_target_class_ids = "all"
        self.voice_disposal_class_id = None
        self.voice_pause_interrupted_motion = False
        self.awaiting_disposal_command = False
        self.last_disposal_class_id = None
        self.restart_scan_requested = False
        self.start_subscription = self.create_subscription(
            Int32,
            'start_condition',
            self.start_condition_callback,
            10,
        )
        self.emergency_stop_subscription = self.create_subscription(
            Int32,
            'emergency_stop',
            self.emergency_stop_callback,
            10,
        )
        self.declare_parameter("voice_command_topic", "voice_command")
        voice_command_topic = (
            self.get_parameter("voice_command_topic").get_parameter_value().string_value
        )
        self.voice_command_subscription = self.create_subscription(
            Int32,
            voice_command_topic,
            self.voice_command_callback,
            10,
            callback_group=self.voice_callback_group,
        )
        self.get_logger().info(
            f"Listening for voice commands on '{voice_command_topic}'."
        )

        gripper.open_gripper()  # 그리퍼 열기
        self.wait_for_gripper_motion()
        self.safe_movej(P0, vel=VELOCITY, acc=ACC) # 초기 위치로 이동

    def start_condition_callback(self, msg):
        if msg.data == 1:
            self.start_requested = True
            self.get_logger().info("Received start_condition=1.")

    def wait_for_start_condition(self):
        self.get_logger().info("Waiting for start_condition=1...")
        while rclpy.ok() and not self.start_requested:
            time.sleep(0.1)

    def emergency_stop_callback(self, msg):
        if msg.data == 1:
            if not self.emergency_stopped:
                self.get_logger().error("Emergency stop requested.")
            self.emergency_stopped = True
            self.request_move_stop()
        else:
            if self.emergency_stopped:
                self.get_logger().info("Emergency stop released.")
            self.emergency_stopped = False

    def voice_command_callback(self, msg):
        command = int(msg.data)
        self.get_logger().warn(f"Received voice_command={command}.")
        if command == 0:
            if not self.voice_paused:
                self.get_logger().warn("Voice command pause requested.")
            self.voice_paused = True
            self.voice_pause_interrupted_motion = True
            self.request_move_stop()
        elif command == 1:
            if self.voice_paused:
                self.get_logger().info(
                    "Voice command resume requested. Using original trash bin."
                )
            self.voice_disposal_class_id = None
            self.voice_paused = False
        elif 2 <= command <= 7:
            class_id = command - 2
            if self.voice_paused or self.awaiting_disposal_command:
                self.voice_disposal_class_id = class_id
                self.voice_paused = False
                self.awaiting_disposal_command = False
                self.restart_scan_requested = False
                self.get_logger().warn(
                    "Voice command selected trash bin "
                    f"class_id={class_id}. Resuming disposal."
                )
            else:
                self.voice_target_class_ids = [class_id]
                self.restart_scan_requested = True
                self.voice_paused = False
                self.get_logger().warn(
                    f"Voice command selected class_id={class_id}. Restarting scan."
                )
                self.request_move_stop()
        else:
            self.get_logger().warn(f"Ignoring unknown voice command: {command}")

    def request_move_stop(self):
        if not self.move_stop_client.service_is_ready():
            self.move_stop_client.wait_for_service(timeout_sec=0.2)
        if not self.move_stop_client.service_is_ready():
            self.get_logger().error("move_stop service is not available.")
            return

        request = MoveStop.Request()
        request.stop_mode = 0
        self.move_stop_client.call_async(request)

    def wait_while_emergency_stopped(self):
        while rclpy.ok() and self.emergency_stopped:
            self.get_logger().warn(
                "Emergency stop is active. Waiting for emergency_stop=0..."
            )
            time.sleep(EMERGENCY_STOP_CHECK_PERIOD_SEC)

    def wait_while_voice_paused(self):
        while rclpy.ok() and self.voice_paused:
            self.get_logger().warn(
                "Voice pause is active. Waiting for voice command 1 or 2~7..."
            )
            time.sleep(VOICE_PAUSE_CHECK_PERIOD_SEC)

    def wait_until_motion_allowed(self):
        self.wait_while_emergency_stopped()
        self.wait_while_voice_paused()

    def should_restart_scan(self):
        return self.restart_scan_requested

    def consume_restart_scan_request(self):
        if not self.restart_scan_requested:
            return False

        self.restart_scan_requested = False
        self.get_logger().info("Restarting scan from the beginning.")
        return True

    def safe_movel(self, *args, **kwargs):
        from DSR_ROBOT2 import movel

        self.wait_until_motion_allowed()
        if self.should_restart_scan():
            return None
        result = movel(*args, **kwargs)
        self.wait_until_motion_allowed()
        return result

    def safe_movej(self, *args, **kwargs):
        from DSR_ROBOT2 import movej

        self.wait_until_motion_allowed()
        if self.should_restart_scan():
            return None
        result = movej(*args, **kwargs)
        self.wait_until_motion_allowed()
        return result

    def wait_for_future(self, future, timeout_sec):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return future.done()

    def publish_task_complete(self):
        msg = Int32()
        msg.data = 1
        self.task_complete_publisher.publish(msg)
        self.get_logger().info("Published task_complete=1.")

    def request_base_positions(self):
        while not self.base_positions_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for get_base_positions service...")

        self.get_logger().info("Calling get_base_positions service...")
        future = self.base_positions_client.call_async(SrvBasePositions.Request())

        if not self.wait_for_future(future, SERVICE_TIMEOUT_SEC):
            future.cancel()
            self.get_logger().error(
                f"Timed out waiting for get_base_positions after {SERVICE_TIMEOUT_SEC:.1f}s."
            )
            return {}

        if future.result() is None:
            self.get_logger().error("Failed to call get_base_positions service.")
            return {}

        response = future.result()
        if not response.success:
            self.get_logger().warn(response.message)
            return {}

        positions = self._parse_base_positions_response(response)
        positions_by_class = self._group_positions_by_class_id(positions)
        self.get_logger().info(response.message)
        for class_id, class_positions in positions_by_class.items():
            self.get_logger().info(
                f"class_id={class_id}, count={len(class_positions)}"
            )
            for position in class_positions:
                self.get_logger().info(
                    "  pose=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]"
                    % (
                        position["x"],
                        position["y"],
                        position["z"],
                        position["rx"],
                        position["ry"],
                        position["rz"],
                    )
                )

        return positions_by_class

    def request_center_of_centers_xyz(self):
        while not self.center_of_center_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for center_of_center_points service...")

        self.get_logger().info("Calling center_of_center_points service...")
        future = self.center_of_center_client.call_async(SrvBasePositions.Request())

        if not self.wait_for_future(future, SERVICE_TIMEOUT_SEC):
            future.cancel()
            self.get_logger().error(
                "Timed out waiting for center_of_center_points "
                f"after {SERVICE_TIMEOUT_SEC:.1f}s."
            )
            return None

        if future.result() is None:
            self.get_logger().error("Failed to call center_of_center_points service.")
            return None

        response = future.result()
        if not response.success:
            self.get_logger().warn(response.message)
            return None

        positions = self._parse_base_positions_response(response)
        if not positions:
            self.get_logger().warn("center_of_center_points returned no positions.")
            return None

        position = positions[0]
        center_xyz = [position["x"], position["y"], position["z"]]
        self.get_logger().info(
            "Center of centers xyz=[%.3f, %.3f, %.3f]"
            % (center_xyz[0], center_xyz[1], center_xyz[2])
        )
        return center_xyz

    def prompt_target_class_ids(self, positions_by_class):
        if not positions_by_class:
            self.get_logger().warn("No detected positions are available.")
            return None

        print("====================================")
        print("Detected class IDs:")
        for class_id, class_positions in sorted(positions_by_class.items()):
            print(f"  {class_id} : {len(class_positions)} object(s)")

        while True:
            user_input = input(
                "Pick class_id(s) to move (ex: 1 or 1,2,3 / all / q): "
            ).strip()
            if user_input.lower() == "q":
                self.get_logger().info("Quit the program...")
                return None

            if user_input.lower() == "all":
                return sorted(positions_by_class.keys())

            class_id_texts = user_input.replace(",", " ").split()
            if not class_id_texts:
                print("Please enter at least one class_id.")
                continue

            try:
                class_ids = [int(class_id_text) for class_id_text in class_id_texts]
            except ValueError:
                print("Please enter numeric class_id values.")
                continue

            missing_class_ids = [
                class_id for class_id in class_ids if class_id not in positions_by_class
            ]
            if missing_class_ids:
                print(f"class_id(s) {missing_class_ids} were not detected.")
                continue

            return class_ids

    def prompt_target_class_ids_before_scan(self):
        while True:
            user_input = input(
                "Pick class_id(s) to move (ex: 1 or 1,2,3 / all / q): "
            ).strip()
            if user_input.lower() == "q":
                self.get_logger().info("Quit the program...")
                return None

            if user_input.lower() == "all":
                return "all"

            class_id_texts = user_input.replace(",", " ").split()
            if not class_id_texts:
                print("Please enter at least one class_id.")
                continue

            try:
                return [int(class_id_text) for class_id_text in class_id_texts]
            except ValueError:
                print("Please enter numeric class_id values.")

    def pick_and_place_class(self, class_ids, positions_by_class):
        target_class_ids = [class_ids] if isinstance(class_ids, int) else class_ids
        target_positions = []
        for class_id in target_class_ids:
            target_positions.extend(positions_by_class.get(class_id, []))

        target_positions = sorted(
            target_positions,
            key=lambda position: (-position["y"], position["x"]),
        )
        if not target_positions:
            self.get_logger().warn(
                f"No positions found for class_id(s)={target_class_ids}."
            )
            return

        self.get_logger().info(
            "Picking %d object(s) with class_id(s)=%s sorted by +y."
            % (len(target_positions), target_class_ids)
        )
        for index, position in enumerate(target_positions, start=1):
            self.wait_until_motion_allowed()
            if self.should_restart_scan():
                return False

            target_pos = self._position_to_pose(position)
            self.get_logger().info(
                "Picking %d/%d: class_id=%d, pose=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]"
                % (
                    index,
                    len(target_positions),
                    position["class_id"],
                    target_pos[0],
                    target_pos[1],
                    target_pos[2],
                    target_pos[3],
                    target_pos[4],
                    target_pos[5],
                )
            )
            if position["class_id"] != 0:
                # 물체에 가까이 이동 ~ 분리수거 통 위
                pick_success = self.pick_and_place_target(
                    position["class_id"],
                    target_pos,
                )
                if pick_success:
                    trash_count_msg = Int32()
                    trash_count_msg.data = (
                        self.last_disposal_class_id
                        if self.last_disposal_class_id is not None
                        else position["class_id"]
                    )
                    self.db_publisher.publish(trash_count_msg)
                    self.wait_until_trash_not_full()
                    if self.should_restart_scan():
                        return False
                else:
                    self.get_logger().warn(
                        "Pick failed. Skipping trash_count publish."
                    )
            else:
                self.side_pick_and_place_target(position["class_id"], target_pos)

            if self.should_restart_scan():
                return False

        return True

    def request_trash_full_flag(self):
        while not self.flag_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for is_trash_full service...")

        future = self.flag_client.call_async(Trigger.Request())

        if not self.wait_for_future(future, SERVICE_TIMEOUT_SEC):
            future.cancel()
            self.get_logger().error(
                f"Timed out waiting for is_trash_full service "
                f"after {SERVICE_TIMEOUT_SEC:.1f}s."
            )
            return None

        if future.result() is None:
            self.get_logger().error("Failed to call is_trash_full service.")
            return None

        response = future.result()
        if not response.success:
            self.get_logger().warn(response.message)
            return None

        return response.message

    def wait_until_trash_not_full(self):
        while True:
            self.wait_until_motion_allowed()
            if self.should_restart_scan():
                return

            flag = self.request_trash_full_flag()
            if flag != '1':
                return
            self.safe_movej(P0, vel=VELOCITY, acc=ACC) # 초기 위치로 이동
            if self.should_restart_scan():
                return
            self.get_logger().warn(
                "Trash bin is full. Waiting until Firebase flag becomes 0..."
            )
            time.sleep(TRASH_FULL_CHECK_PERIOD_SEC)

    def wait_for_gripper_motion(self):
        while gripper.get_status()[0]:
            self.wait_until_motion_allowed()
            if self.should_restart_scan():
                return False
            time.sleep(0.1)
        return True

    def close_gripper_and_wait(self, force_val=100):
        gripper.close_gripper(force_val=force_val)
        return self.wait_for_gripper_motion()

    def is_object_gripped(self):
        status = gripper.get_status()
        width = gripper.get_width()
        if self._is_gripper_safety_active(status):
            self.get_logger().error(
                f"Gripper safety is active. Stopping robot. status={status}"
            )
            self.emergency_stopped = True
            self.request_move_stop()
            return False

        if width <= MIN_GRIPPED_WIDTH_MM:
            self.get_logger().warn(
                f"Grip failed. width={width:.1f}mm <= {MIN_GRIPPED_WIDTH_MM:.1f}mm"
            )
            return False

        if status[1]:
            self.get_logger().info(f"Grip detected. width={width:.1f}mm")
            return True

        self.get_logger().warn(f"Grip failed. width={width:.1f}mm")
        return False

    def _trash_bin_posj_for_class_id(self, class_id):
        if class_id == 1:  # plastic bottle
            return [-77.39, 13.9, 77.34, 0.17, 88.33, -76.44]
        if class_id == 2:  # label o
            return [-77.39, 13.9, 77.34, 0.17, 88.33, -76.44]
        if class_id == 3:  # plastic bottle
            return [-77.39, 13.9, 77.34, 0.17, 88.33, -76.44]
        if class_id == 4:  # can
            return [-47.38, 32.53, 49.7, -0.09, 97.36, -46.6]
        if class_id == 5:  # box
            return [-34.81, 44.83, 41.64, 11.51, 60.07, -33.78]
        return [-77.39, 13.9, 77.34, 0.17, 88.33, -76.44]

    def _consume_disposal_class_id(self, original_class_id):
        if self.voice_disposal_class_id is None:
            disposal_class_id = original_class_id
        else:
            disposal_class_id = self.voice_disposal_class_id

        self.voice_disposal_class_id = None
        self.last_disposal_class_id = disposal_class_id
        return disposal_class_id

    def move_to_trash_bin(self, original_class_id):
        from DSR_ROBOT2 import movej

        while rclpy.ok():
            self.awaiting_disposal_command = True
            self.wait_until_motion_allowed()
            self.awaiting_disposal_command = False
            if self.should_restart_scan():
                return False

            self.voice_pause_interrupted_motion = False
            disposal_class_id = self._consume_disposal_class_id(original_class_id)
            trash_bin_posj = self._trash_bin_posj_for_class_id(disposal_class_id)
            self.get_logger().info(
                f"Moving to trash bin for class_id={disposal_class_id}."
            )

            movej(trash_bin_posj, vel=VELOCITY, acc=ACC)

            if self.voice_pause_interrupted_motion:
                self.get_logger().warn(
                    "Trash bin move was interrupted by voice pause. "
                    "Waiting for the next disposal command."
                )
                self.awaiting_disposal_command = True
                continue

            self.awaiting_disposal_command = True
            self.wait_until_motion_allowed()
            self.awaiting_disposal_command = False
            if self.should_restart_scan():
                return False

            if self.voice_pause_interrupted_motion:
                self.get_logger().warn(
                    "Trash bin move was paused before release. "
                    "Rechecking disposal command."
                )
                self.awaiting_disposal_command = True
                continue

            return True

        self.awaiting_disposal_command = False
        return False

    def move_class_1_or_2_target(self, class_id):
        pat1 = [7.92, -16.78, 110.3, 82.05, -2.01, 8.81]
        pat2 = [17.65, 8.76, 81.7, 89.55, -22.71, 4.27]
        pat3 = [-4.36, 9.67, 84.36, 83.8, 29.81, 2.8]
        check_pos = [9.84, -2.13, 100.49, 77.93, -4.43, 9.73]

        self.wait_until_motion_allowed()
        if self.should_restart_scan():
            return False

        self.get_logger().info(
            f"Moving with class_id={class_id} using class 1/2 custom motion."
        )
        self.safe_movej(pat1, vel=VELOCITY, acc=ACC)
        self.safe_movej(pat2, vel=VELOCITY, acc=ACC)
        self.safe_movej(pat3, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()  # 그리퍼 열기
        if not self.wait_for_gripper_motion():
            return False
        self.safe_movej(check_pos, vel=VELOCITY, acc=ACC)
        try:
            line_count = get_realsense_line_count()
            self.get_logger().info(f"Detected RealSense line count: {line_count}")
        except Exception as e:
            line_count = 0
            self.get_logger().error(f"Failed to detect RealSense line count: {e}")

        if line_count == 0:
            self.get_logger().info(
                "line_count is 0. Moving to original trash bin."
            )
            self.safe_movej(pat3, vel=VELOCITY, acc=ACC)
            gripper.close_gripper(force_val=100)  # 그리퍼 닫기
            if not self.wait_for_gripper_motion():
                return False
            self.safe_movej(pat2, vel=VELOCITY, acc=ACC)
            self.safe_movej(pat1, vel=VELOCITY, acc=ACC)
            return self.move_to_trash_bin(class_id)
        else:
            gripper.close_gripper(force_val=100)  # 그리퍼 닫기
            if not self.wait_for_gripper_motion():
                return False
            testpos1 = [24.12, 1.99, 96.4, 88.28, -18.5, 1.37]
            self.safe_movej(testpos1, vel=VELOCITY, acc=ACC)
            testpos1 = [20.61, 10.33, 87.06, 91.06, -15.02, -2.04]
            self.safe_movej(testpos1, vel=VELOCITY, acc=ACC)
            testpos1 = [20.61, 10.33, 87.06, 91.06, -19.31, -2.04]
            self.safe_movej(testpos1, vel=VELOCITY, acc=ACC)
            testpos1 = [20.61, 10.33, 87.06, 91.06, -6.8, -2.04]
            self.safe_movej(testpos1, vel=VELOCITY, acc=ACC)
            testpos1 = [20.61, 10.33, 87.06, 91.06, -19.31, -2.04]
            self.safe_movej(testpos1, vel=VELOCITY, acc=ACC)
            testpos1 = [20.61, 10.33, 87.06, 91.06, -6.8, -2.04]
            self.safe_movej(testpos1, vel=VELOCITY, acc=ACC)
            testpos1 = [20.61, 10.33, 87.05, 91.06, -49.35, -2.04]
            self.safe_movej(testpos1, vel=VELOCITY, acc=ACC)
            

             # 테스트 위치로 이동

        self.last_disposal_class_id = class_id

        self.wait_until_motion_allowed()
        if self.should_restart_scan():
            return False

        return True

    def pick_and_place_target(self, class_id, target_pos):
        from DSR_ROBOT2 import mwait, DR_MV_MOD_REL

        close_picture_pose = list(target_pos)
        close_picture_pose[2] += 40
        close_picture_pose[1] += -85  # 사진 찍는 위치로 이동 (조정 필요)
        print("1. Moving to close picture pose...")
        self.safe_movel(close_picture_pose, vel=VELOCITY, acc=ACC) # 사진 찍는 위치로 이동
        if self.should_restart_scan():
            return False
        print("Taking picture and getting more accurate position...")
        # 사진 찍고 더 정확한 위치 받아오기 + box 정보
        center_position = None
        self.center_base_positions_client.wait_for_service()
        future = self.center_base_positions_client.call_async(SrvBasePositions.Request())

        if not self.wait_for_future(future, SERVICE_TIMEOUT_SEC):
            future.cancel()
            self.get_logger().error(
                "Timed out waiting for get_center_base_positions "
                f"after {SERVICE_TIMEOUT_SEC:.1f}s."
            )
        elif future.result() is None:
            self.get_logger().error("Failed to call get_center_base_positions service.")
        else:
            response = future.result()
            if not response.success:
                self.get_logger().warn(response.message)
            else:
                center_positions = self._parse_base_positions_response(response)
                if center_positions:
                    center_position = center_positions[0]
                    target_pos = self._position_to_pose(center_position)
                    self.get_logger().info(
                        "Updated target pose from center object: (class_id=%d) "
                        "[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]"
                        % (
                            center_position["class_id"],
                            target_pos[0],
                            target_pos[1],
                            target_pos[2],
                            target_pos[3],
                            target_pos[4],
                            target_pos[5],
                        )
                    )
                else:
                    self.get_logger().warn(
                        "get_center_base_positions returned no positions."
                    )

        if center_position is None:
            self.get_logger().warn(
                "Cannot continue pick: center object was not detected."
            )
            return False

        if center_position["class_id"] == 1:  # label x
            target_pos[2] = 80  # 정확한 위치 (조정 필요)
        elif center_position["class_id"] == 2:  # label o
            target_pos[2] = 80  # 정확한 위치 (조정 필요)
        else:
            target_pos[2] += -80  # 정확한 위치 (조정 필요)
        target_pos = self._limit_pick_z_min(target_pos)

        # pick_pos_up = list(target_pos)
        # pick_pos_up[2] += 100  # 대상 위치 위로 이동
        pick_up = list(target_pos)
        pick_up[2] += 100  # 대상 위치 위로 이동
        print("2. Moving to pick position...")
        self.safe_movel(pick_up, vel=VELOCITY, acc=ACC) # 대상 위치 위로 이동
        if self.should_restart_scan():
            return False
        # movel([-10,0,0,0,0,0], vel=VELOCITY, acc=ACC, ref=DR_TOOL)

        print("3. Moving to target position...")
        self.safe_movel([0,0,-100,0,0,0], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL) # 대상 위치로 이동
        if self.should_restart_scan():
            return False
        mwait()
         # 대상 위치로 이동

        if not self.close_gripper_and_wait(force_val=100):
            return False

        print("4. Moving up with the object...")
        pick_up = list(target_pos)
        pick_up[2] += 150  # 대상 위치 위로 이동
        self.safe_movel(pick_up, vel=VELOCITY, acc=ACC) # 대상 위치 위로 이동
        if self.should_restart_scan():
            return False
        if not self.is_object_gripped():
            print("============Grip failed. Moving back up and skipping this object.============")
            gripper.open_gripper()
            self.wait_for_gripper_motion()
            return False

        print("5. Moving to bucket position...")
        if class_id in (1, 2):
            move_success = self.move_class_1_or_2_target(class_id)
        else:
            move_success = self.move_to_trash_bin(class_id)

        if not move_success:
            return False
        gripper.open_gripper()  # 그리퍼 열기
        if not self.wait_for_gripper_motion():
            return False

        print("6. Operation completed.")
        return True
    
    def side_pick_and_place_target(self, class_id, target_pos):
        from DSR_ROBOT2 import posx, DR_MV_MOD_REL

        print("Target object is too close to move above. Performing side pick-and-place.")
        close_picture_pose = list(target_pos)
        # 사진 찍는 위치로 이동 (조정 필요)
        close_picture_pose[1] += 80 # y
        y_temp = close_picture_pose[1]
        close_picture_pose[2] = 190 # z 뎁스 인식이 잘 안되서 어려울 경우 그냥 값을 고정해도 됨
        close_picture_pose[3] = 90
        close_picture_pose[4] = -90
        close_picture_pose[5] = 90
        print("1. Moving to close picture pose...")
        self.safe_movel(close_picture_pose, vel=VELOCITY, acc=ACC) # 사진 찍는 위치로 이동
        if self.should_restart_scan():
            return False

        # 사진 찍고 더 정확한 위치 받아오기 + box 정보
        print("Taking picture and getting more accurate position...")
        center_position = None
        self.center_base_positions_client.wait_for_service()
        future = self.center_base_positions_client.call_async(SrvBasePositions.Request())

        if not self.wait_for_future(future, SERVICE_TIMEOUT_SEC):
            future.cancel()
            self.get_logger().error(
                "Timed out waiting for get_center_base_positions "
                f"after {SERVICE_TIMEOUT_SEC:.1f}s."
            )
        elif future.result() is None:
            self.get_logger().error("Failed to call get_center_base_positions service.")
        else:
            response = future.result()
            if not response.success:
                self.get_logger().warn(response.message)
            else:
                center_positions = self._parse_base_positions_response(response)
                if center_positions:
                    center_position = center_positions[0]
                    target_pos = self._position_to_pose(center_position)
                    self.get_logger().info(
                        "Updated target pose from center object: "
                        "[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]"
                        % (
                            target_pos[0],
                            target_pos[1],
                            target_pos[2],
                            target_pos[3],
                            target_pos[4],
                            target_pos[5],
                        )
                    )
                else:
                    self.get_logger().warn(
                        "get_center_base_positions returned no positions."
                    )

        # box 가로 너비 계산 (어쩌면 픽셀 단위일 수도 있으니 실제 크기로 변환 필요할 수도 있음, cal_positon에서 처리할 수도 있음)
        


        # ==================================================================================
        if center_position is None:
            self.get_logger().warn(
                "Cannot continue side pick: center object was not detected."
            )
            return False

        # 정확한 위치 (조정 필요)
        target_pos[0] = close_picture_pose[0]  # x는 사진 찍는 위치로 고정
        target_pos[1] = y_temp-80  # y 만약에 뎁스 인식이 잘 안되서 어려울 경우 처음 target_pos에서 y 값을 사용할 수도 있음
        target_pos[2] = 180 # z
        target_pos[3] = 90
        target_pos[4] = -90
        target_pos[5] = 90

        pick_pos_side = list(target_pos)
        pick_pos_side[1] += 100  # 대상 위치 옆으로 이동
        
        print("2. Moving to side position...")
        self.safe_movel(pick_pos_side, vel=VELOCITY, acc=ACC) # 대상 위치 옆으로 이동
        if self.should_restart_scan():
            return False
        print("3. Moving to target position...")
        self.safe_movel(target_pos, vel=VELOCITY, acc=ACC) # 대상 위치로 이동
        if self.should_restart_scan():
            return False
        
        print("gripping...")
        if not self.close_gripper_and_wait(force_val=100):
            return False
        
        print("4. Moving side up with the object...")
        self.safe_movel(posx([0,0,100,0,0,0]), vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL) # 대상 위치 위로 이동
        if self.should_restart_scan():
            return False
        if not self.is_object_gripped():
            gripper.open_gripper()
            self.wait_for_gripper_motion()
            return False

        print("5. Moving to bucket position...")
        if not self.move_to_trash_bin(class_id):
            return False

        gripper.open_gripper()  # 그리퍼 열기
        if not self.wait_for_gripper_motion():
            return False

        print("8. Operation completed.")
        return True
        

    def _parse_base_positions_response(self, response):
        boxes = list(response.boxes)
        if len(boxes) % 4 != 0:
            self.get_logger().warn(
                f"Invalid boxes length: {len(boxes)}. Expected a multiple of 4."
            )

        box_count = len(boxes) // 4
        count = min(
            box_count,
            len(response.class_ids),
            len(response.xs),
            len(response.ys),
            len(response.zs),
            len(response.rxs),
            len(response.rys),
            len(response.rzs),
        )
        positions = []

        for index in range(count):
            box_start = index * 4
            positions.append(
                {
                    "box": boxes[box_start : box_start + 4],
                    "class_id": int(response.class_ids[index]),
                    "x": float(response.xs[index]),
                    "y": float(response.ys[index]),
                    "z": float(response.zs[index]),
                    "rx": float(response.rxs[index]),
                    "ry": float(response.rys[index]),
                    "rz": float(response.rzs[index]),
                }
            )

        return positions

    def _group_positions_by_class_id(self, positions):
        positions_by_class = {}
        for position in positions:
            class_id = position["class_id"]
            positions_by_class.setdefault(class_id, []).append(position)
        return positions_by_class

    def _position_to_pose(self, position):
        return [
            position["x"],
            position["y"],
            position["z"],
            position["rx"],
            position["ry"],
            position["rz"],
        ]

    def _limit_pick_z_min(self, pose):
        limited_pose = list(pose)
        if limited_pose[2] <= PICK_Z_MIN:
            self.get_logger().info(
                "Limiting pick z from %.3f to %.3f."
                % (limited_pose[2], PICK_Z_MIN)
            )
            limited_pose[2] = PICK_Z_MIN
        return limited_pose

    def _is_gripper_safety_active(self, status):
        return any(status[2:7])
    


def main(args=None):
    rclpy.init(args=args)
    dsr_node = rclpy.create_node("robot_move3_dsr", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node
    node = RobotMoveNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        from DSR_ROBOT2 import posx
        node.wait_for_start_condition()
        CENTER_POINT = (500.0, -50.0)
        Z0 = 300
        HIGHT = 150

        run_cluster_check_once()

        gripper.open_gripper()  # 그리퍼 열기
        node.wait_for_gripper_motion()
        while True:
            target_class_ids = node.voice_target_class_ids
            detected_object_count = 0
            restart_scan = False
            for y_offset in (HIGHT / 2, -HIGHT / 2):
                node.wait_until_motion_allowed()
                if node.should_restart_scan():
                    restart_scan = True
                    break

                p = posx([CENTER_POINT[0], CENTER_POINT[1] + y_offset, Z0, 0, 180, 0])
                print(p)
                node.safe_movel(p, vel=VELOCITY, acc=ACC)
                if node.should_restart_scan():
                    restart_scan = True
                    break

                positions_by_class = node.request_base_positions()
                if node.should_restart_scan():
                    restart_scan = True
                    break

                detected_object_count += sum(
                    len(class_positions)
                    for class_positions in positions_by_class.values()
                )
                if target_class_ids is not None:
                    current_class_ids = (
                        sorted(positions_by_class.keys())
                        if target_class_ids == "all"
                        else target_class_ids
                    )
                    if not current_class_ids:
                        node.get_logger().warn("No detected classes in this scan area.")
                        continue
                    node.pick_and_place_class(current_class_ids, positions_by_class)
                    if node.should_restart_scan():
                        restart_scan = True
                        break

            if restart_scan:
                node.consume_restart_scan_request()
                continue

            if detected_object_count == 0:
                node.get_logger().info(
                    "No objects detected in both scan areas. Stopping."
                )
                node.publish_task_complete()
                break
        node.safe_movej(P0, vel=VELOCITY, acc=ACC) # 초기 위치로 이동
    finally:
        executor.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
