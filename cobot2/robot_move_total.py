from cobot2.onrobot import RG
import DR_init
import rclpy
from od_msg.srv import SrvBasePositions
from rclpy.node import Node
import time
from cobot2.test_retain import perform_movec
from std_msgs.msg import Int32
from std_srvs.srv import Trigger
from dsr_msgs2.srv import MoveStop

SERVICE_TIMEOUT_SEC = 15.0
TRASH_FULL_CHECK_PERIOD_SEC = 2.0
EMERGENCY_STOP_CHECK_PERIOD_SEC = 0.1

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
        )
        self.start_requested = False
        self.emergency_stopped = False
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

        gripper.open_gripper()  # 그리퍼 열기
        while gripper.get_status()[0]:
            time.sleep(0.1)
        self.safe_movej(P0, vel=VELOCITY, acc=ACC) # 초기 위치로 이동

    def start_condition_callback(self, msg):
        if msg.data == 1:
            self.start_requested = True
            self.get_logger().info("Received start_condition=1.")

    def wait_for_start_condition(self):
        self.get_logger().info("Waiting for start_condition=1...")
        while rclpy.ok() and not self.start_requested:
            rclpy.spin_once(self, timeout_sec=0.1)

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
            rclpy.spin_once(self, timeout_sec=EMERGENCY_STOP_CHECK_PERIOD_SEC)

    def safe_movel(self, *args, **kwargs):
        from DSR_ROBOT2 import movel

        self.wait_while_emergency_stopped()
        result = movel(*args, **kwargs)
        rclpy.spin_once(self, timeout_sec=0.0)
        self.wait_while_emergency_stopped()
        return result

    def safe_movej(self, *args, **kwargs):
        from DSR_ROBOT2 import movej

        self.wait_while_emergency_stopped()
        result = movej(*args, **kwargs)
        rclpy.spin_once(self, timeout_sec=0.0)
        self.wait_while_emergency_stopped()
        return result

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
        rclpy.spin_until_future_complete(self, future, timeout_sec=SERVICE_TIMEOUT_SEC)

        if not future.done():
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
        rclpy.spin_until_future_complete(self, future, timeout_sec=SERVICE_TIMEOUT_SEC)

        if not future.done():
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
                    trash_count_msg.data = position["class_id"]
                    self.db_publisher.publish(trash_count_msg)
                    self.wait_until_trash_not_full()
                else:
                    self.get_logger().warn(
                        "Pick failed. Skipping trash_count publish."
                    )
            else:
                self.side_pick_and_place_target(position["class_id"], target_pos)

    def request_trash_full_flag(self):
        while not self.flag_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for is_trash_full service...")

        future = self.flag_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=SERVICE_TIMEOUT_SEC)

        if not future.done():
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
            flag = self.request_trash_full_flag()
            if flag != '1':
                return
            self.safe_movej(P0, vel=VELOCITY, acc=ACC) # 초기 위치로 이동
            self.get_logger().warn(
                "Trash bin is full. Waiting until Firebase flag becomes 0..."
            )
            time.sleep(TRASH_FULL_CHECK_PERIOD_SEC)

    def wait_for_gripper_motion(self):
        while gripper.get_status()[0]:
            time.sleep(0.1)

    def close_gripper_and_wait(self, force_val=100):
        gripper.close_gripper(force_val=force_val)
        self.wait_for_gripper_motion()

    def is_object_gripped(self):
        status = gripper.get_status()
        width = gripper.get_width()
        if status[1]:
            self.get_logger().info(f"Grip detected. width={width:.1f}mm")
            return True

        self.get_logger().warn(f"Grip failed. width={width:.1f}mm")
        return False

    def pick_and_place_target(self, class_id, target_pos):
        from DSR_ROBOT2 import mwait, DR_MV_MOD_REL

        if class_id == 1:  # plastic bottle
            thrash_bin_posj = [-77.39, 13.9, 77.34, 0.17, 88.33, -76.44]
        elif class_id == 2:  # label o
            thrash_bin_posj = [-77.39, 13.9, 77.34, 0.17, 88.33, -76.44]
        elif class_id == 3:  # plastic bottle
            thrash_bin_posj = [-77.39, 13.9, 77.34, 0.17, 88.33, -76.44]
        elif class_id == 4:  # can
            thrash_bin_posj = [-47.38, 32.53, 49.7, -0.09, 97.36, -46.6]
        elif class_id == 5:  # box
            thrash_bin_posj = [-34.81, 44.83, 41.64, 11.51, 60.07, -33.78]
        else:
            thrash_bin_posj = [-77.39, 13.9, 77.34, 0.17, 88.33, -76.44]

        close_picture_pose = list(target_pos)
        close_picture_pose[2] += 40
        close_picture_pose[1] += -85  # 사진 찍는 위치로 이동 (조정 필요)
        print("1. Moving to close picture pose...")
        self.safe_movel(close_picture_pose, vel=VELOCITY, acc=ACC) # 사진 찍는 위치로 이동
        print("Taking picture and getting more accurate position...")
        # 사진 찍고 더 정확한 위치 받아오기 + box 정보
        center_position = None
        self.center_base_positions_client.wait_for_service()
        future = self.center_base_positions_client.call_async(SrvBasePositions.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=SERVICE_TIMEOUT_SEC)

        if not future.done():
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

        # pick_pos_up = list(target_pos)
        # pick_pos_up[2] += 100  # 대상 위치 위로 이동
        pick_up = list(target_pos)
        pick_up[2] += 100  # 대상 위치 위로 이동
        print("2. Moving to pick position...")
        self.safe_movel(pick_up, vel=VELOCITY, acc=ACC) # 대상 위치 위로 이동
        # movel([-10,0,0,0,0,0], vel=VELOCITY, acc=ACC, ref=DR_TOOL)

        print("3. Moving to target position...")
        self.safe_movel([0,0,-100,0,0,0], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL) # 대상 위치로 이동
        mwait()
         # 대상 위치로 이동

        self.close_gripper_and_wait(force_val=100)

        print("4. Moving up with the object...")
        pick_up = list(target_pos)
        pick_up[2] += 150  # 대상 위치 위로 이동
        self.safe_movel(pick_up, vel=VELOCITY, acc=ACC) # 대상 위치 위로 이동
        if not self.is_object_gripped():
            print("============Grip failed. Moving back up and skipping this object.============")
            gripper.open_gripper()
            self.wait_for_gripper_motion()
            return False

        print("5. Moving to bucket position...")
        # movel(thrash_bin_pos, vel=VELOCITY, acc=ACC) # 버킷 위치로 이동
        self.safe_movej(thrash_bin_posj, vel=VELOCITY, acc=ACC) # 버킷 위치로 이동
        gripper.open_gripper()  # 그리퍼 열기
        while gripper.get_status()[0]:
            time.sleep(0.1)

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

        # 사진 찍고 더 정확한 위치 받아오기 + box 정보
        print("Taking picture and getting more accurate position...")
        center_position = None
        self.center_base_positions_client.wait_for_service()
        future = self.center_base_positions_client.call_async(SrvBasePositions.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=SERVICE_TIMEOUT_SEC)

        if not future.done():
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

        thrash_bin_pos = [300, -300, 200, 155.1, 179.8, 155.4]
        if center_position["class_id"] == 1:  # label x
            thrash_bin_pos = [300, -300, 200, 155.1, 179.8, 155.4]
        elif center_position["class_id"] == 2:  # label o
            thrash_bin_pos = [300, -300, 200, 155.1, 179.8, 155.4]

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
        print("3. Moving to target position...")
        self.safe_movel(target_pos, vel=VELOCITY, acc=ACC) # 대상 위치로 이동
        
        print("gripping...")
        self.close_gripper_and_wait(force_val=100)
        
        print("4. Moving side up with the object...")
        self.safe_movel(posx([0,0,100,0,0,0]), vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL) # 대상 위치 위로 이동
        if not self.is_object_gripped():
            gripper.open_gripper()
            self.wait_for_gripper_motion()
            return False

        print("5. Moving to bucket position...")
        self.safe_movel(thrash_bin_pos, vel=VELOCITY, acc=ACC) # 버킷 위치로 이동

        gripper.open_gripper()  # 그리퍼 열기
        while gripper.get_status()[0]:
            time.sleep(0.1)

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
    


def main(args=None):
    rclpy.init(args=args)
    dsr_node = rclpy.create_node("robot_move3_dsr", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node
    node = RobotMoveNode()
    try:
        from DSR_ROBOT2 import posx
        # Test mode: start immediately without waiting for Firebase start_condition.
        # node.wait_for_start_condition()
        CENTER_POINT = (500.0, -50.0)
        Z0 = 300
        HIGHT = 150
        target_class_ids = "all"
        center_xyz = node.request_center_of_centers_xyz()
        if center_xyz is not None:
            perform_movec(center_xyz)
        else:
            node.get_logger().warn(
                "center_of_center_points returned no xyz. Skipping stirring motion."
            )

        gripper.open_gripper()  # 그리퍼 열기
        while gripper.get_status()[0]:
            time.sleep(0.1)
        while True:
            detected_object_count = 0
            for y_offset in (HIGHT / 2, -HIGHT / 2):
                p = posx([CENTER_POINT[0], CENTER_POINT[1] + y_offset, Z0, 0, 180, 0])
                print(p)
                node.safe_movel(p, vel=VELOCITY, acc=ACC)
                positions_by_class = node.request_base_positions()
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

            if detected_object_count == 0:
                node.get_logger().info(
                    "No objects detected in both scan areas. Stopping."
                )
                node.publish_task_complete()
                break
        node.safe_movej(P0, vel=VELOCITY, acc=ACC) # 초기 위치로 이동
    finally:
        node.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
