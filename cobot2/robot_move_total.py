from cobot2.onrobot import RG
import DR_init
import rclpy
from od_msg.srv import SrvBasePositions
from rclpy.node import Node
import time

SERVICE_TIMEOUT_SEC = 15.0

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
        from DSR_ROBOT2 import movej
        gripper.open_gripper()  # 그리퍼 열기
        while gripper.get_status()[0]:
            time.sleep(0.1)
        movej(P0, vel=VELOCITY, acc=ACC) # 초기 위치로 이동

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
            return []

        if future.result() is None:
            self.get_logger().error("Failed to call get_base_positions service.")
            return []

        response = future.result()
        if not response.success:
            self.get_logger().warn(response.message)
            return []

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
                self.pick_and_place_target(position["class_id"], target_pos)
            else:
                self.side_pick_and_place_target(position["class_id"], target_pos)
            
    def pick_and_place_target(self, class_id, target_pos):
        from DSR_ROBOT2 import movel, movej, mwait, posx, DR_MV_MOD_REL, trans, DR_TOOL

        if class_id == 1:  # plastic bottle
            thrash_bin_pos = [109.9, -456.8, 183.9, -60.5, 179.5, -59.5] # 1
            thrash_bin_posj = [-77.39, 13.9, 77.34, 0.17, 88.33, -76.44]
        elif class_id == 2:  # label o
            thrash_bin_pos = [109.9, -456.8, 183.9, -60.5, 179.5, -59.5]
            thrash_bin_posj = [-77.39, 13.9, 77.34, 0.17, 88.33, -76.44]
        elif class_id == 3:  # plastic bottle
            thrash_bin_pos = [109.9, -456.8, 183.9, -60.5, 179.5, -59.5]
            thrash_bin_posj = [-77.39, 13.9, 77.34, 0.17, 88.33, -76.44]
        elif class_id == 4:  # can
            thrash_bin_pos = [402.8, -429.4, 189.2, -65.2, 179.5, -64.4]
            thrash_bin_posj = [-47.38, 32.53, 49.7, -0.09, 97.36, -46.6]
        elif class_id == 5:  # box
            thrash_bin_pos = [731.4, -429.4, 170, -17.5, 144.8, -13.7]
            thrash_bin_posj = [-34.81, 44.83, 41.64, 11.51, 60.07, -33.78]
        else:
            thrash_bin_pos = [109.9, -456.8, 183.9, -60.5, 179.5, -59.5]
            thrash_bin_posj = [-77.39, 13.9, 77.34, 0.17, 88.33, -76.44]

        close_picture_pose = list(target_pos)
        close_picture_pose[2] += 40
        close_picture_pose[1] += -70  # 사진 찍는 위치로 이동 (조정 필요)
        print("1. Moving to close picture pose...")
        movel(close_picture_pose, vel=VELOCITY, acc=ACC) # 사진 찍는 위치로 이동
        print("Taking picture and getting more accurate position...")
        # 사진 찍고 더 정확한 위치 받아오기 + box 정보
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

        if center_position["class_id"] == 1:  # label x
            target_pos[2] = 100  # 정확한 위치 (조정 필요)
        elif center_position["class_id"] == 2:  # label o
            target_pos[2] = 100  # 정확한 위치 (조정 필요)
        else:
            target_pos[2] += -70  # 정확한 위치 (조정 필요)

        # pick_pos_up = list(target_pos)
        # pick_pos_up[2] += 100  # 대상 위치 위로 이동
        pick_up = list(target_pos)
        pick_up[2] += 150  # 대상 위치 위로 이동
        print("2. Moving to pick position...")
        movel(pick_up, vel=VELOCITY, acc=ACC) # 대상 위치 위로 이동

        print("3. Moving to target position...")
        movel(target_pos, vel=VELOCITY, acc=ACC) # 대상 위치로 이동
        mwait()

        gripper.close_gripper(force_val=80)  # 그리퍼 닫기
        while gripper.get_status()[0]:
            time.sleep(0.1)

        print("4. Moving up with the object...")
        pick_up = list(target_pos)
        pick_up[2] += 150  # 대상 위치 위로 이동
        movel(pick_up, vel=VELOCITY, acc=ACC) # 대상 위치 위로 이동

        print("5. Moving to bucket position...")
        # movel(thrash_bin_pos, vel=VELOCITY, acc=ACC) # 버킷 위치로 이동
        movej(thrash_bin_posj, vel=VELOCITY, acc=ACC) # 버킷 위치로 이동
        gripper.open_gripper()  # 그리퍼 열기
        while gripper.get_status()[0]:
            time.sleep(0.1)

        print("6. Operation completed.")
    
    def side_pick_and_place_target(self, class_id, target_pos):
        from DSR_ROBOT2 import movel, movej, mwait, posx, DR_MV_MOD_REL

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
        movel(close_picture_pose, vel=VELOCITY, acc=ACC) # 사진 찍는 위치로 이동

        # 사진 찍고 더 정확한 위치 받아오기 + box 정보
        print("Taking picture and getting more accurate position...")
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
        movel(pick_pos_side, vel=VELOCITY, acc=ACC) # 대상 위치 옆으로 이동
        print("3. Moving to target position...")
        movel(target_pos, vel=VELOCITY, acc=ACC) # 대상 위치로 이동
        
        print("gripping...")
        gripper.close_gripper(force_val=100)  # 그리퍼 닫기
        while gripper.get_status()[0]:
            time.sleep(0.1)
        
        print("4. Moving side up with the object...")
        movel(posx([0,0,100,0,0,0]), vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL) # 대상 위치 위로 이동

        print("5. Moving to bucket position...")
        movel(thrash_bin_pos, vel=VELOCITY, acc=ACC) # 버킷 위치로 이동

        gripper.open_gripper()  # 그리퍼 열기
        while gripper.get_status()[0]:
            time.sleep(0.1)

        print("8. Operation completed.")
        

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
        from DSR_ROBOT2 import movel, posx, movej
        CENTER_POINT = (367.6, -20.0)
        Z0 = 300
        WIDTH = 200
        HIGHT = 150
        target_class_ids = node.prompt_target_class_ids_before_scan()
        for i in range(4):
            p = posx([CENTER_POINT[0] + (-WIDTH/2 if i%2==0 else WIDTH/2), CENTER_POINT[1] + (HIGHT/2 if i//2==0 else -HIGHT/2), Z0, 0, 180, 0])
            print(p)
            movel(p, vel=VELOCITY, acc=ACC)
            positions_by_class = node.request_base_positions()
            if target_class_ids is not None:
                current_class_ids = (
                    sorted(positions_by_class.keys())
                    if target_class_ids == "all"
                    else target_class_ids
                )
                node.pick_and_place_class(current_class_ids, positions_by_class)
        movej(P0, vel=VELOCITY, acc=ACC) # 초기 위치로 이동
    finally:
        node.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
