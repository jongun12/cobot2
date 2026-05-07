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
P0 = [0, 0, 90, 0, 90, 0]
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
        # gripper.open_gripper()  # 그리퍼 열기
        # while gripper.get_status()[0]:
        #     time.sleep(0.1)
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

    def prompt_target_class_id(self, positions_by_class):
        if not positions_by_class:
            self.get_logger().warn("No detected positions are available.")
            return None

        print("====================================")
        print("Detected class IDs:")
        for class_id, class_positions in sorted(positions_by_class.items()):
            print(f"  {class_id} : {len(class_positions)} object(s)")

        while True:
            user_input = input("Pick class_id to move (q to quit): ").strip()
            if user_input.lower() == "q":
                self.get_logger().info("Quit the program...")
                return None

            try:
                class_id = int(user_input)
            except ValueError:
                print("Please enter a numeric class_id.")
                continue

            if class_id not in positions_by_class:
                print(f"class_id={class_id} was not detected.")
                continue

            return class_id

    def pick_and_place_class(self, class_id, positions_by_class):
        target_positions = positions_by_class.get(class_id, [])
        if not target_positions:
            self.get_logger().warn(f"No positions found for class_id={class_id}.")
            return

        self.get_logger().info(
            f"Picking {len(target_positions)} object(s) with class_id={class_id}."
        )
        for index, position in enumerate(target_positions, start=1):
            target_pos = self._position_to_pose(position)
            self.get_logger().info(
                "Picking %d/%d: pose=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]"
                % (
                    index,
                    len(target_positions),
                    target_pos[0],
                    target_pos[1],
                    target_pos[2],
                    target_pos[3],
                    target_pos[4],
                    target_pos[5],
                )
            )
            self.pick_and_place_target(target_pos)
    
    def pick_and_place_target(self, target_pos):
        from DSR_ROBOT2 import movel, movej, mwait, posx, DR_MV_MOD_REL

        # gripper.open_gripper()  # 그리퍼 열기
        # while gripper.get_status()[0]:
        #     time.sleep(0.1)

        print("1. Moving to initial position...")
        movej(P0, vel=VELOCITY, acc=ACC) # 초기 위치로 이동
        time.sleep(1)  # 잠시 대기

        close_picture_pose = list(target_pos)
        # 사진 찍는 위치로 이동 (조정 필요)
        close_picture_pose[1] += 70 # y
        close_picture_pose[2] += 40 # z
        close_picture_pose[3] = 90
        close_picture_pose[4] = -90
        close_picture_pose[5] = 90
        print("2. Moving to close picture pose...")
        movel(close_picture_pose, vel=VELOCITY, acc=ACC) # 사진 찍는 위치로 이동
        time.sleep(1)  # 잠시 대기

        # # 사진 찍고 더 정확한 위치 받아오기 + box 정보
        # self.center_base_positions_client.wait_for_service()
        # future = self.center_base_positions_client.call_async(SrvBasePositions.Request())
        # rclpy.spin_until_future_complete(self, future, timeout_sec=SERVICE_TIMEOUT_SEC)

        # if not future.done():
        #     future.cancel()
        #     self.get_logger().error(
        #         "Timed out waiting for get_center_base_positions "
        #         f"after {SERVICE_TIMEOUT_SEC:.1f}s."
        #     )
        # elif future.result() is None:
        #     self.get_logger().error("Failed to call get_center_base_positions service.")
        # else:
        #     response = future.result()
        #     if not response.success:
        #         self.get_logger().warn(response.message)
        #     else:
        #         center_positions = self._parse_base_positions_response(response)
        #         if center_positions:
        #             center_position = center_positions[0]
        #             target_pos = self._position_to_pose(center_position)
        #             self.get_logger().info(
        #                 "Updated target pose from center object: "
        #                 "[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]"
        #                 % (
        #                     target_pos[0],
        #                     target_pos[1],
        #                     target_pos[2],
        #                     target_pos[3],
        #                     target_pos[4],
        #                     target_pos[5],
        #                 )
        #             )
        #         else:
        #             self.get_logger().warn(
        #                 "get_center_base_positions returned no positions."
        #             )

        # box 가로 너비 계산 (어쩌면 픽셀 단위일 수도 있으니 실제 크기로 변환 필요할 수도 있음, cal_positon에서 처리할 수도 있음)
        


        # ==================================================================================

        target_pos[1] += -60  # 정확한 위치 (조정 필요)
        target_pos[3] = 90
        target_pos[4] = -90
        target_pos[5] = 90

        pick_pos_side = list(target_pos)
        pick_pos_side[1] += 100  # 대상 위치 위로 이동
        
        print("3. Moving to side position...")
        movel(pick_pos_side, vel=VELOCITY, acc=ACC) # 대상 위치 위로 이동
        time.sleep(1)  # 잠시 대기
        print("4. Moving to target position...")
        movel(target_pos, vel=VELOCITY, acc=ACC) # 대상 위치로 이동
        
        print("gripping...")

        movel(posx([0,0,50,0,0,0]), vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL) # 대상 위치 위로 이동

        # gripper.move_gripper(800)  # 그리퍼 닫기
        # while gripper.get_status()[0]:
        #     time.sleep(0.1)

        print("6. Moving to bucket position...")
        movel(BUCKET_POS, vel=VELOCITY, acc=ACC) # 버킷 위치로 이동

        # gripper.open_gripper()  # 그리퍼 열기
        # while gripper.get_status()[0]:
        #     time.sleep(0.1)

        print("7. Moving back to initial position...")
        movej(P0, vel=VELOCITY, acc=ACC) # 초기 위치로 이동
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
    
    def shaking1(self):
        print("Performing task...")
        from DSR_ROBOT2 import (
            posx,
            movej,movel,
            set_ref_coord,
            wait,
            movej,wait, task_compliance_ctrl,set_desired_force,DR_AXIS_Z,
            set_ref_coord,DR_FC_MOD_REL,get_tool_force,DR_TOOL ,release_force,
            release_compliance_ctrl,get_current_posj, get_current_posx, DR_WORLD, DR_MV_MOD_REL
        )

        def force_control():
            set_ref_coord(1) # Tool 좌표계 설정
            task_compliance_ctrl(stx=[1000, 1000, 200, 200, 200, 200])
            wait(0.5) # 안정화 대기(필수)
            set_desired_force(fd=[0, 0, 15, 0, 0, 0], dir=[0, 0, 1, 0, 0, 0], mod=DR_FC_MOD_REL)
            while True:
            #force 확인용
                f_list = get_tool_force(DR_TOOL)
                # print(f_list[2])
                if abs(f_list[2]) >= 14:
                    break
            wait(0.5)
            release_force()
            release_compliance_ctrl()
            set_ref_coord(0)
            
        movel(posx([0,0,20,0,0,0]), vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL) # 대상 위치 위로 이동
        time.sleep(0.5)
        movel(posx([0,0,-20,0,0,0]), vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL) # 대상 위치 위로 이동
        time.sleep(0.5)


def main(args=None):
    rclpy.init(args=args)
    dsr_node = rclpy.create_node("robot_move3_dsr", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node
    node = RobotMoveNode()
    from DSR_ROBOT2 import movel, mwait, posx, movec, move_spiral
    # node.pick_and_place_target(posx([358.7, 27, 309.2, 2.6, 179.4, 7.1]))
    start_z = 160
    end_z = 140
    RU = [595, 229, 98, 0, 180, 0] # 우상단
    RD = [595, -133, 98, 0, 180, 0] # 우하단
    LU = [232, 229, 98, 0, 180, 0] # 좌상단
    LD = [232, -133, 98, 0, 180, 0] # 좌하단

    def move_horizontal_scan(scan_z, rows=6):
        VELOCITY = 80
        ACC = 80
        if rows < 2:
            rows = 2

        def interpolate_pose(start_pose, end_pose, ratio):
            return [
                start_value + ((end_value - start_value) * ratio)
                for start_value, end_value in zip(start_pose, end_pose)
            ]

        print(f"Moving horizontal scan in {rows} rows...")
        for row_index in range(rows):
            ratio = row_index / (rows - 1)
            right_pose = interpolate_pose(RU, RD, ratio)
            left_pose = interpolate_pose(LU, LD, ratio)
            right_pose[2] = scan_z
            left_pose[2] = scan_z

            start_pose, end_pose = (
                (right_pose, left_pose)
                if row_index % 2 == 0
                else (left_pose, right_pose)
            )

            if row_index == 0:
                print("Moving to scan start pose...")
                movel(posx(start_pose), vel=VELOCITY, acc=ACC)
                mwait()

            print(f"Scanning row {row_index + 1}/{rows}...")
            movel(posx(end_pose), vel=VELOCITY, acc=ACC)
            mwait()

            if row_index < rows - 1:
                next_ratio = (row_index + 1) / (rows - 1)
                down_pose = interpolate_pose(
                    LU if row_index % 2 == 0 else RU,
                    LD if row_index % 2 == 0 else RD,
                    next_ratio,
                )
                down_pose[2] = scan_z
                print("Moving down to next row...")
                movel(posx(down_pose), vel=VELOCITY, acc=ACC)
                mwait()

    def force_control():
        from DSR_ROBOT2 import (
        posx,
        movej,movel,
        set_ref_coord,
        wait,
        movej,wait, task_compliance_ctrl,set_desired_force,DR_AXIS_Z,
        set_ref_coord,DR_FC_MOD_REL,get_tool_force,DR_TOOL ,release_force,
        release_compliance_ctrl,get_current_posj, get_current_posx, DR_WORLD, DR_MV_MOD_REL
        )
        set_ref_coord(1) # Tool 좌표계 설정
        task_compliance_ctrl(stx=[1000, 1000, 200, 200, 200, 200])
        wait(0.5) # 안정화 대기(필수)
        set_desired_force(fd=[0, 0, 15, 0, 0, 0], dir=[0, 0, 1, 0, 0, 0], mod=DR_FC_MOD_REL)
        while True:
           #force 확인용
            f_list = get_tool_force(DR_TOOL)
            # print(f_list[2])
            if abs(f_list[2]) >= 14:
                break
        wait(0.5)
        release_force()
        release_compliance_ctrl()
        set_ref_coord(0)

    gripper.move_gripper(10)
    while gripper.get_status()[0]:
        time.sleep(0.1)
    VELOCITY = 80
    ACC = 80
    # move_spiral(posx([358.7, 27, 309.2, 2.6, 179.4, 7.1]), radius=20, turns=3, vel=VELOCITY, acc=ACC)
    movel(posx([595, 229, 200, 0, 180, 0]), vel=VELOCITY, acc=ACC)
    print(f"Start scan at z={start_z}...")
    move_horizontal_scan(start_z, rows=6)
    print(f"End scan at z={end_z}...")
    move_horizontal_scan(end_z, rows=6)


    try:
        positions_by_class = node.request_base_positions()
        target_class_id = node.prompt_target_class_id(positions_by_class)
        if target_class_id is not None:
            node.pick_and_place_class(target_class_id, positions_by_class)
    finally:
        node.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
