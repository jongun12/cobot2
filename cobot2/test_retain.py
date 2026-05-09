import time

import DR_init

from cobot2.onrobot import RG


VELOCITY, ACC = 60, 60
JReady = [0, 0, 90, 0, 90, 0]

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"

_gripper = None


def get_gripper():
    global _gripper
    if _gripper is None:
        _gripper = RG(
            GRIPPER_NAME,
            TOOLCHANGER_IP,
            TOOLCHANGER_PORT,
        )
    return _gripper


def perform_movec(base_xyz):
    from DSR_ROBOT2 import (
        DR_BASE,
        get_current_posx,
        movec,
        movej,
        movel,
        posx,
    )

    bx, by, bz = base_xyz
    current_pos = get_current_posx()[0]

    print("\n==============================")
    print("BASE XYZ")
    print(base_xyz)
    print("==============================")

    bz = max(bz, 5.0)
    safe_z = 300

    rx = current_pos[3]
    ry = current_pos[4]
    rz = current_pos[5]

    above_target = posx(bx, by, safe_z, rx, ry, rz)
    pick_point = posx(bx, by, 100, rx, ry, rz)
    first_midpoint = posx(bx + 100, by, 100, rx, ry, rz)
    first_endpoint = posx(bx, by + 100, 100, rx, ry, rz)
    second_midpoint = posx(bx - 100, by, 100, rx, ry, rz)
    second_endpoint = posx(bx, by - 100, 100, rx, ry, rz)

    print("\nSTART MOVEC TASK")

    movel(
        above_target,
        vel=50,
        acc=50,
    )
    get_gripper().close_gripper()
    time.sleep(1.0)

    print("GRIPPER CLOSED")

    movel(
        pick_point,
        vel=20,
        acc=20,
    )

    print("ARRIVED TARGET")
    print("\nCURRENT POS:")
    print(get_current_posx()[0])

    print("\nSTART FIRST MOVEC")
    movec(
        first_midpoint,
        first_endpoint,
        vel=80,
        acc=80,
        ref=DR_BASE,
    )
    print("FIRST MOVEC DONE")

    print("\nSTART SECOND MOVEC")
    movec(
        second_midpoint,
        second_endpoint,
        vel=80,
        acc=80,
        ref=DR_BASE,
    )
    print("SECOND MOVEC DONE")

    movej(
        JReady,
        vel=VELOCITY,
        acc=ACC,
    )
    print("RETURN HOME")
    print("TASK COMPLETE")
