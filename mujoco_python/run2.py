import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import robot_master
from pynput import keyboard
import threading

Robot = robot_master.RobotController()
robot_path = "../robot/scene.xml"

# 电机方向设置 (调整正负号来匹配实际运动方向)
leg_dir          = [-1, 1, 1, -1]
wheel_dir        = [1, -1]
# 目标值
target_leg_pos   = [0.7, 0.7, 0.7, 0.7]
target_wheel_vel = [0.0, 0.0]

roll_error       = 0
kp_roll          = 250*1000
kd_roll          = 0

pos_error        = 0
pos_xz           = 0.1332
kp_pos           = 1100
kd_pos           = 70

vel_error        = 0
kp_vel           = 0
kd_vel           = 0

yaw_error        = 0
kp_yaw           = 0
kd_yaw           = 0

goal_vel = 0.0
target_turn_vel = 0.0

quat = np.array([0.0, 0.0 ,0.0 ,0.0])
gyro = np.array([0.0, 0.0 ,0.0])
acc  = np.array([0.0, 0.0 ,0.0])
vel  = np.array([0.0, 0.0 ,0.0])
pos  = np.array([0.0, 0.0 ,0.0])
roll, pitch, yaw = 0.0, 0.0, 0.0

def on_press(key):
    global goal_vel, target_turn_vel
    
    # 步进量：每次按键增加/减少多少速度
    dv = 0.1   
    dw = 0.5   
    try:
        if key.char == "p":
            goal_vel += dv
        elif key.char == ";":
            goal_vel -= dv
        elif key.char == "l":
            target_turn_vel += dw
        elif key.char == "'":
            target_turn_vel -= dw
        elif key.char == " ":
            goal_vel = 0.0
            target_turn_vel = 0.0
        # 限幅保护：防止一直按住导致速度飙到天上
        goal_vel = max(-1.5, min(1.5, goal_vel))
        target_turn_vel = max(-3.0, min(3.0, target_turn_vel))
    except AttributeError:
        pass
def keyCtrl():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def ui_show():
    while True:
        display_text = (
            f"Vel:{vel[0]:.4}\n"
            f"goal_vel:{goal_vel:.4f}\n"
            f"roll:{roll:.4f}\n"
            f"yaw:{math.degrees(yaw):.4f}\n"
            f"x:{pos[0]:.4f}\n"
            f"pos_error:{pos_error:.4f}\n"
        )
        UI.add_text(pos+[-0.1, -0.1, 0.25], display_text, color=[0, 1, 0, 1])

        pos2 = robot_master.RobotMath.len2pos([float(pos[0]), float(pos[1])], yaw, 0.4)
        pos2.append(float(pos[2]))
        UI.add_arrow(pos, pos2, width=0.01, color=[1, 0, 1, 1])

        UI.add_arrow([0,0,0], [1,0,0], width=0.01, color=[1, 0, 1, 1])
        UI.add_arrow([0,0,0], [0,1,0], width=0.01, color=[1, 1, 0, 1])
        UI.drawer()

def main_loop():
    global target_wheel_vel, roll_error, vel_error, pos_error, yaw_error, quat, gyro, acc, vel, pos, roll, pitch, yaw
    
    quat = Robot.data.sensor('imu_quat').data
    gyro = Robot.data.sensor('imu_gyro').data
    acc  = Robot.data.sensor('imu_acc').data
    vel  = Robot.data.sensor('imu_vel').data
    pos  = Robot.data.qpos[:3]

    # roll, pitch, yaw
    roll, pitch, yaw = robot_master.RobotMath.quat_to_euler(quat)
    
    roll_error = 0 - roll
    # pos_error, yaw_error = robot_master.RobotMath.pos2len_yaw(pos[:2], [0,0])
    # print(f"pos_error: {pos_error:.2f}, yaw_error: {math.degrees(yaw_error):.2f} deg")
    pos_error = 0 - pos[0] + pos_xz
    # print(f"pos_error: {pos_error:.2f}")
    wheel_vel = (kp_roll * roll_error - kd_roll * gyro[0]) - (kp_pos * (pos_error) - kd_pos * vel[0])
    target_wheel_vel = [wheel_vel, wheel_vel]

    # 腿部位置控制
    for i in range(len(Robot.leg_ids)):
        Robot.leg_pos_set(i, target_leg_pos[i] * leg_dir[i])
    # 轮子速度控制
    for i in range(len(Robot.wheel_ids)):
        Robot.wheel_vel_set(i, target_wheel_vel[i] * wheel_dir[i])

if __name__ == "__main__":
    Robot.model_set(robot_path)
    with mujoco.viewer.launch_passive(Robot.model, Robot.data) as viewer:
        Robot.data.qpos[2] = 0.16           # 设定初始高度
        # Robot.model.opt.gravity[2] = -9.81  # 设置重力加速度
        # time.sleep(1)

        UI = robot_master.UiMaker(viewer)
        t1 = threading.Thread(target=keyCtrl)
        t1.daemon = True
        t2 = threading.Thread(target=ui_show)
        t2.daemon = True
        t1.start()
        t2.start()

        try:
            while viewer.is_running():
                step_start = time.time()

                main_loop()

                # 执行仿真步进
                mujoco.mj_step(Robot.model, Robot.data)
                viewer.sync()
                # 确保每个仿真步长的时间一致，避免过快或过慢
                time_until_next_step = Robot.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            print("\n退出程序")
