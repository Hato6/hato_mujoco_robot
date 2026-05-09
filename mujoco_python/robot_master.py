import mujoco
import mujoco.viewer
import numpy as np
import time
import math

class UiMaker:
    def __init__(self, viewer):
        """初始化绘制器，绑定当前的 viewer"""
        self.viewer = viewer
        self.geoms = []  # 用于暂存当前帧需要绘制的所有图形

    def add_text(self, pos, text, color=[1.0, 1.0, 0.0, 1.0]):
        """添加悬浮文字"""
        # 支持简单的 \n 换行拆分
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_pos = np.array(pos) + np.array([0.0, 0.0, -i * 0.05])
            self.geoms.append({'type': 'text', 'pos': line_pos, 'text': line, 'color': color})

    def add_sphere(self, pos, radius=0.03, color=[1.0, 0.0, 0.0, 1.0]):
        """添加球体 (常用于标记质心、目标落脚点)"""
        self.geoms.append({'type': 'sphere', 'pos': pos, 'size': radius, 'color': color})

    def add_box(self, pos, size=[0.05, 0.05, 0.05], color=[0.0, 1.0, 0.0, 1.0]):
        """添加方块 (size 为 [半长, 半宽, 半高])"""
        self.geoms.append({'type': 'box', 'pos': pos, 'size': size, 'color': color})

    def add_line(self, pt1, pt2, width=0.005, color=[0.0, 0.5, 1.0, 1.0]):
        """添加线段 (常用于连接两个关节，或绘制轨迹)"""
        self.geoms.append({'type': 'line', 'pt1': pt1, 'pt2': pt2, 'width': width, 'color': color})

    def add_arrow(self, pt1, pt2, width=0.01, color=[1.0, 0.0, 1.0, 1.0]):
        """添加箭头 (常用于表示速度向量、受力方向)"""
        self.geoms.append({'type': 'arrow', 'pt1': pt1, 'pt2': pt2, 'width': width, 'color': color})

    def drawer(self):
        """将收集到的所有图形渲染到 MuJoCo 画面中"""
        if self.viewer is None:
            return
        with self.viewer.lock():
            # 限制最大渲染数量，防止超出 MuJoCo 的 maxgeom 导致崩溃
            num_geoms_to_draw = min(len(self.geoms), self.viewer.user_scn.maxgeom)
            self.viewer.user_scn.ngeom = num_geoms_to_draw
            for i in range(num_geoms_to_draw):
                cmd = self.geoms[i]
                geom = self.viewer.user_scn.geoms[i]
                color = np.array(cmd['color'], dtype=np.float32)
                if cmd['type'] == 'text':
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3), cmd['pos'], np.eye(3).flatten(), color)
                    geom.label = cmd['text']
                elif cmd['type'] == 'sphere':
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([cmd['size'], 0, 0]), cmd['pos'], np.eye(3).flatten(), color)
                elif cmd['type'] == 'box':
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, np.array(cmd['size']), cmd['pos'], np.eye(3).flatten(), color)
                elif cmd['type'] in ['line', 'arrow']:
                    # MuJoCo 提供了一个专门用来画两点之间连接线的超级函数 mjv_makeConnector
                    g_type = mujoco.mjtGeom.mjGEOM_LINE if cmd['type'] == 'line' else mujoco.mjtGeom.mjGEOM_ARROW
                    p1, p2 = cmd['pt1'], cmd['pt2']
                    mujoco.mjv_connector(geom, g_type, cmd['width'], p1, p2)
                    geom.rgba = color
        # 渲染完毕后，清空列表，准备接收下一帧的图形
        self.geoms.clear()

class RobotMath:
    @staticmethod
    def quat_to_euler(quat):
        w, x, y, z = quat
        # 1. Roll (绕 X 轴旋转)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        # 2. Pitch (绕 Y 轴旋转)
        t2 = +2.0 * (w * y - z * x)
        # 限制 t2 的范围在 [-1, 1] 之间，防止数值计算误差导致 math.asin 报错
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        # 3. Yaw (绕 Z 轴旋转)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return roll, pitch, yaw
    
    def euler_to_quat(roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [w, x, y, z]

    def len2pos(pos, yaw = 0, len = 0.1):
        # 计算给定位置和朝向下，前方一定距离处的坐标
        x = pos[0] + len * math.cos(yaw)
        y = pos[1] + len * math.sin(yaw)
        return [x, y]
    
    def pos2len_yaw(pos1 = [0,0], pos2 = [0,0]):
        len = math.hypot(pos2[0]-pos1[0], pos2[1]-pos1[1])
        if len == 0:
            return 0, 0
        yaw = math.atan2(pos2[1]-pos1[1], pos2[0]-pos1[0])
        return len, yaw

class RobotController:
    def __init__(self):
        # 加载模型和数据
        self.model = mujoco.MjModel.from_xml_path("../robot/robot.xml")
        self.data = mujoco.MjData(self.model)
        # 腿部电机 (位置控制)
        self.leg_actuators = ['lf_motor', 'lb_motor', 'rf_motor', 'rb_motor']
        self.leg_joints = ['jlf_hip_pitch', 'jlb_hip_pitch', 'jrf_hip_pitch', 'jrb_hip_pitch']
        self.leg_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.leg_actuators]
        # 轮子电机 (速度控制)
        self.wheel_actuators = ['l_wheel', 'r_wheel']
        self.wheel_joints = ['jl_motor', 'jr_motor']
        self.wheel_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.wheel_actuators]

        self.kp_leg = 100.0
        self.kd_leg = 5.0
        self.kp_wheel = 1.0
        self.kd_wheel = 0.0

    def model_set(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

    def leg_pos_set(self, motor_index, pos):
        current_pos = self.data.joint(self.leg_joints[motor_index]).qpos[0]
        current_vel = self.data.joint(self.leg_joints[motor_index]).qvel[0]
        error = pos - current_pos
        torque = self.kp_leg * error - self.kd_leg * current_vel
        # 应用到致动器
        self.data.ctrl[self.leg_ids[motor_index]] = torque

    def wheel_vel_set(self, motor_index, vel):
        # current_pos = self.data.joint(self.wheel_joints[i]).qpos[0]
        # current_vel = self.data.joint(self.wheel_joints[motor_index]).qvel[0]
        # vel_error = vel - current_vel
        # torque = self.kp_wheel * vel_error + self.kd_wheel * vel_error
        # 应用到致动器
        # self.data.ctrl[self.wheel_ids[motor_index]] = torque

        self.data.ctrl[self.wheel_ids[motor_index]] = vel

if __name__ == "__main__":
    Robot = RobotController()
    Robot.model_set("../robot/robot.xml")

    # 电机方向设置 (调整正负号来匹配实际运动方向)
    leg_dir   = [-1, 1, 1, -1]
    wheel_dir = [1, -1]
    # 目标值
    target_leg_pos   = [0.7, 0.7, 0.7, 0.7]
    target_wheel_vel = [0.0, 0.0]

    with mujoco.viewer.launch_passive(Robot.model, Robot.data) as viewer:
        # # 设定初始高度
        # Robot.data.qpos[2] = 0.2
        # # 设置重力加速度
        # Robot.model.opt.gravity[2] = 0.0

        while viewer.is_running():
            step_start = time.time()

            # 腿部位置控制
            for i in range(len(Robot.leg_ids)):
                Robot.leg_pos_set(i, target_leg_pos[i] * leg_dir[i])
            # 轮子速度控制
            for i in range(len(Robot.wheel_ids)):
                Robot.wheel_vel_set(i, target_wheel_vel[i] * wheel_dir[i])

            # 执行仿真步进
            mujoco.mj_step(Robot.model, Robot.data)
            viewer.sync()
            # 确保每个仿真步长的时间一致，避免过快或过慢
            time_until_next_step = Robot.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                