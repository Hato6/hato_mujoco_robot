##################################################################
# 这个脚本演示了如何将 URDF 文件转换为 MuJoCo 的 XML 格式
# 你需要安装 mujoco-python 包，并确保你的 URDF 文件路径正确
# 运行这个脚本后，会在当前目录下生成一个名为 robot.xml 的 MuJoCo 模型文件
# 需要注意相关.stl文件也要放在同一目录下，或者在URDF中正确指定路径
# by Hato, 2026-04-25
##################################################################
import mujoco

# 1. 直接加载你的 URDF 文件（此时 MuJoCo 会调用内部的 URDF 解析器）
model = mujoco.MjModel.from_xml_path("robot.urdf")

# 2. 将解析好的模型保存为 MuJoCo 原生 XML 格式
mujoco.mj_saveLastXML("robot.xml", model)

print("转换成功！已生成 robot.xml")