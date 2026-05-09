########################################################################################
# 这个脚本演示了如何使用 mujoco-python 包加载一个 MJCF 模型，并启动一个交互式的可视化窗口
# 你需要注意对应的 .xml 文件的路径是否正确，并且确保相关的资源文件（如 .stl 模型文件）也在正确的位置
# 运行这个脚本后，你应该能看到一个窗口显示你的机器人模型，并且可以通过键盘或鼠标与之交互
# by Hato, 2026-04-25
########################################################################################
import mujoco
import mujoco.viewer

path = "../robot/scene.xml"
# 加载我们修改好的原生 MJCF 文件
model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)

# 启动交互式可视化窗口
mujoco.viewer.launch(model, data)