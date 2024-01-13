import numpy as np
import sys
import os
# sys.path.append(os.path.abspath('../'))
from generator import generator
import matplotlib.pyplot as plt

from random import randint

agent = {"location_noise": 0.3, "speed": 1.3, "speed_noise": 0.2}
runs = 20
paths_no = 10
time_step = 0.5
tension = 0.9
# 调用generate_anchors在图上创建许多路径。 该函数似乎为每条路径随机选择起始节点和结束节点。
# 对于每个生成的路径，使用generate_curve 创建相应的曲线。 此步骤可能是根据指定的张力平滑路径
# example/squigles.yaml 文件的内容是机器人技术中使用的典型配置文件，特别是用于映射和导航任务。 
# 让我们分解每个组件：

# 图片：squigles.png

# 这指定了地图图像的文件名。 该地图可能是环境的 2D 表示，可能描绘了障碍物和自由空间。
# 分辨率：0.5

# 该值代表地图的分辨率。 在机器人技术的背景下，这通常意味着地图图像
# 中的每个像素对应于 0.5 x 0.5 单位的物理区域（单位可以是米、厘米等，具体取决于应用）。
# 原点：[0.0，0.0，0.0]

# 该数组可能表示地图图像中与机器人坐标系原点相对应的坐标。
#  这三个值可能代表 x、y 和 theta（旋转），指示原点在地图中的位置和方向。 
# 在本例中，它设置为 (0.0, 0.0)，不旋转 (0.0)。
# 占用阈值：0.65

# 该阈值用于确定地图的哪些部分被视为“被占用”（即不可导航的空间，如墙壁或障碍物）。
#  将地图图像转换为二进制占用网格时可能会使用它。 值高于此阈值的像素
# （在任何适用的转换（例如求反之后））被标记为已占用。
# 自由阈值：0.196

# 与占用阈值类似，此阈值用于确定地图的哪些部分被视为“空闲”（即可导航空间）。 
# 值低于此阈值的像素被标记为可用空间。
# 否定：0

# 该参数通常用于反转地图图像的颜色。 值 0 可能意味着不应用否定，而值 1 
# 则意味着图像颜色反转。 当地图的颜色方案与处理算法期望的相反时（例如，
# 白色表示自由空间，黑色表示障碍物），这非常有用。
# 此 YAML 文件用作在机器人模拟或导航程序中加载和处理地图图像的配置。 
# 该程序可能使用这些参数来正确解释地图以执行路径规划和导航任务。
mp = generator.load_map("examples/squigles.yaml")

paths = generator.generate_anchors(mp["graph"],
                                             [randint(0, max(mp["graph"].nodes)) for _ in range(0, paths_no)],
                                             [randint(0, max(mp["graph"].nodes)) for _ in range(0, paths_no)])
curves = []
for path in paths:
    curves.append(generator.generate_curve([mp["node_positions"][i] for i in path], tension))

ways = []
for curve in curves:
    for _ in range(0, runs):
        ways.append(generator.execute_steps(time_step, agent, curve))
# simulates the agent's movement along each curve. 
# It's called multiple times for each curve, likely to 
# simulate different runs or variations in the agent's movement.

# visualisation
plt.imshow(np.flipud(mp["map_binary"]), extent=(
    0 - mp["resolution"] / 2, mp["map_binary"].shape[0] * mp["resolution"] - mp["resolution"] / 2,
    0 - mp["resolution"] / 2,
    mp["map_binary"].shape[1] * mp["resolution"] - mp["resolution"] / 2))

plt.plot([mp["node_positions"][i][0] for i in range(0, max(mp["graph"].nodes))],
         [mp["node_positions"][i][1] for i in range(0, max(mp["graph"].nodes))], '+')

for path in paths:
    plt.plot([mp["node_positions"][i][0] for i in path], [mp["node_positions"][i][1] for i in path], 'x')

for curve in curves:
    plt.plot([c[0] for c in curve], [c[1] for c in curve], ':')

for way in ways:
    plt.plot([c[0] for c in way], [c[1] for c in way], "--")
# The map is visualized using matplotlib. The binary representation of the map (mp["map_binary"]) is flipped upside down and displayed.
# Node positions are plotted on the map with a '+' symbol.
# Paths are visualized with 'x' markers.
# The generated curves are plotted with dotted lines (':').
# The agent's trajectories (ways) are visualized with dashed lines ("--").
plt.show()

# gr = dict(zip(mp["graph"].nodes, mp["node_positions"]))
#
# nx.draw(mp["graph"], pos=gr)
#
# for path in paths:
#     path_edges = list(zip(path,path[1:]))
#     nx.draw_networkx_nodes(mp["graph"],gr,nodelist=path,node_color='r')
#     nx.draw_networkx_edges(mp["graph"], gr, edgelist=path_edges, edge_color='r', width=10)
#
# plt.show()
