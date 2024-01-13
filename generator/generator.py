import yaml
from pathlib import Path
import numpy as np
from skimage.io import imread
from numpy import random
import math

import networkx as nx
# A Python library for creating, manipulating, 
# and studying the structure, dynamics, and functions 
# of complex networks. In robotics, networkx can be used for 
# representing and manipulating graphs, which are often used in 
# path planning and navigation algorithms.
from scipy.sparse import lil_matrix
# sparse matrix format that's good for incremental construction
# 大部分是0
from skimage.morphology import dilation
from generator import helpers as he
from generator import cspline



def load_map(yaml_path: str):
    with open(yaml_path, "r") as stream:
        meta = yaml.safe_load(stream)
    if not Path(meta["image"]).is_absolute():
        meta["image"] = (Path(yaml_path).parent / meta["image"]).as_posix()

    map_image = np.flipud(imread(meta["image"]))
    if len(map_image.shape) == 3:  # if map is multilayered keep only one
        map_image = map_image[:, :, 1]

    map_binary_image = (map_image < meta["occupied_thresh"] * 255.0) * 1
    mask = np.ones((4, 4))
    map_binary_image_dilated = dilation(map_binary_image, mask)
    # 扩大障碍物像素范围

    result = np.where(map_binary_image_dilated < 1)
    nodes = list(zip(result[1] * meta["resolution"], result[0] * meta["resolution"]))
    # 然后将索引转换为坐标。 结果数组有两个数组：一个用于 y 
    # 坐标 (result[0])，另一个用于 x 坐标 (result[1])。 
    # 然后根据地图的分辨率（元[“分辨率”]）对其进行缩放，
    # 以将像素索引转换为实际的空间坐标。 这种缩放对于根据现实世界距离解释这些坐标是必要的。
    distances = lil_matrix((len(nodes), len(nodes)), dtype=np.float64)
    # 这些步骤对于机器人的路径规划至关重要。 
    # 通过将地图中的自由空间转换为节点图并计算这些节点之间的距离，
    # 可以应用 A* 或 Dijkstra 等算法来进行有效的路径查找。
    for i in range(0, len(nodes)):
        for j in range(i + 1, len(nodes)):
            dis = math.dist(nodes[i], nodes[j])
            if dis < 1.5 * meta["resolution"]:
                distances[i, j] = dis
    #  two nodes are close enough to be considered connected. 
    # The factor 1.5 is presumably chosen to ensure that nodes a
    # re not too far apart, maintaining a reasonable resolution 
    # of the path planning grid.

    G = nx.from_numpy_array(distances)

    map_data = {
        "map_file": meta["image"],
        "resolution": meta["resolution"],
        "origin": meta["origin"],
        "negate": meta["negate"],
        "occupied_thresh": meta["occupied_thresh"],
        "free_thresh": meta["free_thresh"],
        "map_raw": map_image,
        "map_binary": map_binary_image,
        "graph": G,
        "node_positions": nodes
    }

    return map_data

# # YAML 文件加载：

# # 该函数采用 YAML 文件的路径 (yaml_path) 作为输入。
# # 它打开并读取该文件，使用 yaml.safe_load 将其内容加载到字典（元）中。 该文件可能包含有
# 关地图的元数据，例如图像文件的路径、分辨率、阈值和其他参数。
# # 图像处理：

# # 它检查地图图像的路径是否是绝对路径，并在必要时进行调整。
# # 地图图像使用 imread 加载并上下翻转 (np.flipud) 以匹配坐标系（在图像坐标系和世界坐
# 标系不同的机器人中很常见）。
# # 如果图像具有多个层（例如 RGB），则为简单起见，它仅保留一层。
# # 二进制地图创建：

# # 然后该函数将图像转换为二进制映射，其中低于特定阈值（表示占用空间）的像素被标记为 1，
# 其他像素被标记为 0。该阈值取自元字典。
# # 使用膨胀（来自 skimage.morphology）进一步处理二值图。 膨胀是一种形态学操作，
# 可增加二值图像中区域的大小，通常在机器人技术中用于在障碍物周围创建安全缓冲区。
# # 节点生成和距离计算：

# # 该函数查找所有自由空间的坐标（在膨胀的二值图像中该值小于 1）并将它们存储为节点。
# # 然后，它创建一个稀疏矩阵 (lil_matrix) 来保存节点之间的距离。
# # 如果节点之间的距离在一定范围内（分辨率的 1.5 倍），则计算节点之间的距离并将其存储在该矩阵中。
# # 图构建：

# # 使用 NetworkX 根据距离矩阵构建图 (G)。 该图表示地图上自由空间的连通性，节点之间的边彼此靠近。
# # 组装地图数据：

# # 最后，该函数将所有相关数据组装到字典（map_data）中，包括原始和处理后的地图图像、
# 图形数据、节点位置以及 YAML 文件中的各种参数。
# # 返回值：

# # 该函数返回map_data，其中包含有关地图的所有必要信息，以便在机器人应用程序中进
# 行进一步处理，例如路径规划或导航。
# # 此功能是在机器人技术中加载和处理地图数据的综合方法，对于涉及由 2D 地图表示
# 的环境中的路径规划和避障的任务特别有用。


def generate_anchors(graph, starts, goals):
    paths = []
    for s, g in zip(starts, goals):
        paths.append(nx.shortest_path(graph, source=s, target=g, weight="weight"))
    return paths
# 函数来查找它们之间的最短路径。 
# 该函数可以考虑边的权重，在典型的路径规划场景中，边的权重代表节点之间行进的距离或成本。


def generate_curve(points, tension):
    up = [points[0]]
    d = np.diff(points, axis=0)
    th = []

    for dd in d:
        _, t = he.cart2pol(dd[0], dd[1])
        th.append(t)
    for i in range(len(th) - 1):
        if not th[i] - th[i + 1] == 0:
            up.append(points[i + 1])
    if not up[-1] == points[-1]:
        up.append(points[-1])

    return cspline.cspline(up, tension)
# # The function first calculates the difference (np.diff) between successive 
# points to find the directional changes.
# # It converts these differences into polar coordinates (he.cart2pol) to 
# get the angles (t).
# # It then filters the points, keeping only those where there
#  is a change in direction. This step reduces the number of points,
# focusing on the critical points that define the path's shape.
# 所以才是曲线？
# # Finally, it uses a cubic spline (cspline.cspline) to create 
# a smooth curve through these key points. The tension parameter 
# affects how tightly the curve fits to these points.


def execute_steps(step_length, agent, path):
    # generate noisy path
    way_ref = []
    for p in path:
        way_ref.append([p[0] + random.normal(loc=0, scale=agent["location_noise"]),
                        p[1] + random.normal(loc=0, scale=agent["location_noise"])])
    # way_ref=path
    way = [way_ref[0]]
    ts = []
    current_id = 1
    last_id = len(path) - 1
    while current_id < last_id:
        v = random.normal(loc=agent["speed"], scale=agent["speed_noise"])  # agent["speed"]
        distance_traveled = step_length * v
        distance = math.dist(way[-1], way_ref[current_id])
        while distance < distance_traveled and current_id < last_id:
            distance_traveled = distance_traveled - distance
            if distance > 0:
                way.append(way_ref[current_id])
                # 根据距离来获得默认的路径？
                ts.append(distance / v)
            current_id = current_id + 1
            distance = math.dist(way[-1], way_ref[current_id])

        if current_id < last_id:
            way.append([way[-1][0] + (way_ref[current_id][0] - way[-1][0]) * distance_traveled / distance,
                        way[-1][1] + (way_ref[current_id][1] - way[-1][1]) * distance_traveled / distance])
            ts.append(distance_traveled / v)
        else:
            if math.dist(way[-1], way_ref[last_id]) > 0:
                ts.append(math.dist(way[-1], way_ref[last_id]) / v)
                way.append(way_ref[last_id])

    return way, ts
# execute_steps 函数旨在模拟机器人沿着给定路径的运动，
# 同时考虑到代理的特征及其运动的一些随机性。 以下是该函数每个部分的功能的细分：

# 生成噪声路径：

# 该函数首先创建所提供路径的“嘈杂”版本。 它迭代原始路径中的每个点 (p) 
# 并向其添加随机噪声。 该噪声是使用以零为中心 (loc=0) 的正态分布生成的
# ，其标准差由 agent["location_noise"] 定义。 此步骤模拟现实世界中机器人运动和传感的不准确性。
# 初始化变量：

# way 被初始化为从噪声路径的第一个点开始的列表。
# ts 是一个空列表，稍后将用于存储点之间行驶所需的时间。
# current_id 设置为 1，从路径中的第二个点开始模拟。
# last_id 是路径中最后一个点的索引。
# 模拟沿路径的运动：

# 该函数进入 while 循环，该循环运行直到 current_id 达到 last_id。
# 在此循环中，机器人每一步的速度 (v) 通过绘制以 agent["speed"] 为中心、
# 标准差为 agent["speed_noise"] 的正态分布来确定。 这模拟了机器人速度的变化。
# # 每一步的 distance_traveled 是通过将该速度乘以 step_length 来计算的。
# # 然后，该函数检查到路径中下一个点的剩余距离是否小于 distance_traveled。
#  如果是这样，它将机器人移动到下一个点并计算该段所花费的时间。
# # 重复该过程，直到机器人移动通过路径中的所有点或直到用完一步的行进距离。
# # 处理最后一段：

# # 退出内部 while 循环后，该函数检查机器人是否已到达最后一个点。 
# 如果没有，它将根据剩余的 distance_traveled 计算沿路径的下一个点。
# # 如果机器人位于最后一个点，它会计算最后一段的行程时间并将机器人移动到最后一个点。
# # 返回值：

# # 该函数返回两个列表：way，其中包含带有噪声并根据机器人运动进行调整
# 的模拟路径；ts，其中包含路径上每个点之间行驶所需的时间。
# # 该功能提供了机器人路径跟踪的真实模拟，同时考虑了其运动的可变性
# 和路径跟踪的不准确性。 结果是机器人在现实世界中可能遵循一条更实用、
# 更逼真的路径，而不是一条理想化的、完全准确的路径。