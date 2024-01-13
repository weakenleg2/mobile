import json
import sys
import generator.generator
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import generator.helpers as he
import os
from datetime import datetime
import csv

now = datetime.now()


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


# Opening JSON file
with open(sys.argv[1]) as json_file:
    data = json.load(json_file)

mp = generator.generator.load_map(data["Map"])
start_nodes = []
end_nodes = []

for path in data["paths"]:
    if "start" in path.keys():
        start_nodes.append(closest_node(np.asarray(path["start"]), mp["node_positions"]))
    else:
        start_nodes.append(randint(0, max(mp["graph"].nodes)))
    if "end" in path.keys():
        end_nodes.append(closest_node(np.asarray(path["end"]), mp["node_positions"]))
    else:
        end_nodes.append(randint(0, max(mp["graph"].nodes)))

paths = generator.generator.generate_anchors(mp["graph"], start_nodes, end_nodes)

ways = []
times = []
curves = []

for path, param in zip(paths, data["paths"]):
    curve = generator.generator.generate_curve([mp["node_positions"][i] for i in path], param["tension"])
    curves.append(curve)

    for _ in range(0, param["runs"]):
        for aid in param["agents"]:
            agent = data["agents"][aid]
            w, t = generator.generator.execute_steps(data["time_step"], agent, curve)
            ways.append(w)
            times.append(t)
# check if directory exist

map_name = os.path.splitext(data["Map"])[0]
map_time = now.strftime("%Y_%m_%Y_%H_%M_%S")
dir_name = map_name + "_" + map_time
save_path = os.path.join(data["save_path"], dir_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)

# save to file
ths = []
rhs = []
uxs = []
uys = []
full = []
for i, (way, time) in enumerate(zip(ways, times)):
    d = np.diff(way, axis=0)
    th = []
    rh = []
    for dd in d:
        r, t = he.cart2pol(dd[0], dd[1])
        th.append(t)
        rh.append(r)
    ths.append(th)
    rhs.append(rh)

    uxs.append([dd[0] / t for dd, t in zip(d, time)])
    uys.append([dd[1] / t for dd, t in zip(d, time)])
    T = 0
    for w, dist, t, ang in zip(way[:-1], rh, time, th):
        T = T + t
        full.append([T, i, w[0], w[1], dist / t, ang])

header = ["T", "id", "x", "y", "speed", "orientation"]

with open(os.path.join(save_path, 'data.csv'), 'w') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(full)

# visualisation

plt.imshow(np.flipud(mp["map_binary"]), extent=(
    0 - mp["resolution"] / 2, mp["map_binary"].shape[0] * mp["resolution"] - mp["resolution"] / 2,
    0 - mp["resolution"] / 2,
    mp["map_binary"].shape[1] * mp["resolution"] - mp["resolution"] / 2), cmap="Greys")

# plt.plot([mp["node_positions"][i][0] for i in range(0, max(mp["graph"].nodes))],
#         [mp["node_positions"][i][1] for i in range(0, max(mp["graph"].nodes))], '+')

for path in paths:
    plt.plot([mp["node_positions"][i][0] for i in path], [mp["node_positions"][i][1] for i in path], '-')

for curve in curves:
    plt.plot([c[0] for c in curve], [c[1] for c in curve], ':')

plt.savefig(os.path.join(save_path, 'paths.pdf'))
plt.clf()

plt.imshow(np.flipud(mp["map_binary"]), extent=(
    0 - mp["resolution"] / 2, mp["map_binary"].shape[0] * mp["resolution"] - mp["resolution"] / 2,
    0 - mp["resolution"] / 2,
    mp["map_binary"].shape[1] * mp["resolution"] - mp["resolution"] / 2), cmap="Greys")

for way in ways:
    plt.plot([c[0] for c in way], [c[1] for c in way], "--")

plt.savefig(os.path.join(save_path, 'ways.pdf'))
plt.clf()

plt.imshow(np.flipud(mp["map_binary"]), extent=(
    0 - mp["resolution"] / 2, mp["map_binary"].shape[0] * mp["resolution"] - mp["resolution"] / 2,
    0 - mp["resolution"] / 2,
    mp["map_binary"].shape[1] * mp["resolution"] - mp["resolution"] / 2), cmap="Greys")

for way, ux, uy, th in zip(ways, uxs, uys, ths):
    plt.quiver([c[0] for c in way[:-1]], [c[1] for c in way[:-1]], ux, uy, th, angles='xy', scale_units='xy', scale=1)
plt.savefig(os.path.join(save_path, 'vectors.pdf'))
