import matplotlib.pyplot as plt
import math
import numpy as np


# path = [[0, 0],
#         [1, 0],
#         [1.5, 1.5],
#         [1.5, 2],
#         [1.1, 4.5],
#         [0.7, 4.5]]

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


path = [[0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
        [5, 0]]

way_ref = path
way = [way_ref[0]]
ts = []
ang = []
current_id = 1

step_length = 0.5
v = 1.5

last_id = len(path) - 1

while current_id < last_id:
    distance_traveled = step_length * v
    distance = math.dist(way[-1], way_ref[current_id])
    while distance_traveled > distance and current_id < last_id:
        distance_traveled = distance_traveled - distance
        if distance > 0:
            way.append(way_ref[current_id])
            ts.append(distance / v)
        current_id = current_id + 1
        distance = math.dist(way[-1], way_ref[current_id])

    if current_id < last_id:
        way.append([way[-1][0] + (way_ref[current_id][0] - way[-1][0]) * distance_traveled / distance,
                    way[-1][1] + (way_ref[current_id][1] - way[-1][1]) * distance_traveled / distance])
        ts.append(distance_traveled / v)
    else:
        ts.append(math.dist(way[-1], way_ref[last_id]) / v)
        way.append(way_ref[last_id])

d = np.diff(way, axis=0)
th = []
rh = []
for dd in d:
    r, t = cart2pol(dd[0], dd[1])
    th.append(t)
    rh.append(r)
# segdists = np.sqrt((d ** 2).sum(axis=1))

for w, dist, t, ang in zip(way, rh, ts, th):
    print(w, dist, dist / t, ang)

plt.plot([i[0] for i in path], [i[1] for i in path], 'x')
plt.plot([i[0] for i in path], [i[1] for i in path], '--')

plt.plot([i[0] for i in way], [i[1] for i in way], '+')

plt.figure()
plt.quiver([w[0] for w in way[:-1]], [w[1] for w in way[:-1]], [dd[0] / t for dd, t in zip(d, ts)],
           [dd[1] / t for dd, t in zip(d, ts)])

plt.show()
