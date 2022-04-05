import matplotlib.pyplot as plt 
import numpy as np 
import math 
from scipy.spatial.transform import Rotation as Rot


def slope(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    if x2-x1 ==0:
        return np.pi/2
    # return np.arctan((y2-y1)/(x2-x1))
    return  np.arctan2((y2-y1),(x2-x1))

def rotation(theta):
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    return rot_matrix

goal = [-10,10]

start = [0,4]
theta = slope(start,goal)
rotation_mat = rotation(slope(start,goal))
# rotation_mat =  Rot.from_euler('z', -angle).as_dcm()[0:2, 0:2]

center = [0.5*(goal[0]+start[0]),0.5*(goal[1]+start[1])]
plt_points =[]


c_min = np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2)
c_best = c_min+5
radiusX = c_best/2
radiusY = 0.5*np.sqrt(c_best**2-c_min**2)
for i in range(1000):
    r = radiusX * np.sqrt(np.random.random())
    fi = 2 * np.pi * np.random.random()
    point = [r * np.cos(fi),radiusY / radiusX *  r * np.sin(fi)]
    point = np.dot(rotation_mat,point)
    point = [point[0]+center[0],point[1]+center[1]]
    plt_points.append(point)

plt_points.append(goal)
plt_points.append(start)
for x,y in plt_points:
    plt.scatter(x,y)
# plt.scatter(plt_points)
x = [start[0],goal[0]]
y = [start[1],goal[1]]
plt.plot(x,y)
plt.show()

# plt_points = np.asarray(plt_points)
# data_rot = (rotation_mat @ plt_points.T).T

# for x,y in data_rot:
#     plt.scatter(x,y)
# # plt.scatter(plt_points)
# plt.show()