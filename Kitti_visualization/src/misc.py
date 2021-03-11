#!/usr/bin/env python3
import numpy as np

def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2, l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    corner_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corner_3d_cam2 += np.vstack([x, y, z])
    return corner_3d_cam2

def distance_point_to_segment(P, A, B):
    AP = P-A
    BP = P-B
    AB = B-A
    if np.dot(AB, AP)>=0 and np.dot(-AB, BP)>=0:
        return np.abs(np.cross(AP, AB))/np.linalg.norm(AB), np.dot(AP, AB)/np.dot(AB, AB) * AB + A
    d_PA = np.linalg.norm(AP)
    d_PB = np.linalg.norm(BP)
    if d_PA < d_PB:
        return d_PA, A
    return d_PB, B

def min_distance_cuboids(cub1, cub2):
    minD = 1e5
    for i in range(4):
        for j in range(4):
            d, Q = distance_point_to_segment(cub1[i, :2], cub2[j, :2], cub2[j+1, :2])
            if d < minD:
                minD = d
                minP = cub1[i, :2]
                minQ = Q
                
    for i in range(4):
        for j in range(4):
            d, Q = distance_point_to_segment(cub2[i, :2], cub1[j, :2], cub1[j+1, :2])
            if d < minD:
                minD = d
                minP = cub2[i, :2]
                minQ = Q
    return minP, minQ, minD