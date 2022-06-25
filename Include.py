# -*- coding: utf-8 -*-
"""
Created on Sun May 9 0:47:00 2021

@author: Sammy
"""

from HyperParams import *
from Dataset import *
import random
import time
import networkx as nx
from matplotlib import pyplot as plt

# check if a point is legal in terms of spatial limits
def withinBounds(p):
    return 0 <= p[0] < SPACE_X and 0 <= p[1] < SPACE_Y and 0 <= p[2] < SPACE_Z


def getRing(x0, y0, z0, r):
    coords_list = [[x0-x, y0-y, z0-z] for x in range(-r, r+1) \
                   for y in range(-r, r+1) \
                   for z in range(-r, r+1) \
                       if (r-1)**2 < x**2 + y**2 + z**2 <= r**2 and withinBounds([x0-x, y0-y, z0-z])]
    return coords_list
    '''
    
    # z axis
    for dz in range(-r_3d, r_3d):
        z = dz + z0
        if not withinBounds(z, 2):
            continue    # out of bounds
        r_2d = int((r_3d**2 - dz**2)**.5)

        # y axis
        for dy in range(-r_2d, r_2d):
            y = dy + y0
            if not withinBounds(y, 1):
                continue     # out of bounds

            # x axis
            dx = int((r_2d**2 - dy**2)**.5)
            # for positive and negative x
            for k in [1,-1]:
                x = k*dx + x0
                if withinBounds(x, 0):
                    # all good, is a neighbor
                    coords_list.append((x, y, z))
    return coords_list
'''

#print(getNeighbors(0,0,0,4))

def getNeighbors(coords0):
    coords_list = []
    for dim in [0, 1, 2]:
        for direction in [-1, 1]:
            coords1 = list(coords0[:])
            coords1[dim] += direction
            if withinBounds(coords1):
                coords_list.append(coords1)
    return coords_list
#print(getNeighbors([0,0,0]))

# gets the distance btwn two points
def getDistance(coords_1, coords_2):
    d = 0
    for dim in range(len(coords_1)):
        d += (coords_1[dim] - coords_2[dim])**2
    return d**.5
