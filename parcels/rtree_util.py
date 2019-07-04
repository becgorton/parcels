#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:56:27 2019

@author: bec
"""

import numpy as np
import xarray as xr
import pyindex.core as core
import numba

@numba.njit(cache=True)
def get_relative_coordinates(lon, lat, x, y, xi, yi):
    '''returns relative coordinates xsi, eta
       that are the coordinates of the (x, y) point remapped into a square cell [0,1] x [0,1]
    '''
    invA = np.array([
        1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0,
        -1.0, 1.0, -1.0
    ]).reshape(4, 4)
    px = np.array(
        [lon[yi, xi], lon[yi, xi + 1], lon[yi + 1, xi + 1], lon[yi + 1, xi]])
    px = np.where(px[:] - x > 180, px - 360, px)
    px = np.where(px[:] - x < -180, px + 360, px)
    py = np.array(
        [lat[yi, xi], lat[yi, xi + 1], lat[yi + 1, xi + 1], lat[yi + 1, xi]])
    #print(type(invA))
    #print(type(px))

    a = np.dot(invA, px)
    b = np.dot(invA, py)

    aa = a[3] * b[2] - a[2] * b[3]
    bb = a[3] * b[0] - a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + x * b[
        3] - y * a[3]
    cc = a[1] * b[0] - a[0] * b[1] + x * b[1] - y * a[1]
    if abs(aa) < 1e-12:  # Rectilinear cell, or quasi
        eta = -cc / bb
    else:
        det2 = bb * bb - 4 * aa * cc
        if det2 > 0:  # so, if det is nan we keep the xsi, eta from previous iter
            det = np.sqrt(det2)
            eta = (-bb + det) / (2 * aa)
        else:  # should not happen, apart from singularities
            eta = 1e6
    if abs(a[1] + a[3] *
           eta) < 1e-12:  # this happens when recti cell rotated of 90deg
        xsi = ((y - py[0]) / (py[1] - py[0]) + (y - py[3]) /
               (py[2] - py[3])) * .5
    else:
        xsi = (x - a[0] - a[2] * eta) / (a[1] + a[3] * eta)
    return (xsi, eta)
    