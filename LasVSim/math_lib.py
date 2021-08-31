# coding=utf-8
"""
@author: Xu Chenxiang
@file: math_lib.py
@time: 2019/12/26 14:50
@file_desc: Basic math functions for LasVSim-both 0.2.1.191226_alpha
"""
import math
EPS = 1e-06


def degree_fix(degree):
    """将(-180°,180°]之外的角度转换到该范围内"""
    while degree > 180.0:
        degree -= 360.0
    while degree < -180.0 - EPS:  # ≤-180°
        degree += 360.0
    return degree
