# coding=utf-8
"""
@author: Xu Chenxiang
@file: controller_module.py
@time: 2019/12/26 17:22
@file_desc: Decision module for LasVSim-gui 0.2.1.191211_alpha
"""
import os
from ctypes import *
from _ctypes import FreeLibrary
from LasVSim import math_lib

DAGROUTER_MODEL_PATH = 'E:\Work\\research subject\Code\\2 Urbanroad Code' \
                       '\LasVSim\DecisionModelT_191213\\x64\Release\Decisio' \
                       'nModelT.dll'
# DAGROUTER_MODEL_PATH = 'E:\Work\\research subject\Code\\3 Highway Code' \
#                        '\LasVSim\DecisionModelT_191114\\x64\Release\DecisionModelT.dll'
# DAGROUTER_MODEL_PATH = 'E:\LasVSim Program\SourceCode\Decision' \
#                        '\Cpp_Trajectory Planner_Hybrid A star\Highway' \
#                        '\Decision_0.2.1_191202_alpha_XinLong\\x64\Release' \
#                        '\DecisionModel.dll'

_DAGROUTER_TASK_TURNLEFT = 2
_DAGROUTER_TASK_TURNRIGHT = 3
_DAGROUTER_TASK_GOSTRAIGHT = 1
_DAGROUTER_LIGHT_RED = 0
_DAGROUTER_LIGHT_GREEN = 1
_DAGROUTER_LIGHT_YELLOW = 2


class Planner(object):
    """
    DAG Route Plan Modal for Autonomous Car Simulation System
    a interface with ctypes for the dynamic library
    based on DAG roadnet model and a-star algorithm

    Attributes:
        __ego_length(float): 自车长度，m
        __ego_width(float): 自车宽度，m
        __step_length(float): 仿真步长,ms
        dll(DLL): C风格dll对象（决策算法打包为dll供本模块调用）
        __center(tuple): 当前十字路口中心坐标，m。(x, y)
        __direction(char): 车辆的行驶方向。'E', 'W', 'N', 'S'
        __current_time(float): 当前仿真时间，s

    """

    def __init__(self, step_length, path=None, settings=None):
        """
        决策类构造函数

        Args:
            step_length(float): 仿真步长,ms
            path(string): 决策算法dll文件路径
            settings(Settings obj): LasVSim的Settings类实例（包含当前仿真配置信息）
        """
        self.__lasvsim_version = 'gui'
        self.__step_length = step_length
        if path is None:
            module_path = os.path.dirname(__file__)
            self.dll = CDLL(module_path.replace('\\', '/') + '/' + DAGROUTER_MODEL_PATH)
        else:
            module_path = os.path.dirname(__file__)
            self.dll = CDLL(module_path.replace('\\', '/') + '/' + path)
        self.__ego_length = settings.car_length
        self.__ego_width = settings.car_width
        self.dll.init()

        self.__center = (0.0, 0.0)
        self.__direction = ''
        self.__current_time = 0.0

    def __del__(self):
        """
        决策类析构函数

        Returns:
            None

        """
        self.dll.delete_p()
        FreeLibrary(self.dll._handle)
        del self.dll

    def set_self_pos(self, pos):
        """
        为决策算法输入自车位姿信息

        Args:
            pos(tuple): 自车位姿信息。(x(m), y(m), v(m/s), heading(deg)) 全局坐标系1下

        Returns:
            None

        """
        x, y, v, heading = pos
        x, y, heading = self.__global2local(x, y, heading)
        self.dll.set_ego(c_float(x), c_float(y), c_float(v), c_float(heading))

    def set_other_vehicles(self, vehicles):
        """

        Args:
            vehicles(list): 保存周车数据的数组.[[car1_id, x(m), y(m), v(m/s), heading(deg), width(m), length(m)],...]
                周车位姿数据在全局坐标系1下

        Returns:
            None

        """
        for id, x, y, v, heading, width, length in vehicles:
            x1, y1, a1 = self.__global2local(x, y, heading)
            self.dll.set_vel(c_int(id), c_float(x1), c_float(y1), c_float(v),
                             c_float(length), c_float(width), c_float(a1))

    def __global2local(self, global1_x, global1_y, global1_heading):
        """
        将全局坐标系1转换到局部坐标系1下

        Args:
            global1_x(float): m
            global1_y(float): m
            global1_heading(float): deg

        Returns:
            local1_x(float): m,局部坐标系1
            local1_y(float): m,局部坐标系1
            local1_heading(float): rad,局部坐标系1

        """
        # TODO(Chason)：该部分后期应该交由决策算法自行完成
        xc, yc = self.__center
        if self.__direction is 'E':
            local1_x = global1_x - xc
            local1_y = global1_y - yc
            local1_heading = global1_heading
        elif self.__direction is 'W':
            local1_x = xc - global1_x
            local1_y = yc - global1_y
            local1_heading = global1_heading - 180.0
        elif self.__direction is 'N':
            local1_x = global1_y - yc
            local1_y = xc - global1_x
            local1_heading = global1_heading - 90.0
        else:
            local1_x = yc - global1_y
            local1_y = global1_x - xc
            local1_heading = global1_heading + 90.0
        local1_heading = math_lib.degree_fix(local1_heading)
        local1_heading = float(local1_heading)/180.0 * math_lib.math.pi
        return local1_x, local1_y, local1_heading

    def __local2global(self, local1_x, local1_y, local1_heading):
        """
        将局部坐标系1转换为全局坐标系2

        Args:
            local1_x(float): m
            local1_y(float): m
            local1_heading(float): deg

        Returns:
            global1_x(float): m,全局坐标系2
            global1_y(float): m,全局坐标系2
            global1_heading(float): deg,全局坐标系2

        """
        # TODO(Chason)：该部分后期应该交由决策算法自行完成
        local1_heading = local1_heading / math_lib.math.pi * 180.0
        xc, yc = self.__center
        # TODO(Chason): 后期gui版本和package版本统一
        if self.__lasvsim_version == 'pacakge':
            if self.direction is 'E':
                global1_x = local1_x + xc
                global1_y = local1_y + yc
                global1_heading = -(90.0 - local1_heading) + 90
            elif self.direction is 'W':
                global1_x= xc - local1_x
                global1_y= yc - local1_y
                global1_heading= -(270.0 - local1_heading) + 90
            elif self.direction is 'N':
                global1_x= xc - local1_y
                global1_y= local1_x + yc
                global1_heading= local1_heading + 90
            else:
                global1_x= local1_y + xc
                global1_y= yc - local1_x
                global1_heading= -(180.0 - local1_heading) + 90
            global1_heading = math_lib.degree_fix(global1_heading)
            return global1_x,global1_y,global1_heading
        else:
            if self.__direction is 'E':
                global2_x = local1_x + xc
                global2_y = local1_y + yc
                global2_heading = 90.0 - local1_heading
            elif self.__direction is 'W':
                global2_x = xc - local1_x
                global2_y = yc - local1_y
                global2_heading = 270.0 - local1_heading
            elif self.__direction is 'N':
                global2_x = xc - local1_y
                global2_y = local1_x + yc
                global2_heading = -local1_heading
            else:
                global2_x = local1_y + xc
                global2_y = yc - local1_x
                global2_heading = 180.0 - local1_heading
            global2_heading = math_lib.degree_fix(global2_heading)
            return global2_x, global2_y, global2_heading

    def plan(self, cross_center, direction, cross_task, ego_pos, other_vehicles,
             traffic_light, current_time):
        """
        决策单步更新函数

        Args:
            cross_center(tuple): 当前十字路口中心坐标。(x(m), y(m))
            direction(char): 车辆的行驶方向。'E', 'W', 'N', 'S'
            cross_task(char): 车辆在当前十字路口的驾驶任务. 'S', 'L', 'R'
            ego_pos(tuple): 自车位姿信息。(x(m), y(m), v(m/s), heading(deg)) 全局坐标系1下
            other_vehicles(list): 保存周车数据的数组.[[car1_id, x(m), y(m), v(m/s), heading(deg), width(m), length(m)],...]
                周车位姿数据在全局坐标系1下
            traffic_light(string): 当前十字路口信号灯状态（仅针对自车驾驶任务而言）. 'red' 'green' 'yellow'
            current_time(float): 当前仿真时间，s

        Returns:
            期望轨迹点数组
            例：
            [[t0, x0, y0, v0, heading0], [t1, x1, y1, v1, heading1]...] 全局坐标系2下

        """
        self.__direction = direction
        self.__center = cross_center
        task = _DAGROUTER_TASK_GOSTRAIGHT
        if cross_task is 'L':
            task = _DAGROUTER_TASK_TURNLEFT
        elif cross_task is 'R':
            task = _DAGROUTER_TASK_TURNRIGHT
        light_type = _DAGROUTER_LIGHT_GREEN
        if traffic_light is 'red':
            light_type = _DAGROUTER_LIGHT_RED
        self.__current_time = current_time
        self.set_self_pos(ego_pos)
        self.dll.clear_vel_list()
        self.set_other_vehicles(other_vehicles)
        self.dll.set_task(c_int(task))
        self.dll.set_trafficLight(c_int(light_type))
        self.dll.trajectory_plan()
        track = self.get_track(ego_pos)
        # fflag = c_float*(4)  # TODO(Chason)：调试决策代码用
        # fflag = fflag()  # TODO(Chason)：调试决策代码用
        # self.dll.testAPI(c_int(0), byref(fflag), c_bool(0))  # TODO(Chason)：调试决策代码用
        # print('From C++...', fflag[0], '   ',fflag[1], '   ',fflag[2],'...',fflag[3])  # TODO(Chason)：调试决策代码用
        return track

    def get_track(self, ego_pos):
        """
        从决策dll中获得当前期望轨迹

        Args:
            ego_pos(tuple): 自车位姿信息。(x(m), y(m), v(m/s), heading(deg)) 全局坐标系1下

        Returns:
            期望轨迹点数组
            例：
            [[t0, x0, y0, v0, heading0], [t1, x1, y1, v1, heading1]...] 全局坐标系2下

        """
        c_n = c_int(0)
        self.dll.get_total_num(byref(c_n))
        if c_n.value <= 0:
            return []
        arr = c_float*(c_n.value*4)
        data = arr()
        self.dll.get_optimal_path(byref(data))

        t = self.__current_time
        x, y, v, heading = ego_pos
        if self.__lasvsim_version == 'gui':  # TODO(Chason): 后期两个版本统一
            heading = -heading + 90  # 全局坐标系1转换到全局坐标系2下
        track = [(t, x, y, v, heading)]
        dt = 0.1

        for i in range(c_n.value):
            x2, y2, v2, heading2 = data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]
            x2, y2, heading2 = self.__local2global(x2, y2, heading2)
            x1 = (x+x2)/2
            y1 = (y+y2)/2
            v1 = (v+v2)/2
            heading1 = self.__get_prop_angle(heading, heading2, 0.5)
            t += dt
            track.append((t, x1, y1, v1, heading1))
            t += dt
            track.append((t, x2, y2, v2, heading2))
            x, y, v, heading = x2, y2, v2, heading2
        return track

    def __get_prop_angle(self, heading1, heading2, k):
        """
        get angle between a1 and a2 with proportion k
        unit: degree
        """
        while heading1-heading2 > 180:
            heading1 -= 360
        while heading2-heading1 > 180:
            heading2 -= 360
        a = heading1 * (1 - k) + heading2 * k
        a = a % 360
        if a > 180:
            a -= 360
        return a


if __name__ == "__main__":
    dll=CDLL('Modules/DecisionModel.dll')
    a = dll.clear_vel_list()
    print(a)
    FreeLibrary(dll._handle)
    del dll
