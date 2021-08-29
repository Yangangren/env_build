# coding=utf-8
"""Dynamic module of LasVSim

@Author: Xu Chenxiang
@Date: 2019.12.26
"""
from LasVSim.simulation_setting import *
from math import pi
from LasVSim.data_structures import *
from LasVSim.math_lib import degree_fix
import os


class VehicleDynamicModel(object):
    """
    Vehicle dynamic model for updating vehicle's state
    a interface with ctypes for the dynamic library
    compatible with both CVT and AMT controller model

    Attributes:
        __lasvsim_version(string): LasVSim版本. package, gui
        path(string): 动力学dll文件路径
        type: 动力学模型类型. CVT Car, AMT Car, Truck
        step_length(float): 仿真步长, s
        dll(DLL): C风格dll对象（动力学模型打包为dll供本模块调用）

        car_para(CarParameter obj): 储存动力学模型参数的C风格结构体
        ego_x: 自车x坐标，m（全局坐标系1）
        ego_y: 自车y坐标，m（全局坐标系1）
        heading: 自车偏航角，deg（全局坐标系1）
        ego_vel: 自车速度标量,m/s
        acc: 自车加速度标量，m/s^2
        engine_speed: 发动机转速，rpm
        drive_ratio: 传动比
        engine_torque: 发动机输出扭矩，Nm
        brake_pressure: 制动压力，Mpa
        steer_wheel: 方向盘转角，deg（顺时针为正）
        car_info: 储存自车状态量的C风格结构体
    """
    def __init__(self, x, y, heading, v, settings, step_length, car_parameter=None,
                 model_type=None):
        """
        动力学模块构造函数

        Args:
            x(float): m.(全局坐标系1)
            y(float): m.(全局坐标系1)
            heading(float): deg.(全局坐标系1)
            v(float): m/s.(全局坐标系1)
            settings(Settings obj): LasVSim的Settings类的一个实例
            step_length(float): 仿真步长，ms
            car_parameter(CarParameter obj): 储存动力学模型参数的C风格结构体
            model_type(string): 动力学模型类型. CVT Car, AMT Car, Truck
        """
        self.__lasvsim_version = 'gui'
        if model_type is None or model_type == 'CVT Car':
            self.path = CVT_MODEL_FILE_PATH
            self.type = 'CVT Car'
        elif model_type == 'AMT Car':
            self.path = AMT_MODEL_FILE_PATH
            self.type = model_type
        elif model_type == 'Truck':
            self.path = TRUCK_MODEL_FILE_PATH
            self.type = model_type
        elif model_type == 'EV':
            self.path = EV_MODEL_FILE_PATH
            self.type = model_type
        self.step_length = float(step_length)/1000

        self.car_para = car_parameter
        self.ego_x = x  # m
        self.ego_y = y  # m
        self.ego_heading = heading  # deg,全局坐标系1
        self.ego_vel = v  # m/s
        self.acc = 0.0  # m/s^2
        self.engine_speed = self.car_para.AV_ENGINE_IDLE / 30.0 * pi  # rpm
        self.drive_ratio = 1.0
        self.engine_torque = 0.0  # N.m
        self.brake_pressure = 0.0  # Mpa
        self.steer_wheel = 0.0  # deg
        self.car_info = VehicleInfo()  # 自车信息结构体
        self.pos_time = 0.0  # TODO(Chason): 跟当前仿真时间功能重复，后期删去

        module_path = os.path.dirname(__file__)
        self.dll = CDLL(module_path.replace('\\', '/') + '/' + self.path)
        self.dll.init(c_float(self.ego_x), c_float(self.ego_y),
                      c_float(self.ego_heading), c_float(self.ego_vel),
                      c_float(self.step_length), byref(self.car_para))

    def __del__(self):
        FreeLibrary(self.dll._handle)
        del self.dll

    def sim_step(self, EngTorque=None, BrakPressure=None, SteerWheel=None):
        if EngTorque is None:
            ET = c_float(self.engine_torque)
        else:
            ET = c_float(EngTorque)
            self.engine_torque = EngTorque
        if BrakPressure is None:
            BP = c_float(self.brake_pressure)
        else:
            BP = c_float(BrakPressure)
            self.brake_pressure = BrakPressure
        if SteerWheel is None:
            SW = c_float(self.steer_wheel)
        else:
            SW = c_float(SteerWheel)
            self.steer_wheel = SteerWheel
        #print('ET=',ET)
        #print('SW=',SW)
        x = c_float(0.0)
        y = c_float(0.0)
        heading = c_float(0.0)  # rad
        acc = c_float(0.0)
        v = c_float(0.0)
        r = c_float(0.0)
        i = c_float(0.0)
        road_info = RoadParameter()
        road_info.slope = 0.0
        self.dll.sim(byref(road_info), byref(ET), byref(BP), byref(SW),
                     byref(x), byref(y), byref(heading), byref(acc), byref(v),
                     byref(r), byref(i))
        heading.value = degree_fix(heading.value / pi * 180.0)  # TODO(Chason): 和package不统一，待检查
        (self.ego_x, self.ego_y, self.ego_vel, self.ego_heading, self.acc,
         self.engine_speed, self.drive_ratio) = (x.value, y.value, v.value,
                                                 heading.value, acc.value,
                                                 r.value, i.value)

    def get_pos(self):
        return (self.ego_x, self.ego_y, self.ego_vel, self.ego_heading, self.acc,
                self.engine_speed, self.drive_ratio)

    def set_control_input(self, eng_torque, brake_pressure, steer_wheel):
        self.engine_torque = eng_torque
        self.brake_pressure = brake_pressure
        self.steer_wheel = steer_wheel

    def get_info(self):
        self.dll.get_info(byref(self.car_info))
        #print('=======',self.car_info.Mileage)
        return (self.car_info.Steer_SW,
                self.car_info.Throttle,
                self.car_info.Bk_Pressure,
                self.car_info.Rgear_Tr,
                self.car_info.AV_Eng,
                self.car_info.M_EngOut,
                self.car_info.A,
                self.car_info.Beta / pi * 180.0,
                self.car_info.AV_Y / pi * 180.0,
                self.car_info.Vy,
                self.car_info.Vx,
                self.car_info.Steer_L1,
                self.car_info.StrAV_SW,
                self.car_info.Mfuel,
                self.car_info.Ax,
                self.car_info.Ay,
                self.car_info.Qfuel)

    def get_total_travelled_distance(self):
        return self.car_info.Mileage


