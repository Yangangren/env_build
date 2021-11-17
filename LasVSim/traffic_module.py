# coding=utf-8
"""
@author: Chason Xu
@file: traffic_model.py
@time: 2019.12.26 14:00
@file_desc: traffic module for LasVSim-package 0.2.1.191211_alpha
"""
import math
import optparse
import os
import sys
import traci
import copy
from LasVSim.data_structures import *
from LasVSim.math_lib import degree_fix
import logging
import sumolib
from traci.exceptions import FatalTraCIError
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory "
        "of your sumo installation (it should contain folders 'bin', 'tools' "
        "and 'docs')")

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
VEHICLE_COUNT = 0  # 总车数，包括自车，会随不同交通流类型而改变
WINKER_PERIOD=0.5  # 转向灯闪烁周期，s
_TLS={'6': 'gneJ7', '7': 'gneJ4', '8': 'gneJ8', '11': 'gneJ0', '12': 'gneJ1', '13': 'gneJ2',
     '16':'gneJ6','17':'gneJ3','18':'gneJ9', '0':'gneJ9','1':'gneJ9',
     '2':'gneJ9','3':'gneJ9','4':'gneJ9','5':'gneJ9','9':'gneJ9','10':'gneJ9',
     '14':'gneJ9','15':'gneJ9', '19':'gneJ9','20':'gneJ9','21':'gneJ9',
     '22':'gneJ9','23':'gneJ9','24':'gneJ9'}  # Urban Road地图信号灯id
MAX_TRAFFIC = 400  # 交通流的最大数量

class VehicleModelsReader(object):  # 该部分可直接与gui相替换
    """用于读取不同车辆模型信息文件的类

        LasVSim所用到的所有车辆的模型信息保存在
        Library/vehicle_model_library.csv里，该模块
        用于从该文件中获取每种车辆的模型信息。

        Attributes:
            __type_array(tuple(int)): 车辆类型编号元组.
            __lasvsim_version(string): 代码版本类型. 'gui' or 'package'
            __info(dict(int: tuple)): 保存各类型车辆模型信息的字典

    """

    def __init__(self, model_path):
        """
                VehicleModelsReader类构造函数

                Args:
                    model_path(string): 文件路径
                """
        self.__type_array = [0, 1, 2, 3, 7, 100, 1000, 200, 2000]
        self.__lasvsim_version = 'package'
        self.__info = dict()
        self.__read_file(model_path)

    def __read_file(self, file_path):
        """读取车辆模型信息保存文件

        Args:
            file_path(string): 文件路径

        Returns:

        """
        with open(file_path) as f:
            line = f.readline()
            line = f.readline()
            while len(line)>0:
                data = line.split(',')
                type = int(data[1])
                if type not in self.__type_array:
                    line=f.readline()
                    continue
                length=float(data[7])
                width=float(data[8])
                x=float(data[4])  # 车一侧到原点的距离
                y=float(data[2])  # 车头到原点的距离
                if self.__lasvsim_version == 'package':
                    img_path = []
                else:
                    img_path = 'Resources/Rendering/%d.png' % type
                self.__info[type] = (length, width, x, y, img_path)
                line = f.readline()

    def get_types(self):
        """
        返回保存车辆类型编号的元组

        Returns:
            保存车辆类型编号的元组
        """
        return self.__type_array

    def get_vehicle(self,type):
        """
        返回输入车辆类型的长、宽、车一侧到原点的距离、车头到原点的距离和渲染图片
        的保存路径。

        Args:
            type(int): 车辆类型编号

        Returns:
            车辆参数元组。
            例：
            (length(m), width(m), center_to_side(m), center_to_head(m),
            render_image_path(string))

        """
        if type not in self.__info:
            type=0
        return self.__info[type]

HISTORY_TRAFFIC_SETTING = ['Traffic Type', 'Traffic Density',
                           'Map']  # 上次仿真的交通流配置
RANDOM_TRAFFIC = {}  # 随机交通流分布信息


class Traffic(object):
    """
    LasVSim的交通流模块，用于与sumo间进行数据交互.

    Attributes:
        __own_x(float): 自车x坐标，m（全局坐标系1）
        __own_y(float): 自车y坐标，m（全局坐标系1）
        __own_v(float): 自车速度标量，m/s（全局坐标系1）
        __own_a(float): 自车偏航角，deg(gui)（全局坐标系1）
        __own_lane_pos: 自车距离当前车道停止线的距离,m
        __own_lane_speed_limit: 自车当前车道限速, m/s
        __map_type(string): 地图类型
        __path(string): sumo仿真文件路径
        __step_length(string): 仿真步长,s
        random_traffic: 初始交通流分布数据.
        vehicleName: 交通流id列表.
        traffic_change_flag: 交通流配置改变标志位（True代表交通流需要重新初
            始化）
        veh_type: 交通流车辆类型字典
        vehicles: 交通流数据
        sim_time: 仿真计步数
        type: 交通流类型
        density: 交通流密度
        veh_dict: 储存当前周车name和id的dict，例：{'ego_car',0}
        veh_name: 储存当前周车name的list
        veh_id: 储存当前周车id的list
        veh_name_mid: 储存周车name的list
        veh_name_enter: 储存当前新进入仿真区域的周车name
        veh_name_exit: 储存当前离开仿真区域的周车name
        index_mid: 中间计数器
        id_mid: id中间量
        free_id_count: 未被使用的id个数
        veh_info_dict: sumo返回的交通流数据
        car_x: 周车中心x坐标,m
        car_y: 周车中心y坐标,m
        car_heading: 周车偏航角,deg（平台坐标系）
        traffic_range: sumo返回的交通流范围（以自车为中心，超过这个范围的交
            通流仍在sumo中运行，但不会传递给LasVSim）
        lasvsim_version: LasVSim的版本（'gui' 和 'package'）
        ego_length: 自车长度，m
        ego_width: 自车宽度，m
        current_edge: 自车当前所在道路的id
        current_traffic_density: 自车当前所在车道的交通密度
    """

    def __init__(self, step_length, settings, map_type=None, traffic_type=None,
                 traffic_density=None, simu_method=False,init_traffic=None, isLearner=None):
        """
        交通流类的构造函数

        Args:
            step_length(float): 仿真步长，ms
            map_type(string): 地图类型，(Map1_Urban Road, Map2_Highway)
            traffic_type(string): 交通流类型，（No Traffic, Mixed Traffic, Vehicle Only Traffic）
            traffic_density(string): 交通流密度，（Sparse，Middle，Dense）
            init_traffic(dict): 初始交通流数据，{car1_id: {var1_id: value, var2_id: value...}...} （全局坐标系2下）
            settings(Settings obj): LasVSim的Settings类的一个实例
        """
        # <editor-fold desc="变量及参数初始化">
        self.__own_x = 0.0
        self.__own_y = 0.0
        self.__own_v = 0.0
        self.__own_a = 0.0
        self.__own_lane_pos = float(999.0)  # 自车距离当前车道停止线的距离,m
        self.__own_lane_speed_limit = float(999.0)  # 自车当前车道限速, m/s
        self.traffic_change_flag = True
        self.veh_type = {'car_1': 0, 'car_2': 1, 'car_3': 2, 'truck_1': 100,
                         'motorbike_1': 200, 'motorbike_2': 200,
                         'motorbike_3': 200}
        #self.vehicles = [{}] * MAX_TRAFFIC
        self.sim_time = 0.0
        self.veh_dict = {}
        self.veh_name = []
        self.veh_id = []
        self.veh_name_mid = []
        self.veh_name_enter = []
        self.veh_name_exit = []
        self.index_mid = 0
        self.id_mid = []
        self.free_id_count = 0
        self.veh_info_dict = {}
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_heading = 0.0
        self.traffic_range = 0.0
        self.lasvsim_version = 'package'
        self.current_edge = ''
        self.current_traffic_density = -1
        # </editor-fold>

        # <editor-fold desc="读取仿真配置">
        self.__map_type = map_type
        self.density = traffic_density
        if simu_method == False:
            direction = self.density
        else:
            direction = self.density +'_'+str(simu_method)
        if self.lasvsim_version == 'package':
            module_path = os.path.dirname(__file__)
            self.__path = module_path.replace('\\', '/') + "/Map/" + map_type + "/"+direction+"/"
        else:
            self.__path = "Map/" + map_type + "/"+ "/"+direction+"/"
        self.type = traffic_type  # For example: Normal

        if self.density=="simu":
            self.long_dist_block = 30
            self.lat_dist_block = 3.75
            self.step_length = 600
        elif self.density=='100':
            self.long_dist_block = 8
            self.lat_dist_block = 2
            self.step_length = 400
        else:
            self.long_dist_block = 8
            self.lat_dist_block = 2
            self.step_length = 200
        self.__step_length = str(float(step_length)/1000)
        self.ego_length = settings.car_length
        self.ego_width = settings.car_width
        # </editor-fold>

        # if isLearner:
        #     self.random_traffic = self.__generate_random_traffic()
        # else:
        #     global VEHICLE_COUNT, HISTORY_TRAFFIC_SETTING, RANDOM_TRAFFIC
        #     if traffic_type == 'No Traffic':
        #         VEHICLE_COUNT = 1
        #         self.vehicleName = ['ego']
        #         self.random_traffic = {}
        #     else:
        #         if init_traffic is not None:  # 载入仿真项目时已有初始交通流分布数据
        #             self.random_traffic = init_traffic
        #             RANDOM_TRAFFIC = copy.deepcopy(self.random_traffic)
        #             HISTORY_TRAFFIC_SETTING = [traffic_type, traffic_density, map_type]
        #             self.traffic_change_flag = False
        #         elif (HISTORY_TRAFFIC_SETTING[0] != traffic_type or
        #                 HISTORY_TRAFFIC_SETTING[1] != traffic_density or
        #                 HISTORY_TRAFFIC_SETTING[2] != map_type):
        #             self.random_traffic = self.__generate_random_traffic()
        #             # RANDOM_TRAFFIC = copy.deepcopy(self.random_traffic)
        #             # HISTORY_TRAFFIC_SETTING = [traffic_type, traffic_density, map_type]
        #             self.traffic_change_flag = True
        #         else:
        #             # 本次仿真的交通流配置与上次仿真一致则不用重新初始化随机交通流
        #             self.random_traffic = RANDOM_TRAFFIC
        #             self.traffic_change_flag = False
        #
        #         self.vehicleName = ['ego'] + list(self.random_traffic.keys())
        #         VEHICLE_COUNT = len(self.vehicleName)
        #     VEHICLE_COUNT = 0  # TODO(Chason): VEHICLE_COUNT表示仿真每一步从sumo获得的周车数量，其值会变，因此此处给值无效，后期检查后应该删去

    def __del__(self):
        """
        交通流类的析构函数，主要用于关闭sumo的traci接口

        Returns:
            None

        """
        traci.close()

    def init(self, ego_position):
        """交通流模块初始化函数

        Args:
            ego_position(list): 自车位姿数据. [x, y, v, a](m, m, m/s, deg)
            （package版在全局坐标系1下，gui版在全局坐标系2下）

        Returns:
              None

        """
        if self.lasvsim_version == "package":
            SUMO_BINARY =checkBinary('sumo') #checkBinary('sumo-gui')#
        else:
            SUMO_BINARY = checkBinary('sumo-gui')
        if self.__map_type == MAPS[0]:  # 城市道路场景为封闭交通流
            traci.start(
                [SUMO_BINARY, "-c", self.__path + "configuration.sumocfg",
                 "--step-length", self.__step_length,
                 "--lateral-resolution", "0.375", "--start", "--no-warnings","--no-step-log",
                 "--random", "false",
                 "--seed", "711",])
        elif self.__map_type == MAPS[1] or self.__map_type == MAPS[2]:  # 高速公路场景为开放交通流
            if self.type == 'No Traffic':
                traci.start([SUMO_BINARY, "-c",
                             self.__path+"configuration.sumocfg",
                             "--step-length", self.__step_length,
                             "--quit-on-end",
                             "--no-warnings",
                             "--no-step-log",
                             ])
            else:
                try:
                    traci.start([SUMO_BINARY, "-c",
                                 self.__path+"traffic_generation_"+self.type+".sumocfg",
                                 "--step-length", "0.25",
                                 #"--quit-on-end",
                                 "--lateral-resolution", "0.375",
                                 "--random",
                                 "--no-warnings",
                                 "--no-step-log",
                                 ])
                except FatalTraCIError:
                    print('Retry by other port')
                    port = sumolib.miscutils.getFreeSocketPort()
                    traci.start([SUMO_BINARY, "-c",
                                 self.__path+"traffic_generation_"+self.type+".sumocfg",
                                 "--step-length", "0.25",
                                 #"--quit-on-end",
                                 "--lateral-resolution", "0.375",
                                 "--random",
                                 "--no-warnings",
                                 "--no-step-log",
                                 ],port=port)

        else:
            raise Exception(self.__map_type + ' 该地图还未开放')

        traci.junction.subscribeContext(
            objectID='Jun_0', domain=traci.constants.CMD_GET_VEHICLE_VARIABLE,
            dist=500000.0, varIDs=[traci.constants.VAR_POSITION,
                                   traci.constants.VAR_ANGLE,
                                   traci.constants.VAR_TYPE,
                                   traci.constants.VAR_SPEED,
                                   traci.constants.VAR_LENGTH,
                                   traci.constants.VAR_WIDTH,
                                   traci.constants.VAR_LANE_INDEX,
                                   traci.constants.VAR_ROAD_ID,
                                   ],
            begin=0.0, end=2147483647.0)
        # while traci.simulation.getArrivedNumber() < 1:
        #     traci.simulationStep()
        time_index = 0
        while time_index < self.step_length:
            traci.simulationStep()
            time_index += 1
        self.random_traffic = traci.junction.getContextSubscriptionResults('Jun_0')

        traci.close()

        try:
            traci.start([SUMO_BINARY, "-c",
                         self.__path + "traffic_generation_" + self.type + ".sumocfg",
                         "--step-length", self.__step_length,
                         # "--quit-on-end",
                         "--lateral-resolution", "0.375",
                         "--random",
                         "--no-warnings",
                         "--no-step-log",
                         ])
        except FatalTraCIError:
            print('Retry by other port')
            port = sumolib.miscutils.getFreeSocketPort()
            traci.start([SUMO_BINARY, "-c",
                         self.__path + "traffic_generation_" + self.type + ".sumocfg",
                         "--step-length", self.__step_length,
                         # "--quit-on-end",
                         "--lateral-resolution", "0.375",
                         "--random",
                         "--no-warnings",
                         "--no-step-log",
                         ], port=port)

        # 在sumo的交通流模型中插入自车,通过subscribeContext抓取交通流数据
        self.__own_x, self.__own_y, self.__own_v, self.__own_a = ego_position
        if self.lasvsim_version == 'gui':  # TODO(Chason): gui版本传入位姿在全局坐标系2下，后期应统一改为全局坐标系1
            self.__own_a = -self.__own_a + 90.0

        traci.vehicle.addLegacy(vehID='ego', routeID='self_route',
                                depart=0, pos=0, lane=-6, speed=0,
                                typeID='self_car')


        traci.vehicle.subscribeContext(objectID='ego',
                                       domain=traci.constants.CMD_GET_VEHICLE_VARIABLE,
                                       dist=SUMO_RETURN_RANGE,
                                       varIDs=[traci.constants.VAR_POSITION,
                                        traci.constants.VAR_LENGTH,
                                        traci.constants.VAR_WIDTH,
                                        traci.constants.VAR_ANGLE,
                                        traci.constants.VAR_SIGNALS,
                                        traci.constants.VAR_SPEED,
                                        traci.constants.VAR_TYPE,
                                        traci.constants.VAR_EMERGENCY_DECEL,
                                        traci.constants.VAR_LANE_INDEX,
                                        traci.constants.VAR_LANEPOSITION,
                                        traci.constants.VAR_ROAD_ID,
                                        traci.constants.VAR_LANEPOSITION_LAT,
                                        traci.constants.VAR_ALLOWED_SPEED,],
                                       begin=0, end=2147483647)

        # 设置自车长宽及初始位姿
        # traci.vehicle.setLength('ego', self.ego_length)  # Sumo function
        # traci.vehicle.setWidth('ego', self.ego_width)  # Sumo function
        traci.vehicle.moveToXY(vehID='ego', edgeID='Route_0',lane=-6, x=self.__own_x +
                               self.ego_length / 2 *
                               math.cos(math.radians(self.__own_a)),
                               y= self.__own_y + self.ego_length / 2 *math.sin(math.radians(self.__own_a)),
                               angle=-self.__own_a + 90, keepRoute=0)

        # traci.vehicle.moveToXY(vehID='ego', edgeID='Nan',lane=-6, x=31836,
        #                        y= 9223,
        #                        angle=-self.__own_a + 90, keepRoute=0)


        # 无法通过getContextSubscriptionResults获取route信息，但需要
        # 每辆车的route信息来初始化交通流，因此加入getRoute来获取每
        # 辆车的route。TODO(Chason): 该bug可改正
        self.__initiate_traffic()
        traci.simulationStep()
        #_logger.info('random traffic initialized')

    def get_vehicles(self):
        """Get all vehicles' information including ego vehicle.

        Args:
            veh_info(list): 储存周车数据的C风格结构体数组（全局坐标系1下）

        Returns:
            A list containing all vehicle's current state except ego vehicle.
            For example:

            [{'type':0, 'x':0.0(m), 'y': 0.0(m), 'v':0.0(m/s), 'angle': 0.0(deg),
            'rotation': 0, 'winker': 0, 'winker_time': 0(s), 'render': False},
            {'type':0, 'x':0.0, 'y': 0.0, 'v':0.0, 'angle': 0.0,
            'rotation': 0, 'winker': 0, 'winker_time': 0, 'render': False},...]
            （全局坐标系1下）

        """
        # 通过getContextSubscriptionResults函数获取sumo中的交通流数据
        self.veh_info_dict = traci.vehicle.getContextSubscriptionResults('ego')


        # 更新自车所在位置信息（当前车道的限速，距离当前车道停止线的距离）
        self.veh_info_dict['ego'][traci.constants.VAR_SIGNALS]=0b0000
        if self.__map_type == MAPS[0]:
            self.__own_lane_speed_limit = [5.56, 5.56, 16.67, 16.67][
                self.veh_info_dict['ego'][traci.constants.VAR_LANE_INDEX]]
            if self.veh_info_dict['ego'][traci.constants.VAR_LANE_INDEX] in [2, 3]:
                self.__own_lane_pos = 588.0 - self.veh_info_dict['ego'][
                    traci.constants.VAR_LANEPOSITION]
            else:
                self.__own_lane_pos = 9999.9
        elif self.__map_type == MAPS[1] or  self.__map_type == MAPS[2]:
            self.__own_lane_speed_limit = [27.78, 27.78,33.33, 33.33][
                self.veh_info_dict['ego'][traci.constants.VAR_LANE_INDEX]]
            self.__own_lane_pos = 9999.9
        else:
            raise Exception(self.__map_type + ' 该地图还未开放')
        # 将周车列表转换为平台使用的数据格式

        del self.veh_info_dict['ego']  # 删去自车信息
        #self.__update_veh_list(self.veh_info_dict)
        self.__get_other_car_info(self.veh_info_dict)
        return self.vehicles

    def get_vehicle_num(self):
        """
        返回自车周围一定范围内的交通流数量

        Returns:
              int

        """
        return VEHICLE_COUNT

    def get_light_status(self):
        """
        返回当前车道信号灯状态

        Returns:
            包含横纵向信号灯状态的字典。
            例：
            {'red', 'yellow'}(横向信号扽， 纵向信号灯)
        """
        index = self.__getcenterindex(self.__own_x, self.__own_y)
        if self.__map_type == MAPS[0]:
            traffic_light = traci.trafficlight.getPhase(_TLS[str(int(index))])
        elif self.__map_type == MAPS[1]:
            traffic_light = 0
        else:
            raise Exception(self.__map_type + ' 该地图还未开放')
        if traffic_light == 0:
            h = 1
            v = 0
        else:
            h = 0
            v = 1
        s = ['red', 'green', 'yellow']
        return dict(h=s[h], v=s[v])

    def get_light_values(self):
        """Get current intersection's traffic light state.

        Only indicating allowed or disallowed.

        Returns:
            Two int variables indicating right of way of two directions.
            Variable 'h' indicating horizontal direction's right of way.
            Variable 'v' indicating vertical direction's right of way.
        """
        index = self.__getcenterindex(self.__own_x, self.__own_y)
        if self.__map_type == MAPS[0]:
            traffic_light = traci.trafficlight.getPhase(_TLS[str(int(index))])
        elif self.__map_type == MAPS[1]:
            traffic_light = 0
        else:
            raise Exception(self.__map_type + ' 该地图还未开放')
        if traffic_light == 0:
            h = 1
            v = 0
        else:
            h = 0
            v = 1
        return h, v

    def get_traffic_density(self):
        """返回当前车道上的车流数量

            利用sumo.traci返回自车当前所处车道的周车数量

        Returns:
            int

        """
        self.current_edge = traci.vehicle.getRoadID('ego')
        if self.__map_type == MAPS[0]:
            if ':' in self.current_edge:  # 十字路口内交通密度不统计
                self.current_traffic_density = -1
            else:
                self.current_traffic_density = (
                    traci.edge.getLastStepVehicleNumber(
                        self.current_edge) / 1.176
                )
        elif self.__map_type == MAPS[1] or  self.__map_type == MAPS[2]:
            self.current_traffic_density = (
                traci.edge.getLastStepVehicleNumber(self.current_edge) / 3.688
            )
        else:
            raise Exception(self.__map_type + ' 该地图还未开放')
        return self.current_traffic_density

    def sim_step(self):
        """
        交通流模块单步更新函数

        Returns:
              None

        """
        self.sim_time += float(self.__step_length)
        traci.simulationStep()


    def set_own_car(self, x, y, v, a):
        """将自车的位姿数据传入sumo

        Args:
            x: Ego vehicle's current x coordination of it's shape center, m.（全局坐标系1）
            y: Ego vehicle's current y coordination of it's shape center, m.（全局坐标系1）
            v: Ego vehicle's current velocity, m/s.（全局坐标系1）
            a: Ego vehicle's current heading angle under base coordinate, deg.（全局坐标系1）
        """
        self.__own_x, self.__own_y, self.__own_v, self.__own_a = x, y, v, a
        traci.vehicle.moveToXY(vehID='ego',edgeID='Route_0', lane=-6, x=self.__own_x +
                               self.ego_length / 2 *
                               math.cos(math.radians(self.__own_a)),
                               y=self.__own_y + self.ego_length / 2 *math.sin(math.radians(self.__own_a)),
                               angle=-self.__own_a + 90.0, keepRoute =0)

        # traci.vehicle.moveToXY(vehID='ego', edgeID='Nan',lane=-6, x=31836,
        #                        y= 9223,
        #                        angle=-self.__own_a + 90, keepRoute=0)
    def get_current_lane_speed_limit(self):
        """
        获取当前车道的限速

        Returns:
            m/s

        """
        return self.__own_lane_speed_limit

    def get_current_distance_to_stopline(self):
        """
        获取自车距离当前十字路口停止线的距离，进入十字路口后该值为负

        Returns:
              float(m)

        """
        return self.__own_lane_pos

    def get_dis_to_center_line(self):
        """
        获取自车与当前车道中心线之间的偏离距离

        Returns:
              float(m)

        """


        return traci.vehicle.getLateralLanePosition('ego')

    def get_egolane_index(self):  # 此处与gui不同 左正右负
        return traci.vehicle.getLaneIndex('ego')

    def get_road_related_info_of_ego(self):
        egolane_index = traci.vehicle.getLaneIndex('ego')
        dis2center_line = traci.vehicle.getLateralLanePosition('ego')
        own_lane_speed_limit = self.get_current_lane_speed_limit()
        #current_distance_to_stopline = self.get_current_distance_to_stopline()
        #vertical_light_value = self.get_light_values()[1]
        return dict(dist2current_lane_center=dis2center_line,
                    egolane_index=egolane_index,
                    current_lane_speed_limit=own_lane_speed_limit,
                    RoadID = traci.vehicle.getRoadID('ego'),
                    )

    def __generate_random_traffic(self):
        """
        生成仿真初始时刻的随机交通流

        Returns:
            保存初始交通流数据的字典
            例：
            {car1_id: {var1_id: value, var2_id: value...}...} (全局坐标系2下)

        """
        #  调用sumo
        SUMO_BINARY = checkBinary('sumo-gui')
        traci.start([SUMO_BINARY, "-c",
                     self.__path + "traffic_generation_" + self.type + ".sumocfg",
                     "--step-length", "1",
                     "--random",
                     "--no-warnings",
                     "--no-step-log",
                     ])
        if self.__map_type == MAPS[0]:
            #  等待所有车辆都进入路网
            #  如果1000步之内都没有新的交通流进入路网，则认为交通流已经全部插入
            # 一旦检测当有新的交通流进入路网，则重新开始记数
            # 当交通流全部插入后，让交通流跑一段时间以实现均匀分布
            self.__add_self_car()
            time_index = 0
            while time_index < 1000:
                traci.simulationStep()
                if traci.simulation.getDepartedNumber() > 0:
                    time_index = 0
                else:
                    time_index += 1
            if self.density == 'Sparse':
                time_index = 100
            else:
                time_index = 1000
            while time_index > 0:
                traci.simulationStep()
                time_index -= 1
            random_traffic = traci.vehicle.getContextSubscriptionResults('ego')
            for veh in random_traffic:
                # 无法通过getContextSubscriptionResults获取route信息，但需要
                # 每辆车的route信息来初始化交通流，因此加入getRoute来获取每
                # 辆车的route。TODO(Chason): 该bug可改正
                random_traffic[veh][87] = traci.vehicle.getRoute(vehID=veh)
            # getContextSubscriptionResults返回的车辆同时包括自车，需要删去。
            del random_traffic['ego']
        elif self.__map_type == MAPS[1] or MAPS[5]:
            # 当第一辆车跑完路网的时候开始计时，取500步以后的交通流分布作为仿真
            # 的初始交通流分布保存下来
            # TODO：高速公路采用junction获取交通流数据，因此不需要插入自车,，后期应该将Urban地图也改为同样的方式
            traci.junction.subscribeContext(
                objectID='10', domain=traci.constants.CMD_GET_VEHICLE_VARIABLE,
                dist=300000.0, varIDs=[traci.constants.VAR_POSITION,
                                       traci.constants.VAR_ANGLE,
                                       traci.constants.VAR_TYPE,
                                       traci.constants.VAR_SPEED,
                                       traci.constants.VAR_LENGTH,
                                       traci.constants.VAR_WIDTH],
                begin=0.0, end=2147483647.0)
            # while traci.simulation.getArrivedNumber() < 1:
            #     traci.simulationStep()
            time_index = 0
            while time_index < 10:
                traci.simulationStep()
                time_index += 1
            random_traffic = traci.junction.getContextSubscriptionResults('10')
            # 无法通过getContextSubscriptionResults获取route信息，但需要
            # 每辆车的route信息来初始化交通流，因此加入getRoute来获取每
            # 辆车的route。TODO(Chason): 该bug可改正
            for veh in random_traffic:
                random_traffic[veh][87] = traci.vehicle.getRoute(vehID=veh)
        else:
            raise Exception(self.__map_type + ' 该地图还未开放')
        traci.close()  # 获得初始交通流后即可关闭sumo
        _logger.info('random traffic generated')
        return random_traffic

    def __initiate_traffic(self):
        """
        初始化交通流

        Returns:
              None

        """
        if self.__map_type == MAPS[0]:
            init_speed = 0.0  # TODO(Chason): 后期改正，交通流的初始速度待定
        elif self.__map_type == MAPS[1] or self.__map_type == MAPS[2]:
            init_speed = 20.0  # TODO(Chason): 后期改正，交通流的初始速度待定
        else:
            raise Exception(self.__map_type + ' 该地图还未开放')
        for veh in self.random_traffic:
            # Skip traffic vehicle which overlap with ego vehicle.
            if self.__map_type == MAPS[0]:
                if (math.fabs((self.random_traffic[veh]
                               [traci.constants.VAR_POSITION][0])
                              - self.__own_x) < 20
                    and (math.fabs((self.random_traffic[veh]
                                    [traci.constants.VAR_POSITION][1])
                                   - self.__own_y) < 1.7)):
                    traci._vehicle.remove(vehID=veh)
            # 高速下的重叠判断范围更大一些
            elif self.__map_type == MAPS[1] or self.__map_type == MAPS[2]:
                if (math.fabs((self.random_traffic[veh]
                               [traci.constants.VAR_POSITION][0])
                              - self.__own_x) < self.long_dist_block
                    and (math.fabs((self.random_traffic[veh]
                                    [traci.constants.VAR_POSITION][1])
                                   - self.__own_y) < self.lat_dist_block)):

                    #traci.vehicle.remove(vehID=veh)
                    continue
            else:
                raise Exception(self.__map_type + '  该地图还未开放')

            traci.vehicle.addLegacy(vehID=veh+'init',
                                    routeID="self_route",
                                    depart=0,
                                    pos=0,
                                    lane=self.random_traffic[veh][traci.constants.VAR_LANE_INDEX],
                                    speed=self.random_traffic[veh][traci.constants.VAR_SPEED],
                                    typeID=(self.random_traffic[veh]
                                            [traci.constants.VAR_TYPE]))
            # traci.vehicle.setRoute(vehID=veh,
            #                        edgeList=self.random_traffic[veh][traci.constants.VAR_ROAD_ID])

            traci.vehicle.moveToXY(vehID=veh+'init',
                                   edgeID=self.random_traffic[veh][traci.constants.VAR_ROAD_ID],
                                   lane=self.random_traffic[veh][traci.constants.VAR_LANE_INDEX],
                                   x=(self.random_traffic[veh]
                                      [traci.constants.VAR_POSITION][0]),
                                   y=(self.random_traffic[veh]
                                      [traci.constants.VAR_POSITION][1]),
                                   angle=(self.random_traffic[veh]
                                          [traci.constants.VAR_ANGLE]),
                                   keepRoute=2)

    def __add_self_car(self):
        """
        任意插入一辆车用于获取初始交通流数据

        Returns:
              None

        """
        traci.vehicle.addLegacy(vehID='ego', routeID='Nan',
                                depart=0, pos=0, lane=-6, speed=0,
                                typeID='self_car')
        traci.vehicle.subscribeContext('ego',
                                       traci.constants.CMD_GET_VEHICLE_VARIABLE,
                                       300000, [traci.constants.VAR_POSITION,
                                                traci.constants.VAR_ANGLE,
                                                traci.constants.VAR_TYPE,
                                                traci.constants.VAR_SPEED,
                                                traci.constants.VAR_LENGTH,
                                                traci.constants.VAR_WIDTH],
                                       0, 2147483647)

    def __update_veh_list(self, sumo_vehs_dict):
        """
        更新自车周围一定范围内的周车的id列表

        Args:
            sumo_vehs_dict(dict): 由getContextSubscriptionResults命令直接返回的储存周车数据的字典.{car1_id: {var1_id: value, var2_id: value...}...} （全局坐标系2）

        Returns:
              None

        """
        global VEHICLE_COUNT

        self.veh_name_enter = []
        self.veh_name_exit = []

        for veh in self.veh_dict:
            if veh not in sumo_vehs_dict:  # 该车跑出仿真范围
                self.veh_name_exit.append(veh)
        for veh in sumo_vehs_dict:
            if veh not in self.veh_dict:
                self.veh_name_enter.append(veh)  # 该车刚进入仿真范围

        if len(self.veh_name_enter) > len(
                self.veh_name_exit):  # 进入的车数比离开的车数多，周车列表需要增加长度
            self.index_mid = len(self.veh_name_exit)
            for i in range(self.index_mid):  # 将离开的车的id赋给进入的车
                self.veh_dict[self.veh_name_enter[i]] = self.veh_dict[
                    self.veh_name_exit[i]]
                del self.veh_dict[self.veh_name_exit[i]]
            for i in range(self.index_mid,
                           len(self.veh_name_enter)):  # 对剩余进入的车赋给新的id
                if len(self.id_mid) > 0:  # 优先将未被占用的id赋给新的车辆
                    self.veh_dict[self.veh_name_enter[i]] = self.id_mid[-1]
                    self.id_mid.pop()
                    VEHICLE_COUNT += 1
                else:  # 增加周车列表的长度
                    self.veh_dict[self.veh_name_enter[i]] = VEHICLE_COUNT
                    VEHICLE_COUNT += 1
        else:  # 离开的车数比进入的车数多，周车列表需要减少长度
            self.index_mid = len(self.veh_name_enter)
            for i in range(self.index_mid):  # 将离开的车的id赋给进入的车

                self.veh_dict[self.veh_name_enter[i]] = self.veh_dict[
                    self.veh_name_exit[i]]
                del self.veh_dict[self.veh_name_exit[i]]
            for i in range(self.index_mid, len(self.veh_name_exit)):  # 删去剩余离开的车
                self.id_mid.append(self.veh_dict[self.veh_name_exit[i]])
                del self.veh_dict[self.veh_name_exit[i]]
                VEHICLE_COUNT -= 1

    def __get_other_car_info(self, othercar_dict):
        """
        将sumo返回的周车数据转换为平台的数据格式

        Args:
            othercar_dict(dict): 由getContextSubscriptionResults命令直接返回的储存周车数据的字典.{car1_id: {var1_id: value, var2_id: value...}...} （全局坐标系2）
            veh_dict(dict): 储存当前周车name和id的dict，例：{'ego_car':0}
            veh_info(list): C风格结构体数组，用于储存周车数据（目前功能与self.vehicles重复，后期平台全部通信数据全部改为C风格结构体）（全局坐标系1）

        Returns:
              None

        """
        self.vehicles =dict()

        # for i in range(VEHICLE_COUNT):
        #     self.vehicles[i]['render'] = False
        #     veh_info[i].render_flag = False
        index=0
        for veh in othercar_dict:
            self.car_x, self.car_y = othercar_dict[veh][
                traci.constants.VAR_POSITION]
            self.car_heading = degree_fix(-othercar_dict[veh][
                traci.constants.VAR_ANGLE] + 90.0)  # 转换到平台坐标系下

            # sumo中车辆的位置由车辆车头中心表示，
            # 因此要计算根据sumo给的坐标换算车辆中心的坐标。
            self.car_x = self.car_x - (
                math.cos(self.car_heading / 180 * math.pi) *
                othercar_dict[veh][traci.constants.VAR_LENGTH] / 2)
            self.car_y = self.car_y - (
                math.sin(self.car_heading / 180 * math.pi) *
                othercar_dict[veh][traci.constants.VAR_LENGTH] / 2)

            # veh_info[veh_dict[veh]].veh_x = self.car_x
            # veh_info[veh_dict[veh]].veh_y = self.car_y
            # veh_info[veh_dict[veh]].veh_type = self.veh_type[
            #     othercar_dict[veh][traci.constants.VAR_TYPE]]
            # veh_info[veh_dict[veh]].veh_width = othercar_dict[veh][
            #     traci.constants.VAR_WIDTH]
            # veh_info[veh_dict[veh]].veh_length = othercar_dict[veh][
            #     traci.constants.VAR_LENGTH]
            # veh_info[veh_dict[veh]].veh_heading = self.car_heading
            # veh_info[veh_dict[veh]].max_dec = othercar_dict[veh][
            #     traci.constants.VAR_EMERGENCY_DECEL]
            # veh_info[veh_dict[veh]].veh_dx = othercar_dict[veh][
            #     traci.constants.VAR_SPEED]
            #
            # # 获取车辆信号灯状态
            # if ((othercar_dict[veh][traci.constants.VAR_SIGNALS] & 0b0011) ==
            #         veh_info[veh_dict[veh]].turn_state):
            #     if (self.sim_time - veh_info[veh_dict[veh]].winker_time >
            #             WINKER_PERIOD):
            #         veh_info[veh_dict[veh]].veh_turn_signal -= 1
            #         veh_info[veh_dict[veh]].winker_time = self.sim_time
            # else:
            #     veh_info[veh_dict[veh]].veh_turn_signal = 1
            #     veh_info[veh_dict[veh]].winker_time = self.sim_time
            # if othercar_dict[veh][traci.constants.VAR_SIGNALS] & 0b0001:
            #     veh_info[veh_dict[veh]].turn_state = 2
            # elif othercar_dict[veh][traci.constants.VAR_SIGNALS] & 0b0010:
            #     veh_info[veh_dict[veh]].turn_state = 1
            # else:
            #     veh_info[veh_dict[veh]].turn_state = 0
            # veh_info[veh_dict[veh]].veh_brake_signal = (
            #     othercar_dict[veh][traci.constants.VAR_SIGNALS] & 0b1000)
            # veh_info[veh_dict[veh]].veh_emergency_signal = (
            #     othercar_dict[veh][traci.constants.VAR_SIGNALS] & 0b0100)
            # veh_info[veh_dict[veh]].render_flag = True

            if othercar_dict[veh][traci.constants.VAR_SIGNALS] & 0b0001:
                steer_signal = 1 #right
            elif othercar_dict[veh][traci.constants.VAR_SIGNALS] & 0b0010:
                steer_signal = -1 #left
            else:
                steer_signal = 0

            if othercar_dict[veh][traci.constants.VAR_SIGNALS] & 0b0100:
                brake_signal = 1
            else:
                brake_signal = 0


            self.vehicles[veh]=dict(order=index,
                type=othercar_dict[veh][traci.constants.VAR_TYPE]
                , x=self.car_x, y=self.car_y, heading=self.car_heading,
                v=othercar_dict[veh][traci.constants.VAR_SPEED],
                turn_signal=steer_signal,
                brake_signal=brake_signal,
                length=othercar_dict[veh][traci.constants.VAR_LENGTH],
                width=othercar_dict[veh][traci.constants.VAR_WIDTH],
                lane_index=othercar_dict[veh][traci.constants.VAR_LANE_INDEX],
                route_id=othercar_dict[veh][traci.constants.VAR_ROAD_ID],
                dist2center=othercar_dict[veh][traci.constants.VAR_LANEPOSITION_LAT],
                allowedspeed=othercar_dict[veh][traci.constants.VAR_ALLOWED_SPEED],)

            index+=1
            # self.vehicles.append(dict(
            #     type=othercar_dict[veh][traci.constants.VAR_TYPE]
            #     , x=self.car_x, y=self.car_y, heading=self.car_heading,
            #     v=othercar_dict[veh][traci.constants.VAR_SPEED],
            #     turn_signal=steer_signal,
            #     brake_signal=brake_signal,
            #     length=othercar_dict[veh][traci.constants.VAR_LENGTH],
            #     width=othercar_dict[veh][traci.constants.VAR_WIDTH],
            #     lane_index=othercar_dict[veh][traci.constants.VAR_LANE_INDEX],))

    def __getcenterindex(self, x, y):
        """Get current intersection's id according to current position: (x,y).

            For Urban Road Map only

            Args:
                x: Vehicle shape center's x coordination, float.
                y: Vehicle shape center's y coordination, float.

            Returns:
                Intersection's id in Urban Road Map.

            Raises:
        """
        if (x < -622 - 18 or (x > -622 + 18 and x < 0 - 18) or
                (x > 0 + 18 and x < 622 - 18) or x > 622 + 18):  # horizontal
            roll = (x + 1244) // 622
            if y > 622:
                index = 15 + roll
            elif y < -622:
                index = 6 + roll
            elif 622 - 7.5 < y < 622:
                index = 16 + roll
            elif 0 < y < 7.5:
                index = 10 + roll
            elif -7.5 < y < 0:
                index = 11 + roll
            else:
                index = 5 + roll
        elif (y < -622 - 18 or (y > -622 + 18 and y < 0 - 18) or
                  (y > 18 and y < 622 - 18) or y > 622 + 18):  # vertical
            roll = (y + 1244) // 622  # line
            if x > 622:
                index = 3 + (roll + 1) * 5
            elif x < -622:
                index = 1 + (roll) * 5
            elif 622 - 7.5 < x < 622:
                index = 3 + (roll) * 5
            elif 0 < x < 7.5:
                index = 2 + (roll + 1) * 5
            elif -7.5 < x < 0:
                index = 2 + (roll) * 5
            else:
                index = 1 + (roll + 1) * 5
        else:
            index = round((x + 1244) / 622) + round((y + 1244) / 622) * 5
        return index


if __name__ == "__main__":
    """__update_veh_list单元测试"""
    # traffic = Traffic(step_length=0.1, path=None, traffic_type=None,
    #              traffic_density=None, init_traffic=None,
    #              regenerate_traffic=False, isLearner=False)
    # # traffic.veh_dict = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6}
    # print('Initial veh dict: '),
    # print(traffic.veh_dict)
    #
    # sumo = {'a':123,'b':321,'f':032,'g':33,'k':3}
    # traffic.update_veh_list(sumo)
    # print('出车>进车')
    # print(VEHICLE_COUNT)
    # print(traffic.veh_dict)
    # print(traffic.id_mid)
    #
    # sumo = {'a':123,'b':321,'f':032,'g':33,'h':3}
    # traffic.update_veh_list(sumo)
    # print('出车=进车')
    # print(VEHICLE_COUNT)
    # print(traffic.veh_dict)
    # print(traffic.id_mid)
    #
    # sumo = {'a':123,'b':321,'f':032,'g':33,'c':2,'d':2}
    # traffic.update_veh_list(sumo)
    # print('出车<进车 free>bias')
    # print(VEHICLE_COUNT)
    # print(traffic.veh_dict)
    # print(traffic.id_mid)
    #
    # sumo = {'a':123,'b':321,'g':33,'c':2,'d':2,'m':2,'l':2}
    # traffic.update_veh_list(sumo)
    # print('出车<进车 free=bias')
    # print(VEHICLE_COUNT)
    # print(traffic.veh_dict)
    # print(traffic.id_mid)
    #
    # sumo = {'a':123,'b':321,'g':33,'c':2,'d':2,'m':2,'l':2,'e':3}
    # traffic.update_veh_list(sumo)
    # print('出车<进车 free<bias')
    # print(VEHICLE_COUNT)
    # print(traffic.veh_dict)
    # print(traffic.id_mid)

    """Traffic类单元测试"""
    traffic = Traffic(step_length=100,
                      map_type='Map1_Urban Road',
                      traffic_type='Vehicle Only Traffic',
                      traffic_density='Sparse',
                      init_traffic=None,
                      regenerate_traffic=False,
                      isLearner=False)
    traffic.init((-360.760, -616.375, 0.0, 180))
    while(True):
        traffic.sim_step()
        print(traffic.get_vehicles())
        # print(traffic.veh_dict)