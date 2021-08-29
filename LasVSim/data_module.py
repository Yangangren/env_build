# coding=utf-8
"""
@author: Chenxiang Xu
@file: data_module.py
@time: 2020/1/2 15:28
@file_desc: Data module for LasVSim-gui 0.2.1.191211_alpha
"""
import _struct as struct
from LasVSim import data_structures


# 读写二进制文件用
SIMULATION_TIME = 1  # 仿真时间数据个数.
EGO_POSITION = 4  # 自车位姿信息数据个数. [x, y, v, heading]
EGO_INFO = 19  # 自车动力学参数数据个数. [steer_wheel(deg), throttle(%), brake_press(Mpa), gear_ratio, engine_speed(rpm), engine_torque(Nm), acc(m/s^2), side_slip(deg), yaw_rate(deg/s), lat_v(m/s), lon_v(m/s), front_wheel_angle(deg), steering_rate(deg/s), fuel_consumption(L), lon_acc(m/s^2), lat_acc(m/s^2), fuel_rate(L/s), dis_to_stop_line(m), speed_limit(m/s)]
TRAFFIC_LIGHT = 2  # 交通信号灯数据个数. [horizontal_traffic_light, vertical_traffic_light]
PATH_NUM = 100  # 期望轨迹点数
PATH_INFO = 5  # 期望轨迹点信息数据个数. [t, x, y, v, heading]
OTHER_VEHICLR_INFO = 8  # 周车位姿信息数据个数. [x, y, v, heading, type, signal, lane_index, max_dec]
EVALUATION_INDEX = 4  # 评价指标个数. [risky, fuel_cost, discomfort, efficiency]


class ReplayData:
    """"
    Replay data manager.

    Attributes:
        frame_count(int): 仿真数据总帧数
        frame_index(int): 当前帧
        data(list): 储存仿真数据的列表.[[time(s), ego_x(m), ego_y(m), ego_v(m/s),
            ego_heading(deg), steer_wheel(deg), throttle(%), brake_press(Mpa),
            gear_ratio, engine_speed(rpm), engine_torque(Nm), acc(m/s^2),
            side_slip(deg), yaw_rate(deg/s), lat_v(m/s), lon_v(m/s),
            front_wheel_angle(deg), steering_rate(deg/s), fuel_consumption(L),
            lon_acc(m/s^2), lat_acc(m/s^2), fuel_rate(L/s), dis_to_stop_line(m),
            speed_limit(m/s), path_point_1_t(s), path_point_1_x(m),
            path_point_1_y(m), path_point_1_v(m/s), path_point_1_heading(deg),
            ..., path_point_100_heading(deg), number_of_valid_path_points,
            horizontal_traffic_light_status, vertical_traffic_light_status,
            risky, fuel_cost(L/100km), discomfort(m/s^2), efficiency,
            number_of_surrounding_vehicles, vehicle_1_x(m), vehicle_1_y(m),
            vehicle_1_vel(m/s), vehicle_1_heading(deg), vehicle_1_type,
            vehicle_1_signal, vehicle_1_lane_index, vehicle_1_max_dec(m/s^2),
            ...vehicle_?_max_dex(m/s^2)]...] (全局坐标系1)
        types(list): 各仿真数据的变量类型.[float, float, float, ...]
        vehicle_count(int): 仿真数据所保存的最大周车数目
    """

    def __init__(self, path):
        """
        初始化函数

        Args:
            path(string): 仿真数据储存文件的路径
        """
        self.frame_count = 0
        self.data = []
        self.frame_index = 0
        with open(path) as f:
            head_line = f.readline()
            self.types = [float]*len(head_line.split(','))
            self.vehicle_count = (
                (len(head_line.split(',')) - SIMULATION_TIME - EGO_POSITION -
                 EGO_INFO - TRAFFIC_LIGHT - PATH_NUM * PATH_INFO - EVALUATION_INDEX - 2) / OTHER_VEHICLR_INFO)
            for i in range(self.vehicle_count):
                self.types[5+SIMULATION_TIME+EGO_POSITION+EGO_INFO + 1 +
                           PATH_NUM * PATH_INFO + 2 + EVALUATION_INDEX +
                           i*OTHER_VEHICLR_INFO] = int  # 周车类型的数据类型
                self.types[5+SIMULATION_TIME+EGO_POSITION+EGO_INFO + 1 +
                           PATH_NUM * PATH_INFO + 2 + EVALUATION_INDEX +
                           i*OTHER_VEHICLR_INFO+1] = int  # 周车转向灯的数据类型
                self.types[5+SIMULATION_TIME+EGO_POSITION+EGO_INFO + 1 +
                           PATH_NUM * PATH_INFO + 2 + EVALUATION_INDEX +
                           i*OTHER_VEHICLR_INFO+2] = int  # 周车所在车道的数据类型
            # 交通信号灯的数据类型
            (self.types[SIMULATION_TIME+EGO_POSITION+EGO_INFO + 1 +
                        PATH_NUM * PATH_INFO],
             self.types[SIMULATION_TIME+EGO_POSITION+EGO_INFO + 1 +
                        PATH_NUM * PATH_INFO+1]) = int, int
            self.types[SIMULATION_TIME+EGO_POSITION+EGO_INFO +
                       PATH_NUM * PATH_INFO] = int  # 有效轨迹点数量的数据类型
            # 周车数量的数据类型
            self.types[SIMULATION_TIME + EGO_POSITION + EGO_INFO + 1 +
                       PATH_NUM * PATH_INFO + 2 + EVALUATION_INDEX] = int
            abc = 0
            for line in f:
                str_arr = line.split(',')
                self.data.append([self.types[i](str_arr[i]) for i in range(len(
                    str_arr))])
        self.frame_count = len(self.data)

    def get_current_time(self):
        """
        返回当前帧对应的仿真时间

        Returns:
            unit: s

        """
        return self.data[self.frame_index][0]

    def get_self_pos(self):
        """
        返回当前帧下自车的位姿

        Returns:
            x: m
            y: m
            v: m/s
            heading: deg
            (全局坐标系1)

        """
        start = SIMULATION_TIME
        x, y, v, heading = self.data[self.frame_index][1:5]
        return x, y, v, heading

    def get_current_trajectory(self):
        """
        获取当前时刻期望轨迹点的全局坐标值。

        Returns:
            轨迹点的数组，例如：
            [[x0,y0],[x1,y1],[x2,y2],...]
            (全局坐标系1)
        """
        trajectory_points_num = self.data[self.frame_index][
            SIMULATION_TIME+EGO_POSITION+EGO_INFO + PATH_NUM * PATH_INFO]
        trajectory = []
        for i in range(trajectory_points_num):
            start = (SIMULATION_TIME+EGO_POSITION+EGO_INFO +
                     +i*PATH_INFO + 1)
            end = start+2
            trajectory.append(self.data[self.frame_index][start:end])
        return trajectory

    def get_other_vehicles(self):
        """
        返回当前帧下周车数据

        Returns:
            C风格结构体数组
            [VehInfo(), VehInfo(),...]

        """
        vehicles = []
        current_veh_num = self.data[self.frame_index][
            SIMULATION_TIME + EGO_POSITION + EGO_INFO + 1 +
            PATH_NUM * PATH_INFO + 2 + EVALUATION_INDEX]
        for i in range(current_veh_num):
            start = (SIMULATION_TIME+EGO_POSITION+EGO_INFO + 1 +
                     PATH_NUM * PATH_INFO + 2 + EVALUATION_INDEX + 1 +
                     i*OTHER_VEHICLR_INFO)

            end = start + OTHER_VEHICLR_INFO
            x, y, v, heading, type, signal, lane_index, max_dec = self.data[self.frame_index][start:end]
            veh = data_structures.VehInfo()
            veh.veh_x = x
            veh.veh_y = y
            veh.veh_heading = heading
            veh.veh_type = type
            veh.veh_turn_signal = signal
            veh.render_flag = True
            vehicles.append(veh)
        return vehicles

    def get_light_status(self):
        """
        返回当前十字路口信号灯状态（目前仅分水平方向和垂直方向）

        Returns:
            dict{h: horizontal_traffic_light_status(int),
                 v: vertical_traffic_light_status(int)}

        """
        h_index = (SIMULATION_TIME+EGO_POSITION+EGO_INFO + 1 +
                   PATH_NUM * PATH_INFO)
        v_index = (SIMULATION_TIME+EGO_POSITION+EGO_INFO + 1 +
                   PATH_NUM * PATH_INFO+1)
        h = self.data[self.frame_index][h_index]
        v = self.data[self.frame_index][v_index]
        s = ['red', 'green', 'yellow']
        return dict(h=s[h], v=s[v])

    def step(self, step):
        """控制回放数据的渲染速度"""
        if self.frame_index + step < self.frame_count:
            self.frame_index += step
            return True
        elif self.frame_index < self.frame_count-1:
            self.frame_index = self.frame_count-1
            return True
        else:
            return False

    def set_frame(self,idx):
        if idx >= self.frame_count:
            return
        self.frame_index = idx

    def get_time(self):
        return [d[0] for d in self.data[:self.frame_index+1]]

    def get_speed(self):
        return [d[3]*3.6 for d in self.data[:self.frame_index+1]]

    def get_x(self):
        return [d[1] for d in self.data[:self.frame_index+1]]

    def get_y(self):
        return [d[2] for d in self.data[:self.frame_index+1]]

    def get_yaw(self):
        return [d[4] for d in self.data[:self.frame_index+1]]

    def get_accel(self):
        return [d[11] for d in self.data[:self.frame_index+1]]

    def get_evaluation(self):
        return 0

    def get_data(self, type):
        if type =='Time':
            return self.get_time()
        elif type =='Vehicle Speed':
            return self.get_speed()
        elif type =='Position X':
            return self.get_x()
        elif type =='Position Y':
            return self.get_y()
        elif type =='Heading Angle':
            return self.get_yaw()
        elif type =='Acceleration':
            return self.get_accel()
        elif type =='Steering Wheel':
            return [d[5] for d in self.data[:self.frame_index+1]]
        elif type =='Throttle':
            return [d[6] for d in self.data[:self.frame_index+1]]
        elif type =='Brake Pressure':
            return [d[7] for d in self.data[:self.frame_index + 1]]
        elif type =='Gear':
            return [d[8] for d in self.data[:self.frame_index + 1]]
        elif type =='Engine Speed':
            return [d[9] for d in self.data[:self.frame_index + 1]]
        elif type =='Engine Torque':
            return [d[10] for d in self.data[:self.frame_index + 1]]
        elif type =='Side Slip':
            return [d[12] for d in self.data[:self.frame_index + 1]]
        elif type =='Yaw Rate':
            return [d[13] for d in self.data[:self.frame_index + 1]]
        elif type =='Lateral Velocity':
            return [d[14] for d in self.data[:self.frame_index + 1]]
        elif type =='Longitudinal Velocity':
            return [d[15] for d in self.data[:self.frame_index + 1]]
        elif type =='Front Wheel Angle':
            return [d[16] for d in self.data[:self.frame_index + 1]]
        elif type =='Steering Rate':
            return [d[17] for d in self.data[:self.frame_index + 1]]
        elif type =='Fuel Consumption':
            return [d[18] for d in self.data[:self.frame_index + 1]]
        elif type =='Longitudinal Acceleration':
            return [d[19] for d in self.data[:self.frame_index + 1]]
        elif type =='Lateral Acceleration':
            return [d[20] for d in self.data[:self.frame_index + 1]]
        elif type =='Fuel Rate':
            return [d[21] for d in self.data[:self.frame_index + 1]]
        else:
            return None

    def get_current_state(self, type):
        if type == 'Steering Wheel':
            return self.data[self.frame_index][5]
        elif type == 'Throttle':
            return self.data[self.frame_index][6]
        elif type == 'Brake Pressure':
            return self.data[self.frame_index][7]
        elif type == 'Gear':
            return self.data[self.frame_index][8]


class Data:
    """"
    Simulation data manager.

    Attributes:
        file: 保存仿真数据的二进制文件

        data(list): 储存自车位姿及动力学状态量的列表. [[time(s), ego_x(m),
            ego_y(m), ego_v(m/s), ego_heading(deg), steer_wheel(deg),
            throttle(%), brake_press(Mpa), gear_ratio, engine_speed(rpm),
            engine_torque(Nm), acc(m/s^2), side_slip(deg), yaw_rate(deg/s),
            lat_v(m/s), lon_v(m/s), front_wheel_angle(deg),
            steering_rate(deg/s), fuel_consumption(L), lon_acc(m/s^2),
            lat_acc(m/s^2), fuel_rate(L/s), dis_to_stop_line(m),
            speed_limit(m/s)]...] (全局坐标系1)
        types(list): 各仿真数据的变量类型.[float, float, float, ...]
        vehicle_count(int): 仿真数据所保存的最大周车数目
    """

    def __init__(self):
        self.vehicle_count = 0  # 仿真交通流数量
        self.trajectory_count = 0  # 时空轨迹点个数（单步）
        self.max_path_points = 100  # 最大时空轨迹点个数
        self.max_veh_num = 0  # 仿真过程中自车周围一定范围内的最大车辆数（除去自车）
        self.data = []
        self.file = open('tmp.bin','wb')

    def __del__(self):
        self.file.close()

    def append(self, self_status, self_info, vehicles, light_values, trajectory,
               dis=None, speed_limit=None, evaluation_result=None,
               vehicle_num=None):
        """保存单步仿真数据函数

        完整的仿真数据会在仿真的每一步保存在二进制文件tmp.bin里，同时自车的动力
        学数据还会保存在data类中供plot函数调用"""
        if self.file.closed:
            return
        sim_time, ego_x, ego_y, ego_vel, ego_heading = self_status
        data_line = [sim_time, ego_x, ego_y, ego_vel, ego_heading]
        data_line.extend(list(self_info))
        self.max_veh_num = max(self.max_veh_num, vehicle_num)
        self.data.append(data_line)
        self.file.write(struct.pack('5f', *[sim_time, ego_x, ego_y, ego_vel,
                                            ego_heading]))  # 保存自车位姿信息
        self.file.write(struct.pack('17f', *self_info))  # 保存自车动力学状态
        # 保存自车距停止线距离和所在车道限速
        self.file.write(struct.pack('2f', *[dis, speed_limit]))
        # 保存期望轨迹数据(最多保存100个轨迹点)
        self.trajectory_count = len(trajectory)
        for i in range(100):
            if i < self.trajectory_count:
                self.file.write(
                    struct.pack('5f', *trajectory[i]))
            else:
                self.file.write(
                    struct.pack('5f', *[0.0, 0.0, 0.0, 0.0, 0.0]))
        # 保存期望轨迹点个数
        self.file.write(struct.pack('i', *[self.trajectory_count]))
        # 保存当前十字路口信号灯状态
        self.file.write(struct.pack('2i', *light_values))
        # 保存当前评价结果：安全、经济、舒适、效率
        self.file.write(struct.pack('4f', *evaluation_result))
        # 保存周车数据（仅保存自车周围一定范围内的周车）
        self.vehicle_count = vehicle_num
        self.file.write(struct.pack('i', *[self.vehicle_count]))
        for veh in vehicles:
            if veh['render']:
                self.file.write(struct.pack('4f3if', *[veh['x'],
                                                       veh['y'],
                                                       veh['v'],
                                                       veh['angle'],
                                                       veh['type'],
                                                       veh['rotation'],
                                                       veh['lane_index'],
                                                       veh['max_decel']]))

    def __get_time(self):
        return [d[0] for d in self.data]

    def __get_speed(self):
        return [d[3]*3.6 for d in self.data]

    def __get_x(self):
        return [d[1] for d in self.data]

    def __get_y(self):
        return [d[2] for d in self.data]

    def __get_yaw(self):
        return [d[4] for d in self.data]

    def __get_accel(self):
        return [d[19] for d in self.data]

    def get_data(self,type):
        if type =='Time':
            return self.__get_time()
        elif type =='Vehicle Speed':
            return self.__get_speed()
        elif type =='Position X':
            return self.__get_x()
        elif type =='Position Y':
            return self.__get_y()
        elif type =='Heading Angle':
            return self.__get_yaw()
        elif type =='Acceleration':
            return self.__get_accel()
        elif type =='Steering Wheel':
            return [d[5] for d in self.data]
        elif type =='Throttle':
            return [d[6] for d in self.data]
        elif type =='Brake Pressure':
            return [d[7] for d in self.data]
        elif type =='Gear':
            return [d[8] for d in self.data]
        elif type =='Engine Speed':
            return [d[9] for d in self.data]
        elif type =='Engine Torque':
            return [d[10] for d in self.data]
        elif type =='Side Slip':
            return [d[12] for d in self.data]
        elif type =='Yaw Rate':
            return [d[13] for d in self.data]
        elif type =='Lateral Velocity':
            return [d[14] for d in self.data]
        elif type =='Longitudinal Velocity':
            return [d[15] for d in self.data]
        elif type =='Front Wheel Angle':
            return [d[16] for d in self.data]
        elif type =='Steering Rate':
            return [d[17] for d in self.data]
        elif type =='Fuel Consumption':
            return [d[18] for d in self.data]
        elif type =='Longitudinal Acceleration':
            return [d[19] for d in self.data]
        elif type =='Lateral Acceleration':
            return [d[20] for d in self.data]
        elif type =='Fuel Rate':
            return [d[21] for d in self.data]
        else:
            return None

    def close_file(self):
        self.file.close()

    def export_csv(self, path):
        """将二进制文件中保存的仿真数据导出为csv格式的可读文件"""
        self.close_file()
        with open(path, 'w') as f:
            f.write('t(s),self_x(m),self_y(m),self_speed(m/s),self_yaw(degree)')
            f.write(',Steering Wheel(degree),Throttle(%),Brake Pressure(MPa),'
                    'Gear,Engine Speed(rpm)')
            f.write(',Engine Torque(N*m),Accelerate(m/s2),Side Slip(degree), '
                    'Yaw Rate(degree/s)')
            f.write(',Lateral Velocity(m/s),Longitudinal Velocity(m/s)')
            f.write(',Front Wheel Angle(deg),Steering Rate(deg/s)')
            f.write(',Fuel Consumption(L),Longitudinal Acceleration(m/s^2)')
            f.write(',Lateral Acceleration(m/s^2),Fuel Rate(L/s)')
            f.write(',Distance To Stop Line(m),Speed Limit(m/s)')
            for i in range(100):
                f.write(',path point%d_t(m),path point%d_x(m),path point%d_y, '
                        'path point%d_v, path_point%d_heading'
                        % (i+1, i+1, i+1, i+1, i+1))
            f.write(',valid trajectory point number')
            f.write(',light_horizontal,light_vertical')
            f.write(',Safety,Economy,Comfort,Efficiency')
            f.write(',Vehicle Num')
            for i in range(self.max_veh_num):
                f.write(',vehicle%d_x(m),vehicle%d_y(m),vehicle%d_speed(m/s),'
                        'vehicle%d_yaw(degree),vehicle%d_type,'
                        'vehicle%d_signals,vehicle%d_lane_index,'
                        'vehicle%d_max_decel(m/s^2)'
                        % (i+1, i+1, i+1, i+1, i+1, i+1, i+1, i+1))
            f.write('\n')

            with open('tmp.bin', 'rb') as fbin:
                fmt = '5f'
                buffer = fbin.read(struct.calcsize(fmt))
                while len(buffer) > 0:
                    f.write('%.6f,%.2f,%.2f,%.2f,%.1f'
                            % struct.unpack(fmt, buffer))  # 保存自车位姿信息
                    fmt = '17f'
                    f.write(
                        ',%.0f,%.0f,%.1f,%.1f,%.0f,%.1f,%.2f,%.1f,%.2f,%.2f,'
                        '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f'
                        % struct.unpack(fmt, fbin.read(
                            struct.calcsize(fmt))))  # 保存自车动力学状态
                    fmt = '2f'
                    f.write(',%.2f,%.2f'  # 保存自车距停止线距离和所在车道限速
                            % struct.unpack(fmt,
                                            fbin.read(struct.calcsize(fmt))))
                    for i in range(100):  # 保存期望轨迹数据(最多保存100个轨迹点)
                        fmt = '5f'
                        f.write(',%.2f,%.2f,%.2f,%.2f,%.2f' % struct.unpack(
                            fmt, fbin.read(struct.calcsize(fmt))))
                    fmt = 'i'  # 保存期望轨迹点个数
                    f.write(',%d' % struct.unpack(fmt, fbin.read(
                        struct.calcsize(fmt))))
                    fmt = '2i'  # 保存当前十字路口信号灯状态
                    f.write(',%d,%d' % struct.unpack(fmt, fbin.read(
                        struct.calcsize(fmt))))
                    fmt = '4f'  # 保存当前评价结果：安全、经济、舒适、效率
                    f.write(',%.2f,%.2f,%.2f,%.2f' % struct.unpack(
                        fmt, fbin.read(struct.calcsize(fmt))))
                    # 保存周车数据（仅保存自车周围一定范围内的周车）
                    fmt = 'i'
                    veh_num = struct.unpack(fmt, fbin.read(
                        struct.calcsize(fmt)))[0]
                    f.write(',%d' % veh_num)
                    for i in range(veh_num):
                        fmt = '4f3if'
                        f.write(',%.2f,%.2f,%.2f,%.1f,%d,%d,%d,%.2f'
                                % struct.unpack(fmt, fbin.read(
                            struct.calcsize(fmt))))
                    f.write('\n')
                    fmt = '5f'
                    buffer=fbin.read(struct.calcsize(fmt))


class TrafficData:
    """保存随机交通流初始状态的数据类"""

    def __init__(self):
        self.file = None  # 保存数据的二进制文件
        pass

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def save_traffic(self, traffic, path):
        """将初始交通流数据保存在二进制文件里"""
        self.file = open(path+'/simulation traffic data.bin', 'wb')
        for veh in traffic:
            self.file.write(struct.pack('6f', *[traffic[veh][64],
                                                traffic[veh][66][0],
                                                traffic[veh][66][1],
                                                traffic[veh][67],
                                                traffic[veh][68],
                                                traffic[veh][77]]))
            name_length = len(traffic[veh][79])
            fmt = 'i'
            self.file.write(struct.pack(fmt, *[name_length]))
            fmt = str(name_length)+'s'
            # 将字符串先转成utf-8的编码，再转为字节类型
            # 修改前为：self.file.write(struct.pack(fmt, *[traffic[veh][79]]))
            self.file.write(struct.pack(fmt,
                                        bytes(str(*[traffic[veh][79]]).encode('utf-8'))))
            name_length = len(traffic[veh][87])
            fmt = 'i'
            self.file.write(struct.pack(fmt, *[name_length]))
            for route in traffic[veh][87]:
                name_length = len(route)
                fmt = 'i'
                self.file.write(struct.pack(fmt, *[name_length]))
                fmt = str(name_length) + 's'
                self.file.write(struct.pack(fmt, bytes(str(*[route]).encode('utf-8'))))
        self.file.close()

    def load_traffic(self, path):
        """从二进制文件中读取初始交通流数据"""
        traffic = {}
        with open(path+'/simulation traffic data.bin', 'rb') as traffic_data:
            fmt = '6f'
            buffer = traffic_data.read(struct.calcsize(fmt))
            id = 0
            while len(buffer) > 0:
                # 读取车辆位姿信息，float类型变量
                v, x, y, heading, length, width = struct.unpack(fmt, buffer)

                # 读取车辆类型，string类型变量
                fmt = 'i'
                name_length = struct.unpack(fmt, traffic_data.read(
                    struct.calcsize(fmt)))[0]  # 读取类型名长度
                fmt = str(name_length)+'s'
                type = struct.unpack(fmt, traffic_data.read(
                    struct.calcsize(fmt)))[0]

                # 读取车辆路径，string类型变量
                route = []
                fmt = 'i'
                name_length = struct.unpack(fmt, traffic_data.read(
                    struct.calcsize(fmt)))[0]  # 读取车辆路径长度
                for i in range(name_length):
                    fmt = 'i'
                    route_length = struct.unpack(fmt, traffic_data.read(
                        struct.calcsize(fmt)))[0]  # 读取路径名长度
                    fmt = str(route_length)+'s'
                    route.append(struct.unpack(fmt, traffic_data.read(
                        struct.calcsize(fmt)))[0])
                traffic[str(id)] = {64: v, 66: (x, y), 67: heading, 68: length,
                                    77: width, 79: type, 87: route}
                id += 1
                fmt = '6f'
                buffer = traffic_data.read(struct.calcsize(fmt))
        return traffic