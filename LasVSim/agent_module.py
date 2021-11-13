# coding=utf-8
"""
@author: Xu Chenxiang
@file: agent_module.py
@time: 2019/12/27 13:41
@file_desc: Agent module for LasVSim-package 0.2.1.191226_alpha
"""
import logging
from LasVSim.sensor_module import *
from LasVSim.controller_module import *
from LasVSim.dynamic_module import *
from LasVSim.traffic_module import *
from LasVSim.decision_module import *
from LasVSim.map_module import *
from LasVSim.data_structures import *
import threading


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
_WINKER_PERIOD=0.5


def plan_thread(agent):
    """
    决策模块线程

    Args:
        agent(Agent instance): LasVSim的Agent类实例

    Returns:
        None

    """
    # TODO(Chason): 后期当前驾驶任务应交由地图模块给出
    # rotation = 0  # TODO(Chason): 待检查作用
    agent.pos_time = agent.dynamic.pos_time
    # x0, y0, v0, a0 = agent.mission.pos  # TODO(Chason): 待检查作用
    if agent.mission.status == MISSION_RUNNING:
        task, task_data = agent.mission.current_task
        # route_mission = RouteMission(ROUTE_MISSION_STRAIGHT)  # TODO(Chason): 待检查作用
        if task == MISSION_GOTO_TARGET:
            xt, yt, vt, headingt = task_data
            target_status, target_lane_info = agent.map.map_position(xt, yt)
            if target_status != MAP_IN_ROAD:
                raise Exception('target not in road')
            direction = target_lane_info['direction']
            cross = target_lane_info['target']
            if cross is not None:
                cross_center = agent.map.get_cross_center(cross)
            else:
                cross_center = agent.map.get_out_cross_center(
                    target_lane_info['source'], direction)
            direction = agent.mission.current_lane['direction']
            agent.turn_history = 'S'  # TODO(Chason)：1023
            light = 'green' if agent.is_light_green(direction) else 'red'
            agent.route = agent.planner.plan(cross_center, direction,
                                             agent.turn_history,  # TODO：1023
                                             (agent.dynamic.ego_x,
                                              agent.dynamic.ego_y,
                                              agent.dynamic.ego_vel,
                                              agent.dynamic.ego_heading),
                                            agent.detected_objects,
                                            light, agent.pos_time)
        elif task == MISSION_GOTO_CROSS:
            cross, target_lane_info = task_data
            cross_center = agent.map.get_cross_center(cross)
            direction = agent.mission.current_lane['direction']
            agent.turn_history = agent.get_turn(direction,   # TODO：1023
                                                agent.mission.next_direction)
            light = 'green' if agent.is_light_green(direction) else 'red'
            agent.route = agent.planner.plan(cross_center, direction,
                                             agent.turn_history,  # TODO：1023
                                             (agent.dynamic.ego_x,
                                              agent.dynamic.ego_y,
                                              agent.dynamic.ego_vel,
                                              agent.dynamic.ego_heading),
                                            agent.detected_objects, light,
                                            agent.pos_time)
        else:
            cross = task_data['source']
            cross_center = agent.map.get_cross_center(cross)
            direction = agent.mission.current_lane['direction']
            light = 'green' if agent.is_light_green(direction) else 'red'
            agent.route = agent.planner.plan(cross_center, direction,
                                             agent.turn_history,
                                             (agent.dynamic.ego_x,
                                              agent.dynamic.ego_y,
                                              agent.dynamic.ego_vel,
                                              agent.dynamic.ego_heading),
                                             agent.detected_objects,
                                             light, agent.pos_time)
        agent.get_future_path()
        agent.controller.set_track(agent.route, 0)
    else:
        pass


def control_thread(agent):
    """
    控制模块线程

    Args:
        agent(Agent instance): LasVSim的Agent类实例

    Returns:

    """
    if agent.lasvsim_version == 'package':
        return
    acc, r, i = agent.mission.engine_state
    agent.mission.control_input = agent.controller.sim_step(
        agent.dynamic.ego_x, agent.dynamic.ego_y, agent.dynamic.ego_heading, acc,
        agent.dynamic.ego_vel, r, i)
    eng_torque, brake_pressure, steer_wheel = agent.mission.control_input
    agent.dynamic.set_control_input(eng_torque, brake_pressure, steer_wheel)


def sensor_thread(agent, other_vehicles):
    """
    传感器模块线程

    Args:
        agent(Agent instance): LasVSim的Agent类实例
        other_vehicles: 储存周车数据真值的数组
        [{'type':0, 'x':0.0, 'y': 0.0, 'v':0.0, 'angle': 0.0,
            'rotation': 0, 'winker': 0, 'winker_time': 0, 'render': False},
            {'type':0, 'x':0.0, 'y': 0.0, 'v':0.0, 'angle': 0.0,
            'rotation': 0, 'winker': 0, 'winker_time': 0, 'render': False},...]
            （全局坐标系1）

    Returns:
        None

    """
    if agent.lasvsim_version == 'package':
        return
    agent.sensors.update([agent.dynamic.ego_x, agent.dynamic.ego_y,
                          agent.dynamic.ego_vel, agent.dynamic.ego_heading],
                         other_vehicles)
    agent.detected_objects = agent.sensors.getVisibleVehicles()
    agent.surrounding_objects_numbers = len(agent.detected_objects)


def dynamic_thread(agent):
    """
    动力学模块线程

    Args:
        agent(Agent instance): LasVSim的Agent类实例

    Returns:
        None

    """
    if agent.lasvsim_version == 'package':
        return
    agent.dynamic.sim_step()
    x, y, v, yaw, acc, r, i = agent.dynamic.get_pos()
    yaw = -yaw + 90
    agent.mission.pos = (x, y, v, yaw)
    agent.mission.engine_state = (acc, r, i)


_ENGINE_TORQUE_FLAG = 0
_STEERING_ANGLE_FLAG = 1
_BRAKE_PRESSURE_FLAG = 2
_ACCELERATION_FLAG = 3
_FRONT_WHEEL_ANGLE_FLAG = 4


class Agent(object):
    """LasVSim的Agent模块.

    Attributes:
        length(float): agent车辆长度，m
        width(float): agent车辆宽度，m
        lw(float): agent车辆长宽之差的一半，m
        front_length(float): agent车辆质心到车头的距离，m
        back_length(float): agent车辆质心到车尾的距离，m

        time(float): 仿真时间，s
        pos_time(float): 用于控制器模块的仿真时间，s
        traffic_lights(dict): 当前十字路口信号灯状态. {'v': 'green', 'h': 'red'}
        future_path(list): agent车辆期望轨迹的地理路径点. [[x0,y0], [x1,y1],...] （全局坐标系2下）
        route(list): agent车辆期望轨迹的时空状态点. [[t0, x0, y0, v0, heading0], [t1, x1, y1, v1, heading1],...] 全局坐标系2下
        rotation(int): agent车辆转向状态
        winker(int): agent车辆转向灯状态
        winker_time(float): agent车辆转向灯闪烁时间,s
        turn_history(char): agent车辆在当前十字路口的驾驶任务. 'S'
        other_vehicles(list): 周车数据列表. [{'type':0, 'x':0.0(m), 'y': 0.0(m), 'v':0.0(m/s), 'angle': 0.0(deg), 'rotation': 0, 'winker': 0, 'winker_time': 0(s), 'render': False},...]（全局坐标系1下）
        detected_objects(list): 传感器返回的周车数据列表. [id, x(m), y(m), v(m/s), heading(deg), width(m), length(m)] (全局坐标系1)

        map(Map obj): LasVSim的Map类实例
        mission(Mission obj): LasVSim的Mission类实例
        controller(Controller obj): LasVSim的Controller类实例
        dynamic(Dynamic obj): LasVSim的Dynamic类实例
        sensors(Sensor obj): LasVSim的Sensor类实例

        decision_thread(Thread obj) = 决策线程实例
        sensor_thread(Thread obj)  = 传感器线程实例
        control_thread(Thread obj)  = 控制器线程实例
        dynamic_thread(Thread obj)  = 动力学线程实例

    """

    def __init__(self, mission=None, map=None, sensor=None, controller=None,
                 dynamic=None, settings=None):
        self.lasvsim_version = 'package'
        self.simulation_settings = settings
        self.route = []
        self.plan_output = []  # plan out put. Each member for example: [t,x,y,velocity,heading angle]
        self.plan_output_type = 0  # 0 for temporal-spatial trajectory, 1 for dynamic input(engine torque, brake pressure, steer wheel angle)
        self.flag = [0, 0, 0, 0, 0]  # indicating which decision output is given.

        # sub classes
        self.mission = None  # mission object
        self.map = None  # map object
        self.controller = None  # controller object
        self.dynamic = None  # dynamic object
        self.planner = None  # router object
        self.sensors = None  # sensor object

        # simulation parameter
        self.step_length = 0.0  # simulation step length, s
        self.time = int(0)  # simulation steps
        self.pos_time = 0.0  # simulation time from controller,s  TODO(Chason):后期检查后删去

        # ego vehicle's parameter
        #   vehicle size
        self.length = settings.car_length  # car length, m
        self.width = settings.car_width  # car width, m
        self.lw = (self.length - self.width) / 2.0  # used for collision check
        self.front_length = settings.car_center2head  # distance from car center to front end, m
        self.back_length = self.length - self.front_length  # distance from car center to back end, m
        self.weight = settings.car_weight  # unladen weight, kg
        # self.shape_center_x = settings.points[0][0]  # m  TODO(Chason): 待检查后删去
        # self.shape_center_y = settings.points[0][1]  # m
        # self.velocity = settings.points[0][2]  # m
        # self.heading = settings.points[0][3]  # deg(In base coordinates)
        # self.acceleration = 0.0  # m/s2
        # self.engine_speed = 0.0  # r/min
        # self.gears_ratio = 1
        # self.engine_torque = 0.0  # N*m
        # self.steer_wheel = 0.0  # deg
        # self.brake_pressure = 0.0  # Mpa
        # self.front_wheel_angle = 0.0  # deg
        # self.desired_acceleration = 0.0  # m/s^2
        # self.minimum_turning_radius = 5.0  # m
        # self.maximum_acceleration = 2.0  # m/s2
        #   vehicle signal
        self.rotation = 0  # turn status
        self.winker = 0  # winker flag for turn signal light
        self.winker_time = 0.0  # winker time for turn signal light
        #   vehicle mission
        self.turn_history = 'S'  # TODO：1023

        # simulation environment
        self.traffic_lights = 'green'  # current traffic light status
        self.future_path = [[1.0, 2.0], [1.0, 2.0]]  # future geo track path to drive(2-D)
        self.other_vehicles = []  # 仿真中存在的所有他车
        self.detected_objects = []  # 传感器探测到的周车状态
        self.surrounding_objects_numbers = int(0)  # TODO(Chason): 检查后删去

        if self.lasvsim_version == 'gui':
            self.mission = mission  # 任务导航对象
            self.map = map  # 地图对象
            self.controller = controller  # 控制器对象
            self.dynamic = dynamic  # 动力学对象
            self.sensors = sensor  # 传感器对象
            self.decision_thread = threading.Thread(target=plan_thread, args=(
            self,))  # creating planning thread
            self.sensor_thread = threading.Thread(target=sensor_thread,
                                                  args=(self,))
            self.control_thread = threading.Thread(target=control_thread,
                                                   args=(self,))
            self.dynamic_thread = threading.Thread(target=dynamic_thread,
                                                   args=(self,))
        elif self.lasvsim_version == 'package':
            self.reset()
        else:
            raise Exception('Wrong version: ' + self.lasvsim_version)

    # def __del__(self):
    #     del self.sensors

    def reset(self):
        """Agent resetting method"""
        if hasattr(self, 'controller'):
            del self.controller
        if hasattr(self, 'planner'):
            del self.planner
        if hasattr(self, 'sensors'):
            del self.sensors
        if hasattr(self, 'navigator'):
            del self.mission
        if hasattr(self, 'dynamic'):
            del self.dynamic

        points = self.simulation_settings.points

        """Load dynamic module."""
        step_length = (self.simulation_settings.step_length *
                       self.simulation_settings.dynamic_frequency)  # ms
        if self.simulation_settings.dynamic_type is None:
            pass # TODO 后期考虑加入其他车
        else:
            self.dynamic = VehicleDynamicModel(x=points[0][0],
                                               y=points[0][1],
                                               heading=points[0][3],
                                               v=points[0][2],
                                               settings=self.simulation_settings,
                                               car_parameter=self.simulation_settings.car_para,
                                               step_length=step_length,
                                               model_type=self.simulation_settings.dynamic_type)

        """Load controller module."""
        step_length = (self.simulation_settings.step_length *
                       self.simulation_settings.controller_frequency)  # ms
        if self.simulation_settings.controller_type == CONTROLLER_TYPE[PID]:
            self.controller=CarControllerDLL(path=CONTROLLER_FILE_PATH,
                                             step_length=step_length,
                                             model_type=CONTROLLER_TYPE[PID],
                                             car_parameter=self.simulation_settings.car_para,
                                             input_type=self.simulation_settings.router_output_type,
                                             car_model=self.simulation_settings.dynamic_type)
        elif self.simulation_settings.controller_type == CONTROLLER_TYPE[EXTERNAL]:
            self.controller=CarControllerDLL(path=CONTROLLER_FILE_PATH,
                                             step_length=step_length,
                                             model_type=CONTROLLER_TYPE[PID],
                                             car_parameter=self.simulation_settings.car_para,
                                             input_type=self.simulation_settings.router_output_type,
                                             car_model=self.simulation_settings.dynamic_type)
        else:
            pass  # TODO 后期加入嵌入新算法后的选择功能

        """Load decision module."""
        step_length = (self.simulation_settings.step_length *
                       self.simulation_settings.router_frequency)
        # if self.simulation_settings.router_type == 'No Planner':
        #     self.planner = None
        # elif self.simulation_settings.router_type == '-':
        #     pass  # TODO(Xu Chenxiang):
        # else:
        # self.planner = Planner(step_length=step_length,
        #                        path=self.simulation_settings.router_lib,
        #                        settings=self.simulation_settings)
        # self.decision_thread = threading.Thread(target=plan_thread, args=(self,))

        """Load sensor module."""
        step_length = (self.simulation_settings.step_length *
                       self.simulation_settings.sensor_frequency)
        self.sensors = Sensors(step_length=step_length,
                               sensor_info=self.simulation_settings.sensors)
        self.map = Map()  # TODO
        # self.mission = Mission(self.map, points,
        #                        self.simulation_settings.mission_type)        # self.predictor = VehiclePredictor(self.map)

    def set_router(self, router):
        """
        set route planner for agent
        """
        self.planner = router

    def update_data(self, status, time, traffic_lights, detected_vehicles):
        """
        set input data before route plan
        """
        self.time = time
        x, y, v, a = status
        a = -a + 90
        self.mission.update((x, y, v, a))
        self.traffic_lights = traffic_lights
        self.detected_objects = detected_vehicles

    def is_light_green(self, direction):
        if direction in 'NS':
            return self.traffic_lights['v'] == 'green'
        else:
            return self.traffic_lights['h'] == 'green'

    def plan(self):
        if self.decision_thread.is_alive():
            _logger.info('Planning Delay')
            self.decision_thread.join()
        self.decision_thread = threading.Thread(target=plan_thread, args=(self,))
        self.decision_thread.start()
        self.decision_thread.join()  # TODO(Chason): 关并行

    def get_future_path(self):
        """
        get geo track from temporal-spatial route
        """
        if self.route is not None:
            t,x,y,v,heading = zip(*self.route)
            self.future_path=zip(x,y)

    def get_pos(self, t=None):
        """
        get position for simulation time t
        """
        self.update_dynamic_state()
        x, y, v, a = (self.dynamic.ego_x, self.dynamic.ego_y, self.dynamic.ego_vel,
                      -self.dynamic.ego_heading + 90)
        # a = -a + 90
        status = x, y, v, a
        return status

    def get_view_pos(self):
        """
        get position for display
        """
        x, y, v, a = (self.dynamic.ego_x, self.dynamic.ego_y, self.dynamic.ego_vel,
                      self.dynamic.ego_heading)
        a = -a+90
        if self.mission.pos_status == MAP_IN_FIELD:
            return (x, y), a
        elif self.mission.pos_status == MAP_IN_ROAD:
            road = self.mission.current_lane
            direction = road['direction']
            return (x, y), -self.map.direction_to_angle(direction)
        else:
            if self.mission.cross_task == 'S':
                road = self.mission.current_lane
                direction = road['direction']
                return (x, y), -self.map.direction_to_angle(direction)
            else:
                return (x, y), a

    def get_turn(self,d1,d2):
        """
        get turn status
        """
        dirs='NWSE'
        n1=dirs.find(d1)
        n2=dirs.find(d2)
        if n1<0 or n2<0:
            return None
        elif n1==n2:
            return 'S'
        elif abs(n1-n2) == 2:
            return 'U'
        elif n1+1 == n2 or n1-3 == n2:
            return 'L'
        else:
            return 'R'

    def update_winker(self,r):
        """
        update winker status for turn signal
        """
        if self.rotation!=r:
            self.winker=1
            self.winker_time=self.time
        else:
            if self.time-self.winker_time>=_WINKER_PERIOD:
                self.winker=1-self.winker
                self.winker_time=self.time
            else:
                w=1
                wt=self.time
        self.rotation=r

    def get_control_info(self):
        """
        get car data from controller
        """
        return self.dynamic.get_info()[:4]

    def get_drive_status(self):
        is_in_cross,is_change_lane=True,False
        car2border=0
        x, y, v, a = self.mission.pos
        lane_x, lane_y = x,y
        pos_status,pos_data=self.map.map_position(x, y)
        if pos_status is MAP_IN_ROAD:
            is_in_cross=False
            lane_x,lane_y=self.map.get_road_center_point((x, y))
            dx,dy= MAP_ROAD_WIDTH / 4, 0
            if pos_data['lane']=='R':
                dx*=3
            if pos_data['direction'] in 'SE':
                dx=-dx
            if pos_data['direction'] in 'EW':
                dx,dy=dy,dx
            lane_x,lane_y=lane_x+dx,lane_y+dy
            if pos_data['direction'] in 'NS':
                departure_distance=abs(lane_x-x)
            else:
                departure_distance=abs(lane_y-y)
            if departure_distance>1.0:
                is_change_lane=True
            car2border=(3.75-1.8)/2-departure_distance
        return is_in_cross,is_change_lane,lane_x,lane_y,car2border

    def plan_control_input(self):
        """控制器模块计算控制输入量

        --"""
        if self.control_thread.is_alive():
            self.control_thread.join()
            print('controller delay')
        self.control_thread = threading.Thread(target=control_thread, args=(self,))
        self.control_thread.start()
      #  self.control_thread.join()

    def update_info_from_sensor(self, traffic):
        """
        调用传感器模块对周车数据进行过滤和加噪

        Args:
            surrounding_obj: 周车数据列表
            [{'type':0, 'x':0.0, 'y': 0.0, 'v':0.0, 'angle': 0.0,
            'rotation': 0, 'winker': 0, 'winker_time': 0, 'render': False},
            {'type':0, 'x':0.0, 'y': 0.0, 'v':0.0, 'angle': 0.0,
            'rotation': 0, 'winker': 0, 'winker_time': 0, 'render': False},...]
            （全局坐标系1下）

        Returns:

        """
        if self.lasvsim_version == 'package':
            status = [self.dynamic.ego_x, self.dynamic.ego_y, self.dynamic.ego_vel,
                      self.dynamic.ego_heading]
            self.sensors.update(pos=status, vehicles=traffic)
            self.detected_objects = self.sensors.getVisibleVehicles()
            self.surrounding_objects_numbers = len(self.detected_objects)
        else:
            if self.sensor_thread.is_alive():
                self.sensor_thread.join()
                print('sensor delay')
            self.sensor_thread = threading.Thread(target=sensor_thread,
                                                  args=(self, traffic,))
            self.sensor_thread.start()
            # self.sensor_thread.join()


    def update_dynamic_state(self,):
        """Run ego's dynamic model according to given steps.

        Args:
            steps: Int variable bigger than 1.
        """
        if self.lasvsim_version == 'package':
            self.dynamic.sim_step()
            self.time += 1  # TODO(Chason): 待检查
        else:
            # TODO(Chason): 线程开销可能会抵消通过并行减少的动力学模块计算开销
            if self.mission.arrive_target():
                return
            if self.dynamic_thread.is_alive():
                self.dynamic_thread.join()
                _logger.info('dynamic_delay')
            self.dynamic_thread = threading.Thread(target=dynamic_thread,
                                                   args=(self,))
            self.dynamic_thread.start()

    def set_engine_torque(self, torque):
        self.engine_torque = torque
        self.flag[_ENGINE_TORQUE_FLAG] = 1

    def set_brake_pressure(self, brake):
        self.brake_pressure = brake
        self.flag[_BRAKE_PRESSURE_FLAG] = 1

    def set_steering_angle(self, steer):
        self.steer_wheel = steer
        self.flag[_STEERING_ANGLE_FLAG] = 1

    def set_acceleration(self, acc):
        self.acceleration = acc
        self.flag[_ACCELERATION_FLAG] = 1

    def set_front_wheel_angle(self, angle):
        self.front_wheel_angle = angle
        self.flag[_FRONT_WHEEL_ANGLE_FLAG] = 1

    def update_control_input(self, torque=None, brake=None, steer=None):
        """Compute control input.

        If torque is None, brake is None and steer is not None, then choose
        internal longitudinal controller.
        If torque is not None, brake is not None and steer is None, then choose
        internal lateral controller.
        If torque is  None, brake is  None and steer is None, then choose no
        controller.
        Else choose both internal lateral and longitudinal controller.

        Args:
            torque: Engine's output torque, N*m
            brake: Braking system's main brake pressure, Mpa
            steer: Steering angle, deg
        """
        (engine_torque, brake_pressure,
         steer_wheel) = self.controller.sim_step(self.dynamic.ego_x,
                                                 self.dynamic.ego_y,
                                                 self.dynamic.ego_heading,
                                                 self.dynamic.acc,
                                                 self.dynamic.ego_vel,
                                                 self.dynamic.engine_speed,
                                                 self.dynamic.drive_ratio)
        self.dynamic.set_control_input(eng_torque=engine_torque,
                                       brake_pressure=brake_pressure,
                                       steer_wheel=steer_wheel)

    def update_plan_output(self, traffic_lights):
        """Ego vehicle plans, 内置的决策算法.

        According to current surrounding environment and driving tasks, returns
         a desired spatio-temporal trajectory

         Returns:
             A list containing state information of each point on the desired
             trajectory. For example:
             [[time:s, x:m, y:m, velocity:m/s, heading:deg],
             [0.0, 0.0, 0.0, 0.0,0.0]...]
        """
        # Update current plan state before planning.
        status = [self.dynamic.ego_x, self.dynamic.ego_y, self.dynamic.ego_vel,
                  self.dynamic.ego_heading]
        self.mission.update(status)
        self.traffic_lights = traffic_lights
        self.plan()

if __name__ == "__main__":
    f= open("data.txt",'r')
    a = f.readlines()
    print(a[0])

