# coding=utf-8
"""
@author:
@file: data_module.py
@time: 2020/1/3 20:19
@file_desc: Data module for LasVSim-gui 0.2.1.191211_alpha
"""
import untangle
from xml.dom.minidom import Document
import time
import re
import io
import copy
from LasVSim import data_structures
from LasVSim import data_module
from LasVSim.traffic_module import *
from LasVSim.agent_module import *
from LasVSim.evaluation_module import Evaluator
# from LasVSim.rendering_module import Display
import numpy as np

ACS_SIMULATION_REFRESH_TICKSTEP = 2


def traffic_thread(simulation):
    simulation.traffic.set_own_car(simulation.agent.dynamic.ego_x,
                                   simulation.agent.dynamic.ego_y,
                                   simulation.agent.dynamic.ego_vel,
                                   simulation.agent.dynamic.ego_heading)
    simulation.traffic.sim_step()
    simulation.other_vehicles = simulation.traffic.get_vehicles(simulation.veh_info)
    simulation.other_vehicles_num = simulation.traffic.get_vehicle_num()
    simulation.light_status = simulation.traffic.get_light_status()
    simulation.light_values = simulation.traffic.get_light_values()
    simulation.light_info=simulation.traffic.get_light_info() #add by TKM traffic


class Simulation(object):
    """
    Simulation Class

    Attributes:
        lasvsim_version(string): LasVSim版本. gui, package
        tick_count(int): 当前仿真步数
        sim_time(float): 当前仿真步对应仿真时间,s

        ego_x0(float): 自车碰撞检测圆圆心,m
        ego_y0(float): 自车碰撞检测圆圆心,m
        ego_x1(float): 自车碰撞检测圆圆心,m
        ego_y1(float): 自车碰撞检测圆圆心,m
        surrounding_x0(float): 他车碰撞检车圆圆心,m
        surrounding_y0(float): 他车碰撞检车圆圆心,m
        surrounding_x1(float): 他车碰撞检车圆圆心,m
        surrounding_y1(float): 他车碰撞检车圆圆心,m
        surrounding_lw(float): 他车长宽之差的一半，用作碰撞检测,m
        collision_check_dis(float): 碰撞距离,m

        other_vehicles(list)
    """
    def __init__(self, canvas=None, setting_path=None, learner_settins=None):
        self.lasvsim_version = 'gui'
        self.veh_info = (data_structures.VehInfo * 2000)()
        # self.flag = 0  # TODO(Chason): 检查后删去

        self.ego_x0, self.ego_y0 = 0.0, 0.0  # 自车碰撞检测点0
        self.ego_x1, self.ego_y1 = 0.0, 0.0  # 自车碰撞检测点1
        self.surrounding_x0, self.surrounding_y0 = 0.0, 0.0  # 他车碰撞监测点0
        self.surrounding_x1, self.surrounding_y1 = 0.0, 0.0  # 他车碰撞监测点1
        self.surrounding_lw = 0.0  # 他车长宽之差的一半，用作碰撞检测
        self.collision_check_dis = 0.0  # 碰撞距离

        self.other_vehicles = None  # 仿真中所有他车状态信息（全局坐标系1）
        self.other_vehicles_num = 0  # 仿真中所有他车数量
        self.light_status = None  # 当前十字路口信号灯状态
        self.stopped = False  # 仿真结束标志位

        self.agent = None  # 自车类
        self.traffic = None  # 交通流类

        self.tick_count = 0  # Simulation run time. Counted by simulation steps.
        self.sim_time = 0.0  # Simulation run time. Counted by steps multiply step length.

        self.init_traffic_distribution_data = data_module.TrafficData()  # 初始交通流数据对象
        self.settings = None  # 仿真设置对象
        self.data = None

        if self.lasvsim_version == 'gui':
            self.canvas = canvas  # display canvas
            self.VisualizeEnabled = True
            self.traffic_thread = threading.Thread(target=traffic_thread,
                                                   args=(self,))  # 交通流运行线程
            self.map = Map('Map1_Urban Road')  # 地图对象
            self.disp = Display(self.canvas, self.map)  # 渲染模块对象
            self.learner_settings = None  # Learner模式的仿真设置对象
            self.evaluator = None
        else:
            self.external_control_flag = True  # 外部控制输入标识，若外部输入会覆盖内部控制器
            self.reset(settings=Settings(file_path=setting_path))

    def reset(self, settings=None, init_traffic=None):
        """Clear previous loaded module.

        Args:
            settings: LasVSim's setting class instance. Containing current
                simulation's configuring information.
        """
        if hasattr(self, 'traffic'):
            del self.traffic
        if hasattr(self, 'agent'):
            del self.agent
        if hasattr(self, 'data'):
            del self.data
        if self.lasvsim_version == 'gui':
            if hasattr(self, 'evaluator'):  # TODO(Chason): 后期packgae版本也要加入评价模块
                del self.evaluator

        self.tick_count = 0
        self.settings = settings
        self.stopped = False
        self.data = data_module.Data()
        if self.lasvsim_version == 'gui':
            self.evaluator = Evaluator()

        """Load vehicle module library"""
        if self.lasvsim_version == 'gui':
            vehicle_model = VehicleModelsReader('Library/vehicle_model_library.csv')
        else:
            vehicle_model = VehicleModelsReader('LasVSim/Library/vehicle_model_library.csv')

        """Load traffic module."""
        step_length = self.settings.step_length * self.settings.traffic_frequency  # ms
        self.traffic=Traffic(map_type=settings.map,
                             traffic_type=settings.traffic_type,
                             traffic_density=settings.traffic_lib,
                             step_length=step_length,
                             init_traffic=init_traffic,
                             settings=settings)
        self.traffic.init(settings.points[0])
        if self.lasvsim_version == 'gui':
            self.traffic_thread = threading.Thread(target=traffic_thread,
                                                   args=(self,))
        self.other_vehicles = self.traffic.get_vehicles(self.veh_info)
        self.light_status = self.traffic.get_light_status()
        self.light_info=self.traffic.get_light_info() #add by TKM  得到所有traffic_light

        if self.lasvsim_version == 'gui':
            if settings.map != MAPS[0] and settings.map != MAPS[1]:
                self.VisualizeEnabled = 0
            self.map = Map(settings.map)
            self.disp = Display(self.canvas, self.map, settings)
            self.disp.setVehicleModel(vehicle_model)
            self.disp.own_car_type = self.settings.dynamic_type
        # print('Step Length:  ', self.settings.step_length)
        # print('Sensor Fre:  ', self.settings.sensor_frequency)
        # print('Traffic Fre:  ', self.settings.traffic_frequency)
        # print('Decision Fre:  ', self.settings.router_frequency)
        # print('Controller Fre:  ', self.settings.controller_frequency)
        # print('Dynamic Fre:  ', self.settings.dynamic_frequency)
        if self.lasvsim_version == 'gui':
            """载入控制器模块"""
            step_length = self.settings.step_length*self.settings.controller_frequency  # ms
            if settings.controller_type == CONTROLLER_TYPE[0]:
                controller = CarControllerDLL(path=self.settings.controller_lib,
                                              step_length=step_length,
                                              model_type=CONTROLLER_TYPE[0],
                                              car_parameter=self.settings.car_para,
                                              input_type=settings.router_output_type,
                                              car_model=settings.dynamic_type)
            elif settings.controller_type == CONTROLLER_TYPE[1]:
                if settings.controller_file_type == 'C/C++ DLL':
                    controller = CarControllerDLL(
                        path=self.settings.controller_lib,
                        step_length=step_length,
                        model_type="External",
                        car_parameter=self.settings.car_para,
                        input_type=settings.router_output_type,
                        car_model=settings.dynamic_type)
                elif settings.controller_file_type == 'Python':
                    [file_path, file_name] = os.path.split(
                        self.settings.controller_lib)
                    sys.path.append(file_path)
                    from PythonController import CarControllerPython
                    controller = CarControllerPython(step_length=step_length,
                                                     model_type="External",
                                                     car_parameter=self.settings.car_para,
                                                     input_type=settings.router_output_type,
                                                     car_model=settings.dynamic_type)
            # 载入动力学模块
            step_length = self.settings.step_length*self.settings.dynamic_frequency  # ms
            if settings.dynamic_type is None:
                pass  # TODO(Chason): 后期考虑加入其他车
            else:
                dynamic = VehicleDynamicModel(x=settings.points[0][0],
                                              y=settings.points[0][1],
                                              heading=-settings.points[0][3]+90,
                                              v=settings.points[0][2],
                                              settings=settings,
                                              car_parameter=self.settings.car_para,
                                              step_length=step_length,
                                              model_type=settings.dynamic_type)
            # 载入传感器模块
            step_length = self.settings.step_length*self.settings.traffic_frequency  # ms
            sensors = Sensors(step_length=step_length,
                              path=settings.sensor_model_lib)
            sensors.setSensor(settings.sensors)
            sensors.setVehicleModel(vehicle_model)
            # 载入导航模块
            mission = Mission(self.map, settings.points, settings.mission_type)
            # 载入agent模块
            self.agent = Agent(mission=mission, map=self.map,
                               sensor=sensors, controller=controller,
                               dynamic=dynamic, settings=settings)
            # 载入决策模块
            step_length = self.settings.step_length*self.settings.router_frequency  # ms
            if settings is None or settings.router_type == 'LatticeRouter':
                self.agent.set_router(Planner(step_length=step_length,
                                              path=settings.router_lib,
                                              settings=settings))
            else:
                self.agent.set_router(Planner(step_length=step_length,
                                              path=settings.router_lib,
                                              settings=settings))
            ps = (tuple(self.agent.mission.points[0][:2]),
                  self.agent.mission.points[0][3])
            pt = (tuple(self.agent.mission.points[-1][:2]),
                  self.agent.mission.points[-1][3])
            self.disp.set_start_target(ps, pt)
            # 仿真初始化
            self.agent.get_pos()
            self.light_values = self.traffic.get_light_values()
            self.agent.update_info_from_sensor(self.other_vehicles,self.light_info)
            self.agent.controller.set_track(track=[[0,self.agent.dynamic.ego_x,
                                             self.agent.dynamic.ego_y,
                                             self.agent.dynamic.ego_vel,
                                             self.agent.dynamic.ego_heading]])
            self.disp.set_sensors(self.agent.sensors.sensors)
            self.disp.set_data(self.other_vehicles, self.light_status,
                               (self.agent.dynamic.ego_x,
                                self.agent.dynamic.ego_y,
                                self.agent.dynamic.ego_vel,
                                self.agent.dynamic.ego_heading),
                               self.agent.detected_objects,self.agent.detected_man,
                               self.agent.detected_traffic_sign,self.veh_info,self.agent.state_estimate,self.agent.state_estimate_aof)
            self.disp.set_info(
                dict(t=self.traffic.sim_time, rotation=0, winker=0),
                None)
            self.disp.set_pos(*self.agent.get_view_pos())
            self.disp.draw()
        else:
            """Load agent module."""
            self.agent=Agent(settings=settings)
            self.agent.sensors.setVehicleModel(vehicle_models)

    def sim_step_internal(self, steps=None):
        """单步仿真更新函数（非回放模式）"""
        if steps is None:
            steps = 1
        if self.lasvsim_version == 'gui':
            if self.stopped:
                #print("Simulation Finished")  # TODO(Chason): 改为logger输出
                return False
            disp_refreshed = 0

            # 传感器线程
            if self.tick_count % self.settings.sensor_frequency == 0:
                self.agent.update_info_from_sensor(self.other_vehicles,self.light_info)
                #gui版本从这进入传感器线程tkm

            # 决策线程
            # if self.tick_count % self.settings.router_frequency == 0:  # TODO:待修改
            if self.tick_count % 1 == 0:
                self.agent.update_data((self.agent.dynamic.ego_x,
                                        self.agent.dynamic.ego_y,
                                        self.agent.dynamic.ego_vel,
                                        self.agent.dynamic.ego_heading),
                                       self.traffic.sim_time,
                                       self.light_status,
                                       self.agent.detected_objects)
                self.agent.plan()

            # 控制器线程
            if self.tick_count % self.settings.controller_frequency == 0:
                self.agent.plan_control_input()

            # 动力学线程
            if self.tick_count % self.settings.dynamic_frequency == 0:
                self.agent.update_dynamic_state()

            # 交通流线程
            if self.tick_count % self.settings.traffic_frequency == 0:
                if self.traffic_thread.is_alive():
                    #print('traffic delay')
                    self.traffic_thread.join()
                self.traffic_thread = threading.Thread(target=traffic_thread,
                                                       args=(self,))
                self.traffic_thread.start()
                self.traffic_thread.join()

            # 渲染仿真
            if self.tick_count % steps == 0 and self.VisualizeEnabled:
                self.disp.set_data(self.other_vehicles, self.light_status,
                                   (self.agent.dynamic.ego_x,
                                    self.agent.dynamic.ego_y,
                                    self.agent.dynamic.ego_vel,
                                    self.agent.dynamic.ego_heading),
                                   self.agent.detected_objects,self.agent.detected_man,
                                   self.agent.detected_traffic_sign,self.veh_info,self.agent.state_estimate,self.agent.state_estimate_aof)
                self.disp.set_info(dict(t=self.traffic.sim_time,
                                        rotation=self.agent.rotation,
                                        winker=self.agent.winker),
                                   self.agent.future_path)
                self.disp.set_pos(*self.agent.get_view_pos())
                self.disp.draw()
                disp_refreshed = 1

            # 更新评价模块
            self.update_evaluation_data()

            # 保存当前步仿真数据
            self.data.append(
                self_status=[self.traffic.sim_time,
                             self.agent.dynamic.ego_x,
                             self.agent.dynamic.ego_y,
                             self.agent.dynamic.ego_vel,
                             self.agent.dynamic.ego_heading],
                self_info=self.agent.dynamic.get_info(),
                vehicles=self.other_vehicles,
                light_values=self.light_values,
                trajectory=self.agent.route,
                dis=self.traffic.get_current_distance_to_stopline(),
                speed_limit=self.traffic.get_current_lane_speed_limit(),
                evaluation_result=self.evaluator.get_report()[4:8],
                vehicle_num=self.other_vehicles_num)

            # 如果自车达到目的地或发生碰撞则退出仿真，loop路径下仿真会一直循环不会结束
            if (self.agent.mission.get_status() != MISSION_RUNNING or
                not self.__collision_check()):
                self.stop()
                self.disp.draw()
                disp_refreshed = 1

            self.tick_count += 1
            return disp_refreshed
        else:
            for step in range(steps):
                if self.stopped:
                    #print("Simulation Finished")
                    return False

                # 传感器线程
                if self.tick_count % 1 == 0:
                    self.agent.update_info_from_sensor(self.other_vehicles,self.light_info)
                # 决策线程
                if self.tick_count % 2 == 0:
                    self.agent.update_plan_output(self.light_status)

                # 控制器线程
                if self.tick_count % 1 == 0:
                    self.agent.update_control_input()

                # 动力学线程
                if self.tick_count % 1 == 0:
                    self.agent.update_dynamic_state()

                # 交通流线程
                if self.tick_count % 1 == 0:
                    self.traffic.set_own_car(self.agent.dynamic.ego_x,
                                             self.agent.dynamic.ego_y,
                                             self.agent.dynamic.ego_vel,
                                             self.agent.dynamic.ego_heading)
                    self.traffic.sim_step()
                    self.other_vehicles = self.traffic.get_vehicles(
                        self.veh_info)
                    self.light_status = self.traffic.get_light_status()
                    self.light_info = self.traffic.get_light_info()  # add by TKM  得到所有traffic_light
                    self.other_vehicles_num = self.traffic.get_vehicle_num()

                # 如果自车达到目的地则退出仿真，loop路径下仿真会一直循环不会结束
                if self.agent.mission.get_status() != MISSION_RUNNING:
                    self.stop()

                # 保存当前步仿真数据
                self.data.append(
                    self_status=[
                        self.tick_count * float(self.settings.step_length),
                        self.agent.dynamic.ego_x,
                        self.agent.dynamic.ego_y,
                        self.agent.dynamic.ego_vel,
                        self.agent.dynamic.ego_heading],
                    self_info=self.agent.dynamic.get_info(),
                    vehicles=self.other_vehicles,
                    light_values=self.traffic.get_light_values(),
                    trajectory=self.agent.route,
                    dis=self.traffic.get_current_distance_to_stopline(),
                    speed_limit=self.traffic.get_current_lane_speed_limit(),
                    evaluation_result=[0.0, 0.0, 0.0, 0.0],
                    vehicle_num=self.other_vehicles_num)  # TODO(Chason): 待补充
                self.tick_count += 1

                if not self.__collision_check():
                    self.stop()
                    return False
            return True

    def sim_step(self, steps=None):
        if steps is None:
            steps = 1
        for step in range(steps):
            if self.stopped:
                #print("Simulation Finished")
                return False

            # 传感器线程
            if self.tick_count % self.settings.sensor_frequency == 0:
                self.agent.update_info_from_sensor(self.other_vehicles,self.light_info)
            # 控制器线程
            if self.tick_count % self.settings.controller_frequency == 0:
                self.agent.update_control_input()

            # 动力学线程
            if self.tick_count % self.settings.dynamic_frequency == 0:
                self.agent.update_dynamic_state()

            # 交通流线程
            if self.tick_count % self.settings.traffic_frequency == 0:
                self.traffic.set_own_car(self.agent.dynamic.x,
                                         self.agent.dynamic.y,
                                         self.agent.dynamic.v,
                                         self.agent.dynamic.heading)
                self.traffic.sim_step()
                self.other_vehicles = self.traffic.get_vehicles(self.veh_info)
                self.light_status = self.traffic.get_light_status()
                self.light_info=self.traffic.get_light_info() #add by TKM  得到所有traffic_light

            # 如果自车达到目的地则退出仿真，loop路径下仿真会一直循环不会结束
            if self.agent.mission.get_status() != MISSION_RUNNING:
                self.stop()

            # 保存当前步仿真数据
            self.data.append(
                self_status=[self.tick_count * float(self.settings.step_length),
                             self.agent.dynamic.ego_x,
                             self.agent.dynamic.ego_y,
                             self.agent.dynamic.ego_vel,
                             self.agent.dynamic.ego_heading],
                self_info=self.agent.dynamic.get_info(),
                vehicles=self.other_vehicles,
                light_values=self.traffic.get_light_values(),
                trajectory=self.agent.route,
                dis=self.traffic.get_current_distance_to_stopline(),
                speed_limit=self.traffic.get_current_lane_speed_limit(),
                evaluation_result=[0.0, 0.0, 0.0, 0.0],
                vehicle_num=200)

            self.tick_count += 1

            if not self.__collision_check():
                self.stop()
                return False
        return True

    def load_scenario(self, path):
        """Load an existing LasVSim simulation configuration file.

        Args:
            path:
        """
        if os.path.exists(path):
            settings = Settings()
            settings.load(path+'/simulation setting file.xml')
            #self.reset(settings)
            self.reset(settings, self.init_traffic_distribution_data.load_traffic(path))
            self.simulation_loaded = True
            return
        #print('\033[31;0mSimulation loading failed: 找不到对应的根目录\033[0m')
        self.simulation_loaded = False

    def set_visualize(self, value):
        self.VisualizeEnabled = value

    def get_canvas(self):
        return self.canvas

    def get_self_car_info(self):
        return self.agent.get_control_info()

    def get_time(self):
        if self.lasvsim_version == 'gui':
            return float(self.tick_count)*float(self.settings.step_length)/1000.0
        else:
            return self.traffic.sim_time

    def get_pos(self):
        x = self.agent.dynamic.ego_x
        y = self.agent.dynamic.ego_y
        v = self.agent.dynamic.ego_vel
        heading = -self.agent.dynamic.ego_heading+90  # 全局坐标系1 → 全局坐标系2
        status = x, y, v, heading
        return status

    def get_dynamic_type(self):
        return self.agent.dynamic.model_type

    def export_data(self, path):
        self.data.export_csv(path)

    def get_all_objects(self):
        return self.other_vehicles

    def get_detected_objects(self):
       return self.agent.detected_objects

    def get_ego_position(self):
        return self.agent.dynamic.ego_x, self.agent.dynamic.ego_y

    def get_controller_type(self):
        return self.agent.controller.model_type

    def mission_update(self, pos):
        self.agent.mission.update(pos)

    def stop(self):
        self.stopped = True
        #汤凯明毕业新增
        tmp_err = np.array(self.agent.errors,dtype='object')
        np.save(r'testdata\err.npy', tmp_err)
        tmp_detection = np.array(self.agent.detection,dtype='object')
        np.save(r'testdata\detecton.npy', tmp_detection)
        tmp_result=np.array(self.agent.fusion_result,dtype='object')
        np.save(r'testdata\result.npy', tmp_result)
        tmp_exp2KFresult = np.array(self.agent.fusion_result_exp2_KF, dtype='object')
        np.save(r'testdata\exp2KFresult.npy', tmp_exp2KFresult)
        tmp_exp2AOFresult = np.array(self.agent.fusion_result_exp2_AOF, dtype='object')
        np.save(r'testdata\exp2AOFresult.npy', tmp_exp2AOFresult)
        self.data.close_file()

    def update_evaluation_data(self):
        """将仿真数据传给评价模块"""
        x, y, v, heading = self.get_pos()
        speed = v
        plan_pos = self.agent.controller.get_plan_pos()
        if plan_pos is not None:
            plan_x, plan_y = plan_pos[1:3]
        else:
            plan_x, plan_y = x, y

        (is_in_cross, is_change_lane, lane_x, lane_y,
         car2border) = self.agent.get_drive_status()
        front_d = 1000
        front_speed = -1
        front_x, front_y = plan_x, plan_y+1000
        if not is_change_lane and not is_in_cross:
            status, lane_info = self.map.map_position(x, y)
            for vehicle in self.other_vehicles:
                if not vehicle['render']:
                    continue
                vx, vy = vehicle['x'], vehicle['y']
                if get_distance((x, y), (vx, vy)) > 100.0:
                    continue
                vehicle_status, vehicle_lane_info = self.map.map_position(vx, vy)
                if vehicle_status is not MAP_IN_ROAD:
                    continue
                if lane_info != vehicle_lane_info:
                    continue
                if lane_info['direction'] in 'NS':
                    ds = vy-y
                else:
                    ds = vx-x
                if lane_info['direction'] in 'SW':
                    ds = -ds
                if ds < 0 or ds > front_d:
                    continue
                front_x, front_y = vx, vy
                front_d = ds
                front_speed = vehicle['v']

        (steering_wheel, throttle,
         brake, gear,
         engine_speed, engine_torque,
         accl, sideslip,
         yaw_rate, lateralvelocity,
         longitudinalvelocity, frontwheel,
         steerrate, fuel_total,
         acc_lon, acc_lat, fuel_rate) = self.agent.dynamic.get_info()

        evaluation_data=((x,y,heading),(plan_x,plan_y),(lane_x, lane_y),
                         (front_x,front_y),is_in_cross,is_change_lane,
                          frontwheel, throttle/100.0, brake,
                          longitudinalvelocity, lateralvelocity, accl,
                          car2border, steerrate, fuel_rate ,acc_lon, acc_lat,
                          front_speed, speed, yaw_rate, steering_wheel,
                          engine_speed, engine_torque, gear,
                          self.traffic.get_traffic_density(),
                          self.traffic.get_current_lane_speed_limit(),
                          fuel_total,
                          self.agent.dynamic.get_total_travelled_distance())
        self.evaluator.update(input=evaluation_data, veh_info=self.veh_info)

    def get_current_task(self):
        return self.agent.mission.current_task

    def __collision_check(self):
        # for vehs in self.other_vehicles:
        #     if vehs['render']:
        #         if (fabs(vehs['x']-self.agent.dynamic.ego_x) < 10 and
        #            fabs(vehs['y']-self.agent.dynamic.ego_y) < 2):
        #             self.ego_x0 = (self.agent.dynamic.x +
        #                            cos(self.agent.dynamic.heading/180*pi)*self.agent.lw)
        #             self.ego_y0 = (self.agent.dynamic.y +
        #                            sin(self.agent.dynamic.heading/180*pi)*self.agent.lw)
        #             self.ego_x1 = (self.agent.dynamic.x -
        #                            cos(self.agent.dynamic.heading/180*pi)*self.agent.lw)
        #             self.ego_y1 = (self.agent.dynamic.y -
        #                            sin(self.agent.dynamic.heading/180*pi)*self.agent.lw)
        #             self.surrounding_lw = (vehs['length']-vehs['width'])/2
        #             self.surrounding_x0 = (
        #                 vehs['x'] + cos(
        #                     vehs['angle'] / 180 * pi) * self.surrounding_lw)
        #             self.surrounding_y0 = (
        #                 vehs['y'] + sin(
        #                     vehs['angle'] / 180 * pi) * self.surrounding_lw)
        #             self.surrounding_x1 = (
        #                 vehs['x'] - cos(
        #                     vehs['angle'] / 180 * pi) * self.surrounding_lw)
        #             self.surrounding_y1 = (
        #                 vehs['y'] - sin(
        #                     vehs['angle'] / 180 * pi) * self.surrounding_lw)
        #             self.collision_check_dis = ((vehs['width']+self.agent.width)/2+0.5)**2
        #             if ((self.ego_x0-self.surrounding_x0)**2 +
        #                 (self.ego_y0-self.surrounding_y0)**2
        #                     < self.collision_check_dis):
        #                 return False
        #             if ((self.ego_x0-self.surrounding_x1)**2 +
        #                 (self.ego_y0-self.surrounding_y1)**2
        #                     < self.collision_check_dis):
        #                 return False
        #             if ((self.ego_x1-self.surrounding_x1)**2 +
        #                 (self.ego_y1-self.surrounding_y1)**2
        #                     < self.collision_check_dis):
        #                 return False
        #             if ((self.ego_x1-self.surrounding_x0)**2 +
        #                 (self.ego_y1-self.surrounding_y0)**2
        #                     < self.collision_check_dis):
        #                 return False
        return True

    def reset_canvas(self, canvas):
        self.canvas = canvas
        self.disp.reset_canvas(canvas)

    def init_canvas(self, canvas):
        self.canvas = canvas
        self.disp.init_canvas(canvas)

    def draw(self):
        self.disp.draw()
        pass


DEFAULT_SETTING_FILE = 'Library/default_simulation_setting.xml'


class Settings:
    """
    Simulation Settings Class
    """
    max_sensor_range = 0  # 最大感知范围
    render_buffer = 20.0  # 渲染缓冲范围（渲染范围=最大感知范围+渲染缓冲范围）
    min_render_range = 180.0  # 最小渲染范围（以自车为中心，半径180m）
    car_length = 0.0  # 自车长度，m
    car_width = 0.0  # 自车宽度，m
    car_weight = 0.0  # 自车重量，kg
    lasvsim_version = None  # LasVSim版本（'gui'和'package'）

    def __init__(self, file_path=None):
        self.lasvsim_version = 'package'
        self.car_para = data_structures.CarParameter()  # 自车动力学模型参数
        self.load(file_path)

    def __del__(self):
        pass

    def load(self, filePath=None):
        if filePath is None:
            if self.lasvsim_version == 'package':
                filePath = 'LasVSim/' + DEFAULT_SETTING_FILE
            else:
                filePath = DEFAULT_SETTING_FILE
        self.__parse_xml(filePath)
        self.__load_step_length()
        self.__load_map()
        self.__load_self_car()
        self.__load_mission()
        self.__load_controller()
        self.__load_traffic()
        self.__load_sensors()
        self.__load_router()
        self.__load_dynamic()

    def get_render_range(self):
        range = self.min_render_range
        for sensor in self.sensors:
            if sensor.detection_range > range:
                range = sensor.detection_range
        return range + self.render_buffer

    def __parse_xml(self, path):
        f = open(path)
        self.root = untangle.parse(f.read()).Simulation

    def __load_step_length(self):
        self.step_length = int(self.root.StepLength.cdata)

    def __load_map(self):
        self.map = str(self.root.Map.Type.cdata)

    def __load_mission(self):
        self.mission_type = str(self.root.Mission.Type.cdata)
        self.points = []
        if self.lasvsim_version == 'gui':
            for i in range(len(self.root.Mission.Point)):
                self.points.append([float(self.root.Mission.Point[i].X.cdata),
                                    float(self.root.Mission.Point[i].Y.cdata),
                                    float(self.root.Mission.Point[i].Speed.cdata
                                          ),
                                    -float(self.root.Mission.Point[i].Yaw.cdata)
                                    + 90])
        else:
            for i in range(len(self.root.Mission.Point)):
                self.points.append([float(self.root.Mission.Point[i].X.cdata),
                                    float(self.root.Mission.Point[i].Y.cdata),
                                    float(self.root.Mission.Point[i].Speed.cdata
                                          ),
                                    float(self.root.Mission.Point[i].Yaw.cdata)]
                                   )

    def __load_controller(self):
        self.controller_type = str(self.root.Controller.Type.cdata)
        self.controller_lib = str(self.root.Controller.Lib.cdata)
        self.controller_frequency = int(self.root.Controller.Frequency.cdata)
        self.controller_file_type = str(self.root.Controller.FileType.cdata)

    def __load_dynamic(self):
        self.dynamic_type=str(self.root.Dynamic.Type.cdata)
        self.dynamic_lib=str(self.root.Dynamic.Lib.cdata)
        self.dynamic_frequency = int(self.root.Dynamic.Frequency.cdata)

    def __load_traffic(self):
        self.traffic_type = str(self.root.Traffic.Type.cdata)
        self.traffic_lib = str(self.root.Traffic.Lib.cdata)
        self.traffic_frequency = int(self.root.Traffic.Frequency.cdata)

    def __load_self_car(self):
        self.car_length = float(self.root.SelfCar.Length.cdata)
        self.car_width = float(self.root.SelfCar.Width.cdata)
        self.car_weight = float(self.root.SelfCar.Weight.cdata)
        self.car_center2head = float(self.root.SelfCar.CenterToHead.cdata)
        self.car_faxle2center = float(self.root.SelfCar.FAxleToCenter.cdata)
        self.car_raxle2center = float(self.root.SelfCar.RAxleToCenter.cdata)
        self.car_para.LX_AXLE = self.car_faxle2center + self.car_raxle2center
        self.car_para.LX_CG_SU = self.car_faxle2center
        self.car_para.M_SU = float(self.root.SelfCar.M_SU.cdata)
        self.car_para.IZZ_SU = float(self.root.SelfCar.IZZ_SU.cdata)
        self.car_para.A = float(self.root.SelfCar.A.cdata)
        self.car_para.CFx = float(self.root.SelfCar.CFx.cdata)
        self.car_para.AV_ENGINE_IDLE = float(self.root.SelfCar.AV_ENGINE_IDLE.cdata)
        self.car_para.IENG = float(self.root.SelfCar.IENG.cdata)
        self.car_para.TAU = float(self.root.SelfCar.TAU.cdata)
        self.car_para.R_GEAR_TR1 = float(self.root.SelfCar.R_GEAR_TR1.cdata)
        self.car_para.R_GEAR_FD = float(self.root.SelfCar.R_GEAR_FD.cdata)
        self.car_para.BRAK_COEF = float(self.root.SelfCar.BRAK_COEF.cdata)
        self.car_para.Steer_FACTOR = float(self.root.SelfCar.Steer_FACTOR.cdata)
        self.car_para.M_US = float(self.root.SelfCar.M_US.cdata)
        self.car_para.RRE = float(self.root.SelfCar.RRE.cdata)
        self.car_para.CF = float(self.root.SelfCar.CF.cdata)
        self.car_para.CR = float(self.root.SelfCar.CR.cdata)
        self.car_para.ROLL_RESISTANCE = float(self.root.SelfCar.ROLL_RESISTANCE.cdata)

    def __load_sensors(self):
        self.sensor_model_type = str(self.root.Sensors.Type.cdata)
        self.sensor_model_lib = str(self.root.Sensors.Lib.cdata)
        self.sensor_frequency = int(self.root.Sensors.Frequency.cdata)
        sensor_array = SensorInfo * len(self.root.Sensors.Sensor)
        self.sensors = sensor_array()
        types = ['int', 'int', 'float', 'float', 'float', 'float', 'float',
                 'float', 'float', 'float', 'float', 'float', 'float', 'float']
        attrs = ['id',
                 'type',
                 'detection_angle',
                 'detection_range',
                 'installation_lateral_bias',
                 'installation_longitudinal_bias',
                 'installation_orientation_angle',
                 'accuracy_velocity',
                 'accuracy_location',
                 'accuracy_heading',
                 'accuracy_width',
                 'accuracy_length',
                 'accuracy_height',
                 'accuracy_radius']
        for i in range(len(self.sensors)):
            for j in range(len(attrs)):
                attr = attrs[j]
                dtype = types[j]
                exec(('self.sensors[i].%s=%s(self.root.Sensors.Sensor[i].%'
                      's.cdata)') % (attr, dtype, attr))

    def __load_router(self):
        self.router_output_type = str(self.root.Router.OutputType.cdata)
        self.router_type = str(self.root.Router.Type.cdata)
        self.router_lib = str(self.root.Router.Lib.cdata)
        self.router_frequency = int(self.root.Router.Frequency.cdata)
        self.router_file_type = str(self.root.Router.FileType.cdata)

    def save(self,path):
        file_name=re.split(r'[\\/]',str(path))[-1].split('.')[0]
        doc = Document();
        root = doc.createElement('Simulation')
        doc.appendChild(root)
        info_node=doc.createElement('Info')
        title_node=doc.createElement('Title')
        date_node=doc.createElement('Date')
        author_node=doc.createElement('Author')
        version_node=doc.createElement('Version')
        title_node.appendChild(doc.createTextNode(file_name))
        date_node.appendChild(doc.createTextNode(time.strftime('%Y-%m-%d')))
        author_node.appendChild(doc.createTextNode('Author Name'))
        version_node.appendChild(doc.createTextNode('1.0'))
        info_node.appendChild(title_node)
        info_node.appendChild(date_node)
        info_node.appendChild(author_node)
        info_node.appendChild(version_node)

        step_node = doc.createElement('StepLength')
        step_node.appendChild(doc.createTextNode(str(self.step_length)))

        """Save map info."""
        map_node = doc.createElement('Map')
        type_node = doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.map))

        map_node.appendChild(type_node)

        """Save mission info."""
        mission_node = doc.createElement('Mission')
        type_node = doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.mission_type))
        mission_node.appendChild(type_node)
        for i in range(len(self.points)):
            point_node = doc.createElement('Point')
            mission_node.appendChild(point_node)

            x_node = doc.createElement('X')
            x_node.appendChild(doc.createTextNode('%.3f' % self.points[i][0]))
            y_node = doc.createElement('Y')
            y_node.appendChild(doc.createTextNode('%.3f' % self.points[i][1]))
            speed_node = doc.createElement('Speed')
            speed_node.appendChild(doc.createTextNode('%.0f' % self.points[i][2]))
            yaw_node = doc.createElement('Yaw')
            yaw_node.appendChild(doc.createTextNode('%.0f' % (-self.points[i][3]+90)))
            point_node.appendChild(x_node)
            point_node.appendChild(y_node)
            point_node.appendChild(speed_node)
            point_node.appendChild(yaw_node)

        # 保存自车动力学模型参数
        selfcar_node = doc.createElement('SelfCar')

        length_node=doc.createElement('Length')  # 车长
        length_node.appendChild(doc.createTextNode(str(self.car_length)))
        selfcar_node.appendChild(length_node)

        width_node=doc.createElement('Width')  # 车宽
        width_node.appendChild(doc.createTextNode(str(self.car_width)))
        selfcar_node.appendChild(width_node)

        CenterToHead_node=doc.createElement('CenterToHead')  # 质心距车头距离
        CenterToHead_node.appendChild(doc.createTextNode('2.7'))
        selfcar_node.appendChild(CenterToHead_node)

        FAxleToCenter_node=doc.createElement('FAxleToCenter')  # 悬上质量质心至前轴距离，m
        FAxleToCenter_node.appendChild(doc.createTextNode(str(
            self.car_para.LX_CG_SU)))
        selfcar_node.appendChild(FAxleToCenter_node)

        RAxleToCenter_node=doc.createElement('RAxleToCenter')  # 悬上质量质心至后轴距离，m
        RAxleToCenter_node.appendChild(doc.createTextNode(str(
            self.car_para.LX_AXLE-self.car_para.LX_CG_SU)))
        selfcar_node.appendChild(RAxleToCenter_node)

        Weight_node = doc.createElement('Weight')  # 质量
        Weight_node.appendChild(doc.createTextNode(str(
            self.car_para.M_SU+self.car_para.M_US)))
        selfcar_node.appendChild(Weight_node)

        LX_AXLE = doc.createElement('LX_AXLE')  # 悬上质量，kg
        LX_AXLE.appendChild(doc.createTextNode(str(self.car_para.LX_AXLE)))
        selfcar_node.appendChild(LX_AXLE)

        LX_CG_SU = doc.createElement('LX_CG_SU')  # 悬上质量，kg
        LX_CG_SU.appendChild(doc.createTextNode(str(self.car_para.LX_CG_SU)))
        selfcar_node.appendChild(LX_CG_SU)

        M_SU = doc.createElement('M_SU')  # 悬上质量，kg
        M_SU.appendChild(doc.createTextNode(str(self.car_para.M_SU)))
        selfcar_node.appendChild(M_SU)

        IZZ_SU = doc.createElement('IZZ_SU')  # 转动惯量，kg*m^2
        IZZ_SU.appendChild(doc.createTextNode(str(self.car_para.IZZ_SU)))
        selfcar_node.appendChild(IZZ_SU)

        A_Wind = doc.createElement('A')  # 迎风面积，m^2
        A_Wind.appendChild(doc.createTextNode(str(self.car_para.A)))
        selfcar_node.appendChild(A_Wind)

        CFx = doc.createElement('CFx')  # 空气动力学侧偏角为零度时的纵向空气阻力系数
        CFx.appendChild(doc.createTextNode(str(self.car_para.CFx)))
        selfcar_node.appendChild(CFx)

        AV_ENGINE_IDLE = doc.createElement('AV_ENGINE_IDLE')  # 怠速转速，rpm
        AV_ENGINE_IDLE.appendChild(doc.createTextNode(str(
            self.car_para.AV_ENGINE_IDLE)))
        selfcar_node.appendChild(AV_ENGINE_IDLE)

        IENG = doc.createElement('IENG')  # 曲轴转动惯量，kg*m^2
        IENG.appendChild(doc.createTextNode(str(self.car_para.IENG)))
        selfcar_node.appendChild(IENG)

        TAU = doc.createElement('TAU')  # 发动机-变速箱输入轴 时间常数，s
        TAU.appendChild(doc.createTextNode(str(self.car_para.TAU)))
        selfcar_node.appendChild(TAU)

        R_GEAR_TR1 = doc.createElement('R_GEAR_TR1')  # 最低档变速箱传动比
        R_GEAR_TR1.appendChild(doc.createTextNode(str(self.car_para.R_GEAR_TR1)))
        selfcar_node.appendChild(R_GEAR_TR1)

        R_GEAR_FD = doc.createElement('R_GEAR_FD')  # 主减速器传动比
        R_GEAR_FD.appendChild(doc.createTextNode(str(self.car_para.R_GEAR_FD)))
        selfcar_node.appendChild(R_GEAR_FD)

        BRAK_COEF = doc.createElement('BRAK_COEF')  # 液压缸变矩系数,Nm/(MPa)
        BRAK_COEF.appendChild(doc.createTextNode(str(self.car_para.BRAK_COEF)))
        selfcar_node.appendChild(BRAK_COEF)

        Steer_FACTOR = doc.createElement('Steer_FACTOR')  # 转向传动比
        Steer_FACTOR.appendChild(doc.createTextNode(str(
            self.car_para.Steer_FACTOR)))
        selfcar_node.appendChild(Steer_FACTOR)

        M_US = doc.createElement('M_US')  # 簧下质量，kg
        M_US.appendChild(doc.createTextNode(str(self.car_para.M_US)))
        selfcar_node.appendChild(M_US)

        RRE = doc.createElement('RRE')  # 车轮有效滚动半径，m
        RRE.appendChild(doc.createTextNode(str(self.car_para.RRE)))
        selfcar_node.appendChild(RRE)

        CF = doc.createElement('CF')  # 前轮侧偏刚度，N/rad
        CF.appendChild(doc.createTextNode(str(self.car_para.CF)))
        selfcar_node.appendChild(CF)

        CR = doc.createElement('CR')  # 后轮侧偏刚度，N/rad
        CR.appendChild(doc.createTextNode(str(self.car_para.CR)))
        selfcar_node.appendChild(CR)

        ROLL_RESISTANCE = doc.createElement('ROLL_RESISTANCE')  # 滚动阻力系数
        ROLL_RESISTANCE.appendChild(doc.createTextNode(str(
            self.car_para.ROLL_RESISTANCE)))
        selfcar_node.appendChild(ROLL_RESISTANCE)

        """Save traffic setting info."""
        traffic_node = doc.createElement('Traffic')
        type_node=doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.traffic_type))
        traffic_node.appendChild(type_node)
        lib_node = doc.createElement('Lib')
        lib_node.appendChild(doc.createTextNode(self.traffic_lib))
        traffic_node.appendChild(lib_node)
        fre_node = doc.createElement('Frequency')
        fre_node.appendChild(doc.createTextNode(str(self.traffic_frequency)))
        traffic_node.appendChild(fre_node)

        """Save controller setting info."""
        control_node = doc.createElement('Controller')
        type_node=doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.controller_type))
        control_node.appendChild(type_node)
        lib_node = doc.createElement('Lib')
        lib_node.appendChild(doc.createTextNode(self.controller_lib))
        control_node.appendChild(lib_node)
        fre_node = doc.createElement('Frequency')
        fre_node.appendChild(doc.createTextNode(str(self.controller_frequency)))
        control_node.appendChild(fre_node)
        file_node = doc.createElement('FileType')
        file_node.appendChild(doc.createTextNode(self.controller_file_type))
        control_node.appendChild(file_node)

        """Save vehicle dynamic model parameters."""
        dynamic_node = doc.createElement('Dynamic')
        type_node = doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.dynamic_type))
        dynamic_node.appendChild(type_node)
        lib_node = doc.createElement('Lib')
        lib_node.appendChild(doc.createTextNode(self.dynamic_lib))
        dynamic_node.appendChild(lib_node)
        fre_node = doc.createElement('Frequency')
        fre_node.appendChild(doc.createTextNode(str(self.dynamic_frequency)))
        dynamic_node.appendChild(fre_node)

        """Save decision module info."""
        router_node = doc.createElement('Router')
        output_type_node = doc.createElement('OutputType')
        output_type_node.appendChild(doc.createTextNode(self.router_output_type))
        router_node.appendChild(output_type_node)
        type_node=doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.router_type))
        router_node.appendChild(type_node)
        lib_node = doc.createElement('Lib')
        lib_node.appendChild(doc.createTextNode(self.router_lib))
        router_node.appendChild(lib_node)
        fre_node = doc.createElement('Frequency')
        fre_node.appendChild(doc.createTextNode(str(self.router_frequency)))
        router_node.appendChild(fre_node)
        file_node = doc.createElement('FileType')
        file_node.appendChild(doc.createTextNode(self.router_file_type))
        router_node.appendChild(file_node)

        """Save sensor module info."""
        sensors_node = doc.createElement('Sensors')
        type_node=doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.sensor_model_type))
        sensors_node.appendChild(type_node)
        lib_node = doc.createElement('Lib')
        lib_node.appendChild(doc.createTextNode(self.sensor_model_lib))
        sensors_node.appendChild(lib_node)
        fre_node = doc.createElement('Frequency')
        fre_node.appendChild(doc.createTextNode(str(self.sensor_frequency)))
        sensors_node.appendChild(fre_node)
        for i in range(len(self.sensors)):
            s = self.sensors[i]
            s_node = doc.createElement('Sensor')
            sensors_node.appendChild(s_node)

            ID_node = doc.createElement('id')
            ID_node.appendChild(doc.createTextNode('%d' % i))
            s_node.appendChild(ID_node)

            Type_node = doc.createElement('type')
            Type_node.appendChild(doc.createTextNode('%d' % s.type))
            s_node.appendChild(Type_node)

            Angle_node = doc.createElement('detection_angle')
            Angle_node.appendChild(
                doc.createTextNode('%.0f' % s.detection_angle))
            s_node.appendChild(Angle_node)

            Radius_node = doc.createElement('detection_range')
            Radius_node.appendChild(
                doc.createTextNode('%.1f' % s.detection_range))
            s_node.appendChild(Radius_node)

            Installation_lat_node=doc.createElement('installation_lateral_bias')
            Installation_lat_node.appendChild(
                doc.createTextNode('%.3f' % s.installation_lateral_bias))
            s_node.appendChild(Installation_lat_node)

            Installation_long_node = doc.createElement(
                'installation_longitudinal_bias')
            Installation_long_node.appendChild(
                doc.createTextNode('%.3f'% s.installation_longitudinal_bias))
            s_node.appendChild(Installation_long_node)

            Orientation_node=doc.createElement('installation_orientation_angle')
            Orientation_node.appendChild(
                doc.createTextNode('%.0f' % s.installation_orientation_angle))
            s_node.appendChild(Orientation_node)

            Accuracy_Vel_node=doc.createElement('accuracy_velocity')
            Accuracy_Vel_node.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_velocity))
            s_node.appendChild(Accuracy_Vel_node)

            Accuracy_Location_node=doc.createElement('accuracy_location')
            Accuracy_Location_node.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_location))
            s_node.appendChild(Accuracy_Location_node)

            Accuracy_Yaw_node=doc.createElement('accuracy_heading')
            Accuracy_Yaw_node.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_heading))
            s_node.appendChild(Accuracy_Yaw_node)

            Accuracy_Width_node=doc.createElement('accuracy_width')
            Accuracy_Width_node.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_width))
            s_node.appendChild(Accuracy_Width_node)

            Accuracy_Length_node=doc.createElement('accuracy_length')
            Accuracy_Length_node.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_length))
            s_node.appendChild(Accuracy_Length_node)

            Detect_Turnlight=doc.createElement('accuracy_height')
            Detect_Turnlight.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_height))
            s_node.appendChild(Detect_Turnlight)

            Detect_Vehicletype=doc.createElement('accuracy_radius')
            Detect_Vehicletype.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_radius))
            s_node.appendChild(Detect_Vehicletype)

        root.appendChild(info_node)
        root.appendChild(step_node)
        root.appendChild(map_node)
        root.appendChild(mission_node)
        root.appendChild(selfcar_node)
        root.appendChild(traffic_node)
        root.appendChild(control_node)
        root.appendChild(dynamic_node)
        root.appendChild(router_node)
        root.appendChild(sensors_node)

        buffer = io.StringIO()
        doc.writexml(buffer, addindent="\t", newl='\n', encoding='utf-8')
        txt = re.sub('\n\t+[^<^>]*\n\t+',
                     lambda x: re.sub('[\t\n]', '', x.group(0)),
                     buffer.getvalue())
        open(path, 'w').write(txt)


if __name__ == "__main__":
    class AAA(object):
        def __init__(self,a):
            self.a = a
    def ppp(a):
        a.a = 33
        print(a.a)
    b = AAA(0)
    ppp(b)
    print(b.a)


