# # coding=utf-8
# """
# Setting Dialog of Autonomous Car Simulation System
# Author: Li Bing
# Date: 2017-8-23
# """
#
# import copy
# import os
# # from PyQt4.Qt import *
# # from PyQt4 import QtCore, QtGui,uic
# import LasVSim.data_module
# from LasVSim.navigation_module import *
# from LasVSim.simulator import *
#
# from LasVSim.sensor_setting import *
# from LasVSim.data_structures import *
#
#
# class SettingsDialog(QtGui.QDialog):
#     """仿真配置界面对象
#
#     Args:
#     """
#
#     def __init__(self, parent=None):
#         # 读取QT文件
#         QtGui.QDialog.__init__(self, parent)
#         uic.loadUi('QT UI/Settings.ui', self)
#         font = QFont()
#         font.setPixelSize(12)
#         self.setFont(font)
#
#         # 设置窗口的logo
#         icon = QIcon()
#         icon.addPixmap(QPixmap("Resources/Logos/setting.ico"), QIcon.Normal)
#         self.setWindowIcon(icon)
#
#         # 设置窗口响应事件
#         self.buttonOK.clicked.connect(self.on_ok)
#         self.buttonCancel.clicked.connect(self.on_cancel)
#         # 地图相关响应事件
#         self.buttonMap.clicked.connect(self.on_map)
#         self.comboMap.currentIndexChanged.connect(self.on_map_change)
#         # 交通流相关响应事件
#         self.comboTraffic.currentIndexChanged.connect(
#             self.on_traffic_type_change)
#         self.buttonTraffic.clicked.connect(self.on_traffic_parameter)
#         self.comboTrafficDensity.currentIndexChanged.connect(
#             self.on_traffic_density_change)
#         # 动力学相关响应事件
#         self.buttonModel.clicked.connect(self.on_dynamic)
#         self.comboModel.currentIndexChanged.connect(self.on_dynamic_change)
#         # 传感器相关响应事件
#         self.buttonSensorEdit.clicked.connect(self.on_sensor_edit)
#         # 决策相关响应事件
#         self.comboDecisionInternal.currentIndexChanged.connect(
#             self.on_planner_type_change)
#         self.comboDecisionType.currentIndexChanged.connect(
#             self.on_planner_output_change)
#         self.buttonBrowseDecision.clicked.connect(self.on_browse_planner)
#         # 控制器相关响应事件
#         self.buttonBrowseController.clicked.connect(self.on_browse_controller)
#         self.comboControllerInternal.currentIndexChanged.connect(
#             self.on_controller_type_change)
#
#         self.settings = None  # 仿真设置对象
#         self.editDynamicPath = ''
#         self.points = []  # 全局路径点
#         self.mission_type = None  # 任务类型：单向/循环
#
#     def init_view(self, settings):
#         """初始化UI界面。
#
#         Args：
#         """
#         self.settings = settings
#         self.points = settings.points
#         self.mission_type = settings.mission_type
#         self.__refresh()  # 初始窗口时需要刷新窗口两次
#         self.__refresh()  # 因为前三次刷新窗口每次窗口都会变化
#         self.__refresh()
#
#     def __refresh(self):
#         """刷新窗口界面。"""
#
#         # 刷新传感器配置信息
#         self.set_sensors(self.settings.sensors)
#         self.editSensorsPath.setText(self.settings.sensor_model_lib)
#         self.editSensorRatio.setText(str(self.settings.sensor_frequency))
#
#         # 刷新地图缩略图
#         if self.settings.map == MAPS[0]:
#             img = QImage('Resources/Tools/urban_road_map.png')
#             self.comboMap.setCurrentIndex(0)
#         elif self.settings.map == MAPS[1]:
#             img = QImage('Resources/Tools/highway_map.png')
#             self.comboMap.setCurrentIndex(1)
#             self.points = [[-876.35, -5.74, 20.0, 90.0],
#                            [876.35, -5.74, 22.22, 90.0]]
#             pass  # TODO(Chason): 后期加入其余地图缩略图
#         else:
#             img = QImage('Resources/Tools/urban_road_map.png')
#         self.label_map.setPixmap(QPixmap.fromImage(img))
#
#         # 刷新交通流配置信息
#         if self.settings.traffic_type == TRAFFIC_TYPE[NO_TRAFFIC]:
#             self.comboTraffic.setCurrentIndex(0)
#             self.comboTrafficDensity.setEnabled(0)
#             self.buttonTraffic.setEnabled(0)
#         elif self.settings.traffic_type == TRAFFIC_TYPE[MIXED_TRAFFIC]:
#             self.comboTraffic.setCurrentIndex(1)
#             self.comboTrafficDensity.setEnabled(1)
#             self.buttonTraffic.setEnabled(1)
#         else:
#             self.comboTraffic.setCurrentIndex(2)
#             self.comboTrafficDensity.setEnabled(1)
#             self.buttonTraffic.setEnabled(1)
#         if self.settings.traffic_lib == TRAFFIC_DENSITY[MIDDLE]:
#             self.comboTrafficDensity.setCurrentIndex(1)
#         elif self.settings.traffic_lib == TRAFFIC_DENSITY[DENSE]:
#             self.comboTrafficDensity.setCurrentIndex(2)
#         elif self.settings.traffic_lib == TRAFFIC_DENSITY[SPARSE]:
#             self.comboTrafficDensity.setCurrentIndex(0)
#         if self.settings.map == MAPS[1] and self.comboTraffic.currentIndex() == 1:
#             self.settings.traffic_type = TRAFFIC_TYPE[VEHICLE_ONLY_TRAFFIC]
#             self.comboTraffic.setCurrentIndex(VEHICLE_ONLY_TRAFFIC)
#         self.editTrafficRatio.setText(str(self.settings.traffic_frequency))
#
#         # 刷新控制器配置信息
#         if self.settings.controller_type == CONTROLLER_TYPE[0]:
#             self.comboControllerInternal.setCurrentIndex(0)
#             self.editControllerPath.setEnabled(0)
#             self.buttonBrowseController.setEnabled(0)
#             self.editControllerPath.setText('Modules/Controller.dll')
#         elif self.settings.controller_type == CONTROLLER_TYPE[1]:
#             self.editControllerPath.setText('-')
#             self.editControllerPath.setEnabled(1)
#             self.buttonBrowseController.setEnabled(1)
#             if self.settings.controller_file_type == 'C/C++ DLL':
#                 self.comboControllerInternal.setCurrentIndex(1)
#             else:
#                 self.comboControllerInternal.setCurrentIndex(2)
#         self.editControllRatio.setText(str(self.settings.controller_frequency))
#
#         # 刷新动力学配置信息
#         if self.settings.dynamic_type == CAR_TYPE[AMT_CAR]:
#             self.comboModel.setCurrentIndex(AMT_CAR)
#         elif self.settings.dynamic_type == CAR_TYPE[CVT_CAR]:
#             self.comboModel.setCurrentIndex(CVT_CAR)
#         elif self.settings.dynamic_type == CAR_TYPE[TRUCK]:
#             self.comboModel.setCurrentIndex(TRUCK)
#         elif self.settings.dynamic_type == CAR_TYPE[EV]:
#             self.comboModel.setCurrentIndex(EV)
#         self.editWeight.setText(str(self.settings.car_para.M_SU +
#                                     self.settings.car_para.M_US))
#         self.editLength.setText(str(self.settings.car_length))
#         self.editWidth.setText(str(self.settings.car_width))
#
#         self.editDynamicRatio.setText(str(self.settings.dynamic_frequency))
#
#         # 刷新决策模块配置信息
#         if self.settings.router_type == DECISION_TYPE[MAP1_XINLONG]:
#             if self.settings.map == MAPS[1]:
#                 # 地图与算法不匹配
#                 self.comboDecisionInternal.setCurrentIndex(MAP2_XINLONG)
#                 self.settings.router_lib = DEFAULT_DECISION_FILE_PATH[MAP2_XINLONG-2]
#             else:
#                 self.comboDecisionInternal.setCurrentIndex(MAP1_XINLONG)
#                 self.settings.router_lib = DEFAULT_DECISION_FILE_PATH[MAP1_XINLONG-2]
#             self.editDecisionPath.setEnabled(0)
#             self.buttonBrowseDecision.setEnabled(0)
#             self.comboDecisionType.setEnabled(0)
#             self.comboDecisionType.setCurrentIndex(SPATIO_TEMPORAL_TRAJECTORY)
#         elif self.settings.router_type == DECISION_TYPE[MAP2_XINLONG]:
#             if self.settings.map == MAPS[0]:
#                 # 地图与算法不匹配
#                 self.comboDecisionInternal.setCurrentIndex(MAP1_XINLONG)
#                 self.settings.router_lib = DEFAULT_DECISION_FILE_PATH[MAP1_XINLONG-2]
#             else:
#                 self.comboDecisionInternal.setCurrentIndex(MAP2_XINLONG)
#                 self.settings.router_lib = DEFAULT_DECISION_FILE_PATH[MAP2_XINLONG-2]
#             self.editDecisionPath.setEnabled(0)
#             self.buttonBrowseDecision.setEnabled(0)
#             self.comboDecisionType.setEnabled(0)
#             self.comboDecisionType.setCurrentIndex(SPATIO_TEMPORAL_TRAJECTORY)
#         elif self.settings.router_type == DECISION_TYPE[MAP1_CHENXIANG]:
#             if self.settings.map == MAPS[1]:
#                 # 地图与算法不匹配
#                 self.comboDecisionInternal.setCurrentIndex(MAP2_CHENXIANG)
#                 self.settings.router_lib = DEFAULT_DECISION_FILE_PATH[MAP2_CHENXIANG-2]
#             else:
#                 self.comboDecisionInternal.setCurrentIndex(MAP1_CHENXIANG)
#                 self.settings.router_lib = DEFAULT_DECISION_FILE_PATH[MAP1_CHENXIANG-2]
#             self.editDecisionPath.setEnabled(0)
#             self.buttonBrowseDecision.setEnabled(0)
#             self.comboDecisionType.setEnabled(0)
#             self.comboDecisionType.setCurrentIndex(SPATIO_TEMPORAL_TRAJECTORY)
#         elif self.settings.router_type == DECISION_TYPE[MAP2_CHENXIANG]:
#             if self.settings.map == MAPS[0]:
#                 # 地图与算法不匹配
#                 self.comboDecisionInternal.setCurrentIndex(MAP1_CHENXIANG)
#                 self.settings.router_lib = DEFAULT_DECISION_FILE_PATH[MAP1_CHENXIANG-2]
#             else:
#                 self.comboDecisionInternal.setCurrentIndex(MAP2_CHENXIANG)
#                 self.settings.router_lib = DEFAULT_DECISION_FILE_PATH[MAP2_CHENXIANG-2]
#             self.editDecisionPath.setEnabled(0)
#             self.buttonBrowseDecision.setEnabled(0)
#             self.comboDecisionType.setEnabled(0)
#             self.comboDecisionType.setCurrentIndex(SPATIO_TEMPORAL_TRAJECTORY)
#         else:
#             self.editDecisionPath.setEnabled(1)
#             self.buttonBrowseDecision.setEnabled(1)
#             self.comboDecisionType.setEnabled(1)
#             if self.settings.router_output_type == DECISION_OUTPUT_TYPE[
#                     SPATIO_TEMPORAL_TRAJECTORY]:
#                 self.comboDecisionType.setCurrentIndex(
#                     SPATIO_TEMPORAL_TRAJECTORY)
#             elif self.settings.router_output_type == DECISION_OUTPUT_TYPE[
#                 DYNAMIC_OUTPUT]:
#                 self.comboDecisionType.setCurrentIndex(DYNAMIC_OUTPUT)
#             else:
#                 self.comboDecisionType.setCurrentIndex(KINEMATIC_OUTPUT)
#             if self.settings.router_file_type == FILE_TYPE[0]:
#                 self.comboDecisionInternal.setCurrentIndex(0)
#             else:
#                 self.comboDecisionInternal.setCurrentIndex(1)
#         self.editDecisionRatio.setText(str(self.settings.router_frequency))
#         self.editDecisionPath.setText(self.settings.router_lib)
#
#         self.editStepLength.setText(str(self.settings.step_length))
#
#     def on_traffic_type_change(self):
#         if self.settings.map == MAPS[1]:
#             if self.comboTraffic.currentIndex() == MIXED_TRAFFIC:
#                 self.comboTraffic.setCurrentIndex(VEHICLE_ONLY_TRAFFIC)
#         self.settings.traffic_type = TRAFFIC_TYPE[
#             self.comboTraffic.currentIndex()]
#         self.__refresh()
#
#     def on_traffic_density_change(self):
#         self.settings.traffic_lib = TRAFFIC_DENSITY[
#             self.comboTrafficDensity.currentIndex()]
#         self.__refresh()
#
#     def on_controller_type_change(self):
#         self.settings.controller_type = CONTROLLER_TYPE[
#             self.comboControllerInternal.currentIndex()]
#         if self.comboControllerInternal.currentIndex() == 0:
#             self.settings.controller_lib = CONTROLLER_FILE_PATH
#             self.settings.controller_file_type = FILE_TYPE[0]
#         else:
#             self.settings.controller_lib = '-'
#             self.settings.controller_file_type = FILE_TYPE[
#                 self.comboControllerInternal.currentIndex()-1]
#         self.__refresh()
#
#     def on_dynamic_change(self):
#         if self.comboModel.currentIndex() == CVT_CAR:
#             self.settings.car_width = 1.87
#             self.settings.car_length = 5.20
#             self.settings.dynamic_lib = CVT_MODEL_FILE_PATH
#             self.settings.car_para.CFx = 0.3
#             self.settings.car_para.R_GEAR_FD = 2.65
#             self.settings.car_para.LX_AXLE = 3.16
#             self.settings.car_para.LX_CG_SU = 1.265
#             self.settings.car_para.M_SU = 1820
#             self.settings.car_para.IZZ_SU = 4095.0
#             self.settings.car_para.A = 3.0
#             self.settings.car_para.IENG = 0.4
#             self.settings.car_para.Steer_FACTOR = 16.5
#             self.settings.car_para.M_US = 200
#             self.settings.car_para.RRE = 0.353
#             self.settings.car_para.CF = -128915.5
#             self.settings.car_para.CR = -117481.8
#             self.settings.car_para.AV_ENGINE_IDLE = 750
#             self.settings.car_para.R_GEAR_TR1 = 4.6
#             self.settings.car_para.ROLL_RESISTANCE = 0.0041
#             self.settings.car_para.BRAK_COEF = 1100.0
#             self.settings.car_para.TAU = 0.3
#         elif self.comboModel.currentIndex() == AMT_CAR:
#             self.settings.dynamic_lib = AMT_MODEL_FILE_PATH
#             self.settings.car_width = 1.916
#             self.settings.car_length = 4.5
#             self.settings.car_para.CFx = 0.30
#             self.settings.car_para.R_GEAR_FD = 4.10
#             self.settings.car_para.LX_AXLE = 2.91
#             self.settings.car_para.LX_CG_SU = 1.015
#             self.settings.car_para.M_SU = 1270
#             self.settings.car_para.IZZ_SU = 1536.7
#             self.settings.car_para.A = 2.2
#             self.settings.car_para.IENG = 0.16
#             self.settings.car_para.Steer_FACTOR = 14.3
#             self.settings.car_para.M_US = 142
#             self.settings.car_para.RRE = 0.325
#             self.settings.car_para.CF = -128915.5
#             self.settings.car_para.CR = -85943.6
#             self.settings.car_para.AV_ENGINE_IDLE = 750
#             self.settings.car_para.R_GEAR_TR1 = 3.538
#             self.settings.car_para.ROLL_RESISTANCE = 0.0038
#             self.settings.car_para.BRAK_COEF = 800.0
#             self.settings.car_para.TAU = 0.2
#         elif self.comboModel.currentIndex() == TRUCK:
#             self.settings.dynamic_lib = TRUCK_MODEL_FILE_PATH
#             self.settings.car_width = 2.1
#             self.settings.car_length = 6.2
#             self.settings.car_para.CFx = 0.69
#             self.settings.car_para.R_GEAR_FD = 5.0
#             self.settings.car_para.LX_AXLE = 5.0
#             self.settings.car_para.LX_CG_SU = 1.11
#             self.settings.car_para.M_SU = 4455.0
#             self.settings.car_para.IZZ_SU = 34802.6  # 34802.6
#             self.settings.car_para.A = 6.8
#             self.settings.car_para.IENG = 1.4
#             self.settings.car_para.Steer_FACTOR = 25.0
#             self.settings.car_para.M_US = 1305.0
#             self.settings.car_para.RRE = 0.51
#             self.settings.car_para.CF = -319703.0  # -166000
#             self.settings.car_para.CR = -97687.0  # -181000
#             self.settings.car_para.AV_ENGINE_IDLE = 750.0
#             self.settings.car_para.R_GEAR_TR1 = 7.59
#             self.settings.car_para.ROLL_RESISTANCE = 0.0041
#             self.settings.car_para.BRAK_COEF = 40000.0 / 7.0
#             self.settings.car_para.TAU = 0.5
#         elif self.comboModel.currentIndex() == EV:
#             self.settings.dynamic_lib = EV_MODEL_FILE_PATH
#             self.settings.car_width = 1.8  # 待定
#             self.settings.car_length = 4.5  # 待定
#             self.settings.car_para.CFx = 0.28
#             self.settings.car_para.R_GEAR_FD = 7.94
#             self.settings.car_para.LX_AXLE = 2.7
#             self.settings.car_para.LX_CG_SU = 1.0  # 待定
#             self.settings.car_para.M_SU = 1400  # ，但是总质量1521系网上查到。不影响计算
#             self.settings.car_para.IZZ_SU = 1536  # 待定
#             self.settings.car_para.A = 2.22  # 待定
#             self.settings.car_para.IENG = 0.21  # 电机不考虑该参数
#             self.settings.car_para.Steer_FACTOR = 15.0
#             self.settings.car_para.M_US = 136  # 待定，但是总质量1521系网上查到。不影响计算
#             self.settings.car_para.RRE = 0.316
#             self.settings.car_para.CF = -99323.8  # 待定
#             self.settings.car_para.CR = -60348.6  # 待定
#             self.settings.car_para.AV_ENGINE_IDLE = 750  # 待定
#             self.settings.car_para.R_GEAR_TR1 = 3.62  # 待定
#             self.settings.car_para.ROLL_RESISTANCE = 0.014  # 滚动阻力（网上查到）   良好的沥青或混凝土路面	0.010-0.018
#             self.settings.car_para.BRAK_COEF = 800.0  # 待定
#             self.settings.car_para.TAU = 0.1  # 电机不考虑该参数
#         self.settings.dynamic_type = CAR_TYPE[self.comboModel.currentIndex()]
#         self.__refresh()
#
#     def on_planner_type_change(self):
#         """
#         决策算法选择
#
#         Returns:
#
#         """
#         self.settings.router_type = DECISION_TYPE[
#             self.comboDecisionInternal.currentIndex()]
#         if self.comboDecisionInternal.currentIndex() == MAP1_XINLONG:
#             self.settings.router_lib = DEFAULT_DECISION_FILE_PATH[MAP1_XINLONG-2]
#             self.settings.router_output_type = DECISION_OUTPUT_TYPE[
#                 SPATIO_TEMPORAL_TRAJECTORY]
#             self.settings.router_file_type = FILE_TYPE[C_DLL_FILE]
#         elif self.comboDecisionInternal.currentIndex() == MAP2_XINLONG:
#             self.settings.router_lib = DEFAULT_DECISION_FILE_PATH[MAP2_XINLONG-2]
#             self.settings.router_output_type = DECISION_OUTPUT_TYPE[
#                 SPATIO_TEMPORAL_TRAJECTORY]
#             self.settings.router_file_type = FILE_TYPE[C_DLL_FILE]
#         elif self.comboDecisionInternal.currentIndex() == MAP1_CHENXIANG:
#             self.settings.router_lib = DEFAULT_DECISION_FILE_PATH[MAP1_CHENXIANG-2]
#             self.settings.router_output_type = DECISION_OUTPUT_TYPE[
#                 SPATIO_TEMPORAL_TRAJECTORY]
#             self.settings.router_file_type = FILE_TYPE[C_DLL_FILE]
#         elif self.comboDecisionInternal.currentIndex() == MAP2_CHENXIANG:
#             self.settings.router_lib = DEFAULT_DECISION_FILE_PATH[MAP2_CHENXIANG-2]
#             self.settings.router_output_type = DECISION_OUTPUT_TYPE[
#                 SPATIO_TEMPORAL_TRAJECTORY]
#             self.settings.router_file_type = FILE_TYPE[C_DLL_FILE]
#         else:
#             self.settings.router_lib = '-'
#             self.settings.router_file_type = FILE_TYPE[
#                 self.comboDecisionInternal.currentIndex()]
#         self.__refresh()
#
#     def on_planner_output_change(self):
#         self.settings.router_output_type = DECISION_OUTPUT_TYPE[
#             self.comboDecisionType.currentIndex()]
#         self.__refresh()
#
#     def on_browse_controller(self):
#         file_path=QFileDialog.getOpenFileName(self, "load controller model",
#                                               ".", "Dynamic Libary(*.dll)")
#         if len(file_path) == 0:
#             return
#         self.editControllerPath.setText(file_path)
#
#     def on_browse_planner(self):
#         file_path=QFileDialog.getOpenFileName(self, "load planner", ".",
#                                               "Dynamic Libary(*.dll)")
#         if len(file_path) == 0:
#             return
#         self.editDecisionPath.setText(file_path)
#
#     def on_cancel(self):
#         self.done(0)
#
#     def update_settings(self):
#         # 配置报错
#         # TODO:(Chason) 后期加入检查仿真设置是否错误的功能
#         pass
#
#         # 更新仿真步长
#         if int(self.editStepLength.text()) < 0:
#             self.settings.step_length = 100
#         else:
#             self.settings.step_length = int(self.editStepLength.text())
#         self.editStepLength.setText(str(self.settings.step_length))
#
#         # 更新全局驾驶路径
#         self.settings.mission_type = self.mission_type
#         self.settings.points = self.points
#
#         # 更新传感器配置参数
#         self.settings.sensors = self.sensors
#         self.settings.sensor_model_lib = str(self.editSensorsPath.text())
#         if int(self.editSensorRatio.text()) < 0:
#             self.settings.sensor_frequency = 1
#         else:
#             self.settings.sensor_frequency = int(self.editSensorRatio.text())
#
#         # 更新交通流配置参数
#         if int(self.editTrafficRatio.text()) < 0:
#             self.settings.traffic_frequency = 1
#         else:
#             self.settings.traffic_frequency = int(self.editTrafficRatio.text())
#
#         # 更新动力学配置参数
#         if int(self.editDynamicRatio.text()) < 0:
#             self.settings.dynamic_frequency = 1
#         else:
#             self.settings.dynamic_frequency = int(self.editDynamicRatio.text())
#         self.settings.car_length = float(self.editLength.text())
#         self.settings.car_width = float(self.editWidth.text())
#
#         # 更新控制器配置参数
#         self.settings.controller_lib = str(self.editControllerPath.text())
#         if int(self.editControllRatio.text()) < 0:
#             self.settings.controller_frequency = 1
#         else:
#             self.settings.controller_frequency = int(
#                 self.editControllRatio.text())
#
#         # 更新决策模块配置参数
#         self.settings.router_lib = str(self.editDecisionPath.text())
#         if int(self.editDecisionRatio.text()) < 0:
#             self.settings.router_frequency = 1
#         else:
#             self.settings.router_frequency = int(self.editDecisionRatio.text())
#
#     def on_ok(self):
#         self.update_settings()
#         self.done(1)
#
#     def on_map(self):
#         dialog = MapDialog(self.settings.map)
#         ret = dialog.exec_()
#         if ret == 1:
#             self.points = dialog.mission
#             self.mission_type = dialog.mission_type
#
#     def on_map_change(self):
#         self.settings.map = MAPS[self.comboMap.currentIndex()]
#         self.__refresh()
#
#     def on_dynamic(self):
#
#         dialog = VehicleModelDialog(car_info=self.settings.car_para,
#                                     car_type=self.settings.dynamic_type)
#         ret = dialog.exec_()
#         if ret == 1:
#             self.settings.car_para = dialog.car_info
#
#     def on_traffic_parameter(self):
#         # TODO(Chason): 后期统一rou.xml文件和sumocfg.xml文件中对交通流类型的命名
#         type = ['No Traffic',
#                 'Mixed',
#                 'VehicleOnly'][self.comboTraffic.currentIndex()]
#         density = ['40', '200', '500'][self.comboTrafficDensity.currentIndex()]
#         map = MAPS[self.comboMap.currentIndex()]
#         file_path = 'notepad Map/'+map+'/'+'grid-map-traffic_'+type+'_'+density+'.rou.xml'
#         os.system(file_path)
#
#     def set_sensors(self, sensors):
#         sensor_lib = SensorLibrary()
#         types = sensor_lib.types()
#         self.tableWidget.clearContents()
#         for i in range(len(sensors)):
#             self.tableWidget.insertRow(i)
#             self.tableWidget.setItem(i, 0, QTableWidgetItem(
#                 types[int(sensors[i].type)]))
#             self.tableWidget.setItem(i, 1, QTableWidgetItem(
#                 str(sensors[i].detection_range)))
#             self.tableWidget.setItem(i, 2, QTableWidgetItem(
#                 str(sensors[i].detection_angle)))
#             self.tableWidget.setItem(i, 3, QTableWidgetItem(
#                 str(sensors[i].installation_orientation_angle)))
#         self.sensors = copy.copy(sensors)
#
#     def on_sensor_edit(self):
#         dialog = SensorDialog()
#         dialog.init_view(self.settings.sensors)
#         ret = dialog.exec_()
#         if ret != 1:
#             return
#
#         n = len(dialog.sensors)
#         sensors = (SensorInfo * n)()
#         for i in range(n):
#             sensors[i] = dialog.sensors[i]
#             sensors.ID = i
#         self.set_sensors(sensors)
#
#
# class LearnerDialog(QtGui.QDialog):
#     """Learner环境配置界面对象
#
#     Args:
#     """
#
#     def __init__(self, parent=None, simulation=None):
#         # 读取QT文件
#         QtGui.QDialog.__init__(self, parent)
#         uic.loadUi('QT UI/LearnerSettings.ui', self)
#         font = QFont()
#         font.setPixelSize(12)
#         self.setFont(font)
#
#         # 设置窗口的logo
#         icon = QIcon()
#         icon.addPixmap(QPixmap("Resources/Logos/setting.ico"), QIcon.Normal)
#         self.setWindowIcon(icon)
#
#         # 设置窗口响应事件
#         self.buttonCancel.clicked.connect(self.__on_cancel)
#         self.buttonSave.clicked.connect(self.__on_save)
#         # 地图相关响应事件
#         self.buttonMap.clicked.connect(self.__on_map)
#         self.comboMap.currentIndexChanged.connect(self.__on_map_change)
#         # 交通流相关响应事件
#         self.buttonTraffic.clicked.connect(self.__on_traffic_parameter)
#         self.comboTraffic.currentIndexChanged.connect(
#             self.__on_traffic_type_change)
#         self.comboTrafficDensity.currentIndexChanged.connect(
#             self.__on_traffic_density_change)
#         # 动力学相关响应事件
#         self.buttonModel.clicked.connect(self.__on_dynamic)
#         self.comboModel.currentIndexChanged.connect(self.__on_dynamic_change)
#         # 传感器相关响应事件
#         self.buttonSensorEdit.clicked.connect(self.__on_sensor_edit)
#         # Action相关响应事件
#         self.comboActionType.currentIndexChanged.connect(self.__on_action_change)
#
#         self.settings = None  # 仿真设置对象
#         self.editDynamicPath = ''
#         self.points = []  # 全局路径点
#         self.mission_type = None  # 任务类型：单向/循环
#         self.learner_traffic = None
#         self.history_traffic = ['Traffic Type', 'Traffic Density', 'Map']
#
#     def init_view(self, settings):
#         """初始化UI界面。
#
#         Args：
#         """
#         self.settings = settings
#         self.points = settings.points
#         self.mission_type = settings.mission_type
#
#         # learner模式默认控制器采用内部控制器，取消决策模块
#         self.settings.controller_type = CONTROLLER_TYPE[PID]
#         self.settings.controller_lib = CONTROLLER_FILE_PATH
#         self.settings.controller_file_type = FILE_TYPE[C_DLL_FILE]
#         self.settings.controller_frequency = 1
#
#         self.settings.router_type = '-'
#         self.settings.router_lib = '-'
#         self.settings.router_frequency = '-'
#         self.settings.router_file_type = '-'
#
#         self.__refresh()  # 初始窗口时需要刷新窗口两次
#         self.__refresh()  # 因为前三次刷新窗口每次窗口都会变化
#         self.__refresh()
#
#     def __refresh(self):
#         """刷新窗口界面。
#
#         Args：
#         """
#         # 设置传感器配置信息
#         self.__set_sensors(self.settings.sensors)
#         self.editSensorsPath.setText(self.settings.sensor_model_lib)
#         self.editSensorRatio.setText(str(self.settings.sensor_frequency))
#
#         # 显示地图缩略图
#         if self.settings.map == MAPS[0]:
#             img = QImage('Resources/Tools/urban_road_map.png')
#         elif self.settings.map == MAPS[1]:
#             img = QImage('Resources/Tools/highway_map.png')
#         else:
#             raise Exception(self.settings.map + '暂未开放')
#         self.label_map.setPixmap(QPixmap.fromImage(img))
#
#         # 设置交通流配置信息
#         if self.settings.traffic_type == TRAFFIC_TYPE[NO_TRAFFIC]:
#             self.comboTraffic.setCurrentIndex(0)
#             self.comboTrafficDensity.setEnabled(0)
#             self.buttonTraffic.setEnabled(0)
#         elif self.settings.traffic_type == TRAFFIC_TYPE[MIXED_TRAFFIC]:
#             self.comboTraffic.setCurrentIndex(1)
#             self.comboTrafficDensity.setEnabled(1)
#             self.buttonTraffic.setEnabled(1)
#         else:
#             self.comboTraffic.setCurrentIndex(2)
#             self.comboTrafficDensity.setEnabled(1)
#             self.buttonTraffic.setEnabled(1)
#         if self.settings.traffic_lib == TRAFFIC_DENSITY[MIDDLE]:
#             self.comboTrafficDensity.setCurrentIndex(1)
#         elif self.settings.traffic_lib == TRAFFIC_DENSITY[DENSE]:
#             self.comboTrafficDensity.setCurrentIndex(2)
#         elif self.settings.traffic_lib == TRAFFIC_DENSITY[SPARSE]:
#             self.comboTrafficDensity.setCurrentIndex(0)
#         if self.settings.map == MAPS[1] and self.comboTraffic.currentIndex() == 1:
#             self.settings.traffic_type = TRAFFIC_TYPE[VEHICLE_ONLY_TRAFFIC]
#             self.comboTraffic.setCurrentIndex(VEHICLE_ONLY_TRAFFIC)
#         self.editTrafficRatio.setText(str(self.settings.traffic_frequency))
#
#         # 设置动力学配置信息
#         if self.settings.dynamic_type == CAR_TYPE[AMT_CAR]:
#             self.comboModel.setCurrentIndex(AMT_CAR)
#         elif self.settings.dynamic_type == CAR_TYPE[CVT_CAR]:
#             self.comboModel.setCurrentIndex(CVT_CAR)
#         elif self.settings.dynamic_type == CAR_TYPE[TRUCK]:
#             self.comboModel.setCurrentIndex(TRUCK)
#         elif self.settings.dynamic_type == CAR_TYPE[EV]:
#             self.comboModel.setCurrentIndex(EV)
#         self.editWeight.setText(str(self.settings.car_para.M_SU +
#                                     self.settings.car_para.M_US))
#         self.editLength.setText(str(self.settings.car_length))
#         self.editWidth.setText(str(self.settings.car_width))
#         self.editDynamicRatio.setText(str(self.settings.dynamic_frequency))
#
#         # 设置action配置信息
#         if self.settings.router_output_type == DECISION_OUTPUT_TYPE[
#             KINEMATIC_OUTPUT]:
#             self.comboActionType.setCurrentIndex(KINEMATIC_OUTPUT-1)
#         else:
#             self.comboActionType.setCurrentIndex(DYNAMIC_OUTPUT-1)
#
#         self.editStepLength.setText(str(self.settings.step_length))
#
#     def __on_dynamic_change(self):
#         if self.comboModel.currentIndex() == CVT_CAR:
#             self.settings.dynamic_lib = CVT_MODEL_FILE_PATH
#             self.settings.car_para.CFx = 0.3
#             self.settings.car_para.R_GEAR_FD = 2.65
#             self.settings.car_para.LX_AXLE = 3.16
#             self.settings.car_para.LX_CG_SU = 1.265
#             self.settings.car_para.M_SU = 1820
#             self.settings.car_para.IZZ_SU = 4095.0
#             self.settings.car_para.A = 3.0
#             self.settings.car_para.IENG = 0.4
#             self.settings.car_para.Steer_FACTOR = 16.5
#             self.settings.car_para.M_US = 200
#             self.settings.car_para.RRE = 0.353
#             self.settings.car_para.CF = -128915.5
#             self.settings.car_para.CR = -117481.8
#             self.settings.car_para.AV_ENGINE_IDLE = 750
#             self.settings.car_para.R_GEAR_TR1 = 4.6
#             self.settings.car_para.ROLL_RESISTANCE = 0.0041
#             self.settings.car_para.BRAK_COEF = 1100.0
#             self.settings.car_para.TAU = 0.3
#         elif self.comboModel.currentIndex() == AMT_CAR:
#             self.settings.dynamic_lib = AMT_MODEL_FILE_PATH
#             self.settings.car_para.CFx = 0.30
#             self.settings.car_para.R_GEAR_FD = 4.10
#             self.settings.car_para.LX_AXLE = 2.91
#             self.settings.car_para.LX_CG_SU = 1.015
#             self.settings.car_para.M_SU = 1270
#             self.settings.car_para.IZZ_SU = 1536.7
#             self.settings.car_para.A = 2.2
#             self.settings.car_para.IENG = 0.16
#             self.settings.car_para.Steer_FACTOR = 14.3
#             self.settings.car_para.M_US = 142
#             self.settings.car_para.RRE = 0.325
#             self.settings.car_para.CF = -128915.5
#             self.settings.car_para.CR = -85943.6
#             self.settings.car_para.AV_ENGINE_IDLE = 750
#             self.settings.car_para.R_GEAR_TR1 = 3.538
#             self.settings.car_para.ROLL_RESISTANCE = 0.0038
#             self.settings.car_para.BRAK_COEF = 800.0
#             self.settings.car_para.TAU = 0.2
#         elif self.comboModel.currentIndex() == TRUCK:
#             self.settings.dynamic_lib = TRUCK_MODEL_FILE_PATH
#             self.settings.car_para.CFx = 0.69
#             self.settings.car_para.R_GEAR_FD = 5.0
#             self.settings.car_para.LX_AXLE = 5.0
#             self.settings.car_para.LX_CG_SU = 1.11
#             self.settings.car_para.M_SU = 4455.0
#             self.settings.car_para.IZZ_SU = 34802.6
#             self.settings.car_para.A = 6.8
#             self.settings.car_para.IENG = 1.4
#             self.settings.car_para.Steer_FACTOR = 25.0
#             self.settings.car_para.M_US = 1305.0
#             self.settings.car_para.RRE = 0.51
#             self.settings.car_para.CF = -319703.0
#             self.settings.car_para.CR = -97687.0
#             self.settings.car_para.AV_ENGINE_IDLE = 750.0
#             self.settings.car_para.R_GEAR_TR1 = 7.59
#             self.settings.car_para.ROLL_RESISTANCE = 0.0041
#             self.settings.car_para.BRAK_COEF = 40000.0 / 7.0
#             self.settings.car_para.TAU = 0.5
#         elif self.comboModel.currentIndex() == EV:
#             self.settings.dynamic_lib = EV_MODEL_FILE_PATH
#             self.settings.car_para.CFx = 0.28
#             self.settings.car_para.R_GEAR_FD = 7.94
#             self.settings.car_para.LX_AXLE = 2.7
#             self.settings.car_para.LX_CG_SU = 1.0  # 待定
#             self.settings.car_para.M_SU = 800  # 但是总质量1521系网上查到。不影响计算
#             self.settings.car_para.IZZ_SU = 1400  # 待定
#             self.settings.car_para.A = 2.22  # 待定
#             self.settings.car_para.IENG = 0.21  # 电机不考虑该参数
#             self.settings.car_para.Steer_FACTOR = 15.0
#             self.settings.car_para.M_US = 136  # 待定，但是总质量1521系网上查到。不影响计算
#             self.settings.car_para.RRE = 0.316
#             self.settings.car_para.CF = -99323.8
#             self.settings.car_para.CR = -60348.6
#             self.settings.car_para.AV_ENGINE_IDLE = 750.0 # 待定
#             self.settings.car_para.R_GEAR_TR1 = 3.62  # 待定
#             self.settings.car_para.ROLL_RESISTANCE = 0.014  # 滚动阻力（网上查到）   良好的沥青或混凝土路面	0.010-0.018
#             self.settings.car_para.BRAK_COEF = 800.0  # 待定
#             self.settings.car_para.TAU = 0.1 #待定
#
#         self.settings.dynamic_type = CAR_TYPE[self.comboModel.currentIndex()]
#         self.__refresh()
#
#     def __on_dynamic(self):
#         dialog = VehicleModelDialog(car_info=self.settings.car_para)
#         ret = dialog.exec_()
#         if ret == 1:
#             self.settings.car_para = dialog.car_info
#
#     def __on_traffic_type_change(self):
#         if self.settings.map == MAPS[1]:
#             if self.comboTraffic.currentIndex() == MIXED_TRAFFIC:
#                 self.comboTraffic.setCurrentIndex(VEHICLE_ONLY_TRAFFIC)
#         self.settings.traffic_type = TRAFFIC_TYPE[
#             self.comboTraffic.currentIndex()]
#         self.__refresh()
#
#     def __on_traffic_density_change(self):
#         self.settings.traffic_lib = TRAFFIC_DENSITY[
#             self.comboTrafficDensity.currentIndex()]
#         self.__refresh()
#
#     def __on_cancel(self):
#         self.__update_settings()
#         self.done(0)
#
#     def __on_save(self):
#         self.__update_settings()
#         simulation_path = QFileDialog.getSaveFileName(
#             self, "new simulation", ".", "LasVSim Simulation")
#         if len(simulation_path) == 0:
#             return
#         simulation_path = simulation_path + '_LasVSim'
#         os.mkdir(simulation_path)
#
#         if (self.history_traffic[0] != self.settings.traffic_type or
#             self.history_traffic[1] != self.settings.traffic_lib or
#             self.history_traffic[2] != self.settings.map):
#             traffic = Traffic(step_length=1, map_type=self.settings.map,
#                               traffic_type=self.settings.traffic_type,
#                               traffic_density=self.settings.traffic_lib,
#                               isLearner=True,
#                               settings=self.settings)
#             self.learner_traffic = traffic.random_traffic
#         traffic_data = data_module.TrafficData()
#         traffic_data.save_traffic(self.learner_traffic,simulation_path)
#         self.settings.save(simulation_path + '/simulation setting file.xml')
#
#
#         self.done(1)
#         # self.settings.save(file_path)
#
#     def __update_settings(self):
#         # 配置报错
#         # TODO(Chason): 后期加入检查仿真设置是否错误的功能
#         pass
#
#         # 更新仿真步长
#         if int(self.editStepLength.text()) < 0:
#             self.settings.step_length = 100
#         else:
#             self.settings.step_length = int(self.editStepLength.text())
#         self.editStepLength.setText(str(self.settings.step_length))
#
#         # 更新全局驾驶路径
#         self.settings.mission_type = self.mission_type
#         self.settings.points = self.points
#
#         # 更新传感器配置参数
#         self.settings.sensors = self.sensors
#         self.settings.sensor_model_lib = str(self.editSensorsPath.text())
#         if int(self.editSensorRatio.text()) < 0:
#             self.settings.sensor_frequency = 1
#         else:
#             self.settings.sensor_frequency = int(self.editSensorRatio.text())
#         self.editSensorRatio.setText(str(self.settings.sensor_frequency))
#
#         # 更新交通流配置参数
#         if int(self.editTrafficRatio.text()) < 0:
#             self.settings.traffic_frequency = 1
#         else:
#             self.settings.traffic_frequency = int(self.editTrafficRatio.text())
#
#         # 更新动力学配置参数
#         if int(self.editDynamicRatio.text()) < 0:
#             self.settings.dynamic_frequency = 1
#         else:
#             self.settings.dynamic_frequency = int(self.editDynamicRatio.text())
#
#         # 更新action类型
#         self.settings.router_output_type = DECISION_OUTPUT_TYPE[
#             self.comboActionType.currentIndex()+1]
#
#     def __on_map(self):
#         dialog = MapDialog(map=self.settings.map)
#         ret = dialog.exec_()
#         if ret == 1:
#             self.points = dialog.mission
#             self.mission_type = dialog.mission_type
#
#     def __on_map_change(self):
#         self.settings.map = MAPS[self.comboMap.currentIndex()]
#         self.__refresh()
#
#     def __set_sensors(self, sensors):
#         types = SensorLibrary().types()
#         self.tableWidget.clearContents()
#         for i in range(len(sensors)):
#             self.tableWidget.insertRow(i)
#             self.tableWidget.setItem(i, 0, QTableWidgetItem(
#                 types[int(sensors[i].type)]))
#             self.tableWidget.setItem(i, 1, QTableWidgetItem(
#                 str(sensors[i].detection_range)))
#             self.tableWidget.setItem(i, 2, QTableWidgetItem(
#                 str(sensors[i].detection_angle)))
#             self.tableWidget.setItem(i, 3, QTableWidgetItem(
#                 str(sensors[i].installation_orientation_angle)))
#         self.sensors = copy.copy(sensors)
#
#     def __on_traffic_parameter(self):
#         # TODO(Chason): 后期统一rou.xml文件和sumocfg.xml文件中对交通流类型的命名
#         type = ['No Traffic',
#                 'Mixed',
#                 'VehicleOnly'][self.comboTraffic.currentIndex()]
#         density = ['40', '200', '500'][self.comboTrafficDensity.currentIndex()]
#         map = MAPS[self.comboMap.currentIndex()]
#         file_path = 'notepad Map/' + map + '/' + 'grid-map-traffic_' + type + '_' + density + '.rou.xml'
#         os.system(file_path)
#
#     def __on_sensor_edit(self):
#         dialog = SensorDialog()
#         dialog.init_view(self.settings.sensors)
#         ret = dialog.exec_()
#         if ret != 1:
#             return
#
#         n = len(dialog.sensors)
#         sensors = (SensorInfo * n)()
#         for i in range(n):
#             sensors[i] = dialog.sensors[i]
#             sensors.ID = i
#         self.__set_sensors(sensors)
#
#     def __on_action_change(self):
#         self.settings.router_output_type = DECISION_OUTPUT_TYPE[
#             self.comboActionType.currentIndex()+1]
#         self.__refresh()
#
#
# class VehicleModelDialog(QtGui.QDialog):
#     def __init__(self, car_type,car_info=None,parent=None ):
#         QtGui.QDialog.__init__(self, parent)
#         uic.loadUi('QT UI/VehicleParameterSetting/VehicleParameterSetting.ui',
#                    self)
#         font = QFont()
#         font.setPixelSize(12)
#         self.setFont(font)
#
#         self.buttonOK.clicked.connect(self.on_ok)
#         self.buttonCancel.clicked.connect(self.on_cancel)
#         self.buttonEditMass.clicked.connect(self.on_mass)
#         self.buttonEditAerodynamics.clicked.connect(self.on_aerodynamic)
#         self.buttonEditPowertrain.clicked.connect(self.on_power_train)
#         self.buttonEditBrake.clicked.connect(self.on_brake)
#         self.buttonEditSteering.clicked.connect(self.on_steering)
#         self.buttonEditTire.clicked.connect(self.on_tire)
#         self.car_info = car_info
#         self.car_type = car_type
#
#     def on_ok(self):
#         self.done(1)
#
#     def on_cancel(self):
#         self.done(0)
#
#     def on_aerodynamic(self):
#         dialog = AerodynamicsDialog(CFx=self.car_info.CFx,
#                                     A=self.car_info.A)
#         ret = dialog.exec_()
#         if ret == 1:
#             self.car_info.CFx = dialog.CFx
#             self.car_info.A = dialog.A
#
#     def on_mass(self):
#         dialog = MassDialog(car_info=self.car_info)
#         ret = dialog.exec_()
#         if ret == 1:
#             self.car_info.IZZ_SU = dialog.IZZ_SU
#             self.car_info.LX_AXLE = dialog.LX_AXLE
#             self.car_info.M_SU = dialog.M_SU
#             self.car_info.M_US = dialog.M_US
#             self.car_info.LX_CG_SU = dialog.LX_CG_SU
#
#     def on_tire(self):
#         dialog = TireDialog(car_info=self.car_info)
#         ret = dialog.exec_()
#         if ret == 1:
#             self.car_info.CF = dialog.CF
#             self.car_info.CR = dialog.CR
#             self.car_info.RRE = dialog.RRE
#             self.car_info.ROLL_RESISTANCE = dialog.ROLL_RESISTANCE
#
#     def on_brake(self):
#         dialog = BrakeSystemDialog(car_info=self.car_info)
#         ret = dialog.exec_()
#         if ret == 1:
#             self.car_info.BRAK_COEF = dialog.BRAK_COEF
#
#     def on_steering(self):
#         dialog = SteeringSystemDialog(car_info=self.car_info)
#         ret = dialog.exec_()
#         if ret == 1:
#             self.car_info.Steer_FACTOR = dialog.Steer_FACTOR
#
#     def on_power_train(self):
#         dialog = PowerTrainDialog(car_info=self.car_info,car_type=self.car_type)
#         ret = dialog.exec_()
#         if ret == 1:
#             self.car_info.AV_ENGINE_IDLE = dialog.AV_ENGINE_IDLE
#             self.car_info.IENG = dialog.IENG
#             self.car_info.TAU = dialog.TAU
#             self.car_info.R_GEAR_TR1 = dialog.R_GEAR_TR1
#             self.car_info.R_GEAR_FD = dialog.R_GEAR_FD
#
#
# class AerodynamicsDialog(QtGui.QDialog):
#     def __init__(self, parent=None, CFx=None, A=None):
#         QtGui.QDialog.__init__(self, parent)
#         uic.loadUi('QT UI/VehicleParameterSetting/Aerodynamics.ui',
#                    self)
#         font = QFont()
#         font.setPixelSize(12)
#         self.setFont(font)
#
#         self.buttonOK.clicked.connect(self.__on_ok)
#         self.buttonCancel.clicked.connect(self.__on_cancel)
#
#         self.A = A
#         self.CFx = CFx
#
#         self.__init_view()
#
#     def __init_view(self):
#         self.CFxEdit.setText(str(self.CFx))
#         self.AEdit.setText(str(self.A))
#
#     def __update_data(self):
#         self.A = float(self.AEdit.text())
#         self.CFx = float(self.CFxEdit.text())
#
#     def __on_ok(self):
#         self.__update_data()
#         self.done(1)
#
#     def __on_cancel(self):
#         self.done(0)
#
#
# class MassDialog(QtGui.QDialog):
#     def __init__(self, parent=None, car_info=None):
#         QtGui.QDialog.__init__(self, parent)
#         uic.loadUi('QT UI/VehicleParameterSetting/Mass.ui',
#                    self)
#         font = QFont()
#         font.setPixelSize(12)
#         self.setFont(font)
#
#         self.buttonOK.clicked.connect(self.__on_ok)
#         self.buttonCancel.clicked.connect(self.__on_cancel)
#
#         self.M_SU = car_info.M_SU
#         self.M_US = car_info.M_US
#         self.IZZ_SU = car_info.IZZ_SU
#         self.LX_AXLE = car_info.LX_AXLE
#         self.LX_CG_SU = car_info.LX_CG_SU
#
#         self.__init_view()
#
#     def __init_view(self):
#         self.EditM_SU.setText(str(self.M_SU))
#         self.EditM_US.setText(str(self.M_US))
#         self.EditIZZ_SU.setText(str(self.IZZ_SU))
#         self.EditLX_AXLE.setText(str(self.LX_AXLE))
#         self.EditLX_CG_SU.setText(str(self.LX_CG_SU))
#
#     def __update_data(self):
#         self.M_SU = float(self.EditM_SU.text())
#         self.M_US = float(self.EditM_US.text())
#         self.IZZ_SU = float(self.EditIZZ_SU.text())
#         self.LX_AXLE = float(self.EditLX_AXLE.text())
#         self.LX_CG_SU = float(self.EditLX_CG_SU.text())
#
#     def __on_ok(self):
#         self.__update_data()
#         self.done(1)
#
#     def __on_cancel(self):
#         self.done(0)
#
#
# class TireDialog(QtGui.QDialog):
#     def __init__(self, parent=None, car_info=None):
#         QtGui.QDialog.__init__(self, parent)
#         uic.loadUi('QT UI/VehicleParameterSetting/Tire.ui',
#                    self)
#         font = QFont()
#         font.setPixelSize(12)
#         self.setFont(font)
#
#         self.buttonOK.clicked.connect(self.__on_ok)
#         self.buttonCancel.clicked.connect(self.__on_cancel)
#
#         self.CF = car_info.CF
#         self.CR = car_info.CR
#         self.RRE = car_info.RRE
#         self.ROLL_RESISTANCE = car_info.ROLL_RESISTANCE
#
#         self.__init_view()
#
#     def __init_view(self):
#         self.EditCF.setText(str(self.CF))
#         self.EditCR.setText(str(self.CR))
#         self.EditRRE.setText(str(self.RRE))
#         self.EditROLL_RESISTANCE.setText(str(self.ROLL_RESISTANCE))
#
#     def __update_data(self):
#         self.CF = float(self.EditCF.text())
#         self.CR = float(self.EditCR.text())
#         self.RRE = float(self.EditRRE.text())
#         self.ROLL_RESISTANCE = float(self.EditROLL_RESISTANCE.text())
#
#     def __on_ok(self):
#         self.__update_data()
#         self.done(1)
#
#     def __on_cancel(self):
#         self.done(0)
#
#
# class BrakeSystemDialog(QtGui.QDialog):
#     def __init__(self, parent=None, car_info=None):
#         QtGui.QDialog.__init__(self, parent)
#         uic.loadUi('QT UI/VehicleParameterSetting/BrakeSystem.ui',
#                    self)
#         font = QFont()
#         font.setPixelSize(12)
#         self.setFont(font)
#
#         self.buttonOK.clicked.connect(self.__on_ok)
#         self.buttonCancel.clicked.connect(self.__on_cancel)
#
#         self.BRAK_COEF = car_info.BRAK_COEF
#
#         self.__init_view()
#
#     def __init_view(self):
#         self.EditBRAK_COEF.setText(str(self.BRAK_COEF))
#
#     def __update_data(self):
#         self.BRAK_COEF = float(self.EditBRAK_COEF.text())
#
#     def __on_ok(self):
#         self.__update_data()
#         self.done(1)
#
#     def __on_cancel(self):
#         self.done(0)
#
#
# class SteeringSystemDialog(QtGui.QDialog):
#     def __init__(self, parent=None, car_info=None):
#         QtGui.QDialog.__init__(self, parent)
#         uic.loadUi('QT UI/VehicleParameterSetting/SteeringSystem.ui',
#                    self)
#         font = QFont()
#         font.setPixelSize(12)
#         self.setFont(font)
#
#         self.buttonOK.clicked.connect(self.__on_ok)
#         self.buttonCancel.clicked.connect(self.__on_cancel)
#
#         self.Steer_FACTOR = car_info.Steer_FACTOR
#
#         self.__init_view()
#
#     def __init_view(self):
#         self.EditSteer_FACTOR.setText(str(self.Steer_FACTOR))
#
#     def __update_data(self):
#         self.Steer_FACTOR = float(self.EditSteer_FACTOR.text())
#
#     def __on_ok(self):
#         self.__update_data()
#         self.done(1)
#
#     def __on_cancel(self):
#         self.done(0)
#
#
# class PowerTrainDialog(QtGui.QDialog):
#     def __init__(self, parent=None, car_info=None,car_type = None):
#         QtGui.QDialog.__init__(self, parent)
#         uic.loadUi('QT UI/VehicleParameterSetting/PowerTrain.ui',
#                    self)
#
#
#         font = QFont()
#         font.setPixelSize(12)
#         self.setFont(font)
#
#         self.buttonOK.clicked.connect(self.__on_ok)
#         self.buttonCancel.clicked.connect(self.__on_cancel)
#
#         self.AV_ENGINE_IDLE = car_info.AV_ENGINE_IDLE
#         self.IENG = car_info.IENG
#         self.TAU = car_info.TAU
#         self.R_GEAR_TR1 = car_info.R_GEAR_TR1
#         self.R_GEAR_FD = car_info.R_GEAR_FD
#
#         self.__init_view()
#
#     def __init_view(self):
#         self.EditAV_ENGINE_IDLE.setText(str(self.AV_ENGINE_IDLE))
#         self.EditIENG.setText(str(self.IENG))
#         self.EditTAU.setText(str(self.TAU))
#         self.EditR_GEAR_TR1.setText(str(self.R_GEAR_TR1))
#         self.EditR_GEAR_FD.setText(str(self.R_GEAR_FD))
#
#     def __update_data(self):
#         self.AV_ENGINE_IDLE = float(self.EditAV_ENGINE_IDLE.text())
#         self.IENG = float(self.EditIENG.text())
#         self.TAU = float(self.EditTAU.text())
#         self.R_GEAR_TR1 = float(self.EditR_GEAR_TR1.text())
#         self.R_GEAR_FD = float(self.EditR_GEAR_FD.text())
#
#     def __on_ok(self):
#         self.__update_data()
#         self.done(1)
#
#     def __on_cancel(self):
#         self.done(0)
#
# if __name__ == "__main__":
#     """QT UI 单元测试"""
#     settings = Settings()
#     settings.load('Library/default_simulation_setting.xml')
#     app = QtGui.QApplication(sys.argv)
#
#     index = 0
#     # 0 测试simulation settings 窗口
#     # 1 测试learner settings 窗口
#     flag = 3  # 0 输出动力学模块 1 输出决策模块 2 输出控制器模块 3 输出交通流模块
#     if index == 0:
#         # Simulation Setting UI窗口测试
#         while True:
#             win = SettingsDialog()
#             win.init_view(settings)
#             win.show()   # show window
#             win.exec_()
#             settings = copy.deepcopy(win.settings)
#             if flag == 0:
#                 print("动力学模型："),
#                 print(win.settings.dynamic_type)
#                 print("动力学路径："),
#                 print(win.settings.dynamic_lib)
#                 print("动力学频率："),
#                 print(win.settings.dynamic_frequency)
#             elif flag == 1:
#                 print("决策输出类型："),
#                 print(win.settings.router_output_type)
#                 print("决策算法名字："),
#                 print(win.settings.router_type)
#                 print("决策文件类型："),
#                 print(win.settings.router_file_type)
#                 print("决策频率："),
#                 print(win.settings.router_frequency)
#                 print("决策路径："),
#                 print(win.settings.router_lib)
#             elif flag == 2:
#                 print("控制器类型："),
#                 print(win.settings.controller_type)
#                 print("控制器文件类型："),
#                 print(win.settings.controller_file_type)
#                 print("控制器路径："),
#                 print(win.settings.controller_lib)
#                 print("控制器频率："),
#                 print(win.settings.controller_frequency)
#             elif flag == 3:
#                 print("交通流类型："),
#                 print(win.settings.traffic_type)
#                 print("交通流路径："),
#                 print(win.settings.traffic_lib)
#                 print("交通流频率："),
#                 print(win.settings.traffic_frequency)
#
#             print(' ')
#     elif index == 1:
#         # Learner Settings UI 窗口测试
#         while True:
#             win = LearnerDialog()
#             win.init_view(settings)
#             win.show()   # show window
#             win.exec_()
#             settings = copy.deepcopy(win.settings)
#             if flag == 0:
#                 print("动力学模型："),
#                 print(win.settings.dynamic_type)
#                 print("动力学路径："),
#                 print(win.settings.dynamic_lib)
#                 print("动力学频率："),
#                 print(win.settings.dynamic_frequency)
#             elif flag == 1:
#                 print("决策输出类型："),
#                 print(win.settings.router_output_type)
#                 print("决策算法名字："),
#                 print(win.settings.router_type)
#                 print("决策文件类型："),
#                 print(win.settings.router_file_type)
#                 print("决策频率："),
#                 print(win.settings.router_frequency)
#                 print("决策路径："),
#                 print(win.settings.router_lib)
#             elif flag == 2:
#                 print("控制器类型："),
#                 print(win.settings.controller_type)
#                 print("控制器文件类型："),
#                 print(win.settings.controller_file_type)
#                 print("控制器路径："),
#                 print(win.settings.controller_lib)
#                 print("控制器频率："),
#                 print(win.settings.controller_frequency)
#             elif flag == 3:
#                 print("交通流类型："),
#                 print(win.settings.traffic_type)
#                 print("交通流路径："),
#                 print(win.settings.traffic_lib)
#                 print("交通流频率："),
#                 print(win.settings.traffic_frequency)
#             print(' ')
#
#     sys.exit(app.exec_())
