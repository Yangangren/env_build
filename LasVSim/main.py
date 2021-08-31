# coding=utf-8
"""
@author: Chason Xu
@file: main.py
@time: 2019.12.26 14:00
@file_desc: Main entrance of LasVSim-gui for LasVSim-gui 0.2.1.191211_alpha
"""
import sys
import logging
import importlib
importlib.reload(sys)
#sys.setdefaultencoding('utf-8')
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from multiprocessing import freeze_support#含multiprocessing的程序打包成windows可执行文件后需要在 if __name__ == '__main__'后调用
import os
import subprocess
import numpy as np
from PyQt4.Qt import *
from PyQt4 import QtCore, QtGui,uic
from sip import voidptr
from simulator import *
from rendering_module import MapView
from simulation_setting import *
from math_lib import degree_fix

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
global simulation   # simulation object
map_view = MapView()  # 绘制左下角的小地图
qtCreatorFile = "QT UI/MainWindow.ui"  # main window ui resource
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
main_window = None    # 主窗口对象


class Plotter:
    """
    车辆状态量数据绘制类.

    Attributes:
        plot_points(int): 数据图默认范围内显示的最多数据个数
        curve_types(list):
        curve_type_info(dict):
        fig_arr(list): 储存所有fig图的数组
        replay_mode(bool): 回放模式标志位
        data(Data instance): 仿真数据储存类
        replay_data(ReplayData instance): 回放数据储存类
    """
    def __init__(self):
        self.plot_points = 0  # 数据图默认范围内显示的最多数据个数
        self.curve_types = ['Position X', 'Position Y', 'Vehicle Speed',
                            'Heading Angle', 'Acceleration',
                            'Longitudinal Velocity', 'Lateral Velocity',
                            'Steering Wheel', 'Throttle', 'Brake Pressure',
                            'Gear', 'Engine Speed', 'Engine Torque',
                            'Side Slip', 'Yaw Rate']
        self.curve_type_info = {'Position X':
                                {'unit': 'm', 'min': -1000.0, 'max': 1000.0},
                                'Position Y':
                                {'unit': 'm', 'min': -1000.0, 'max': 1000.0},
                                'Vehicle Speed':
                                {'unit': 'km/h', 'min': -0.0, 'max': 120.0},
                                'Heading Angle':
                                {'unit': 'deg', 'min': -180.0, 'max': 180.0},
                                'Acceleration':
                                {'unit': 'm/s^2', 'min': -4.0, 'max': 4.0},
                                'Longitudinal Velocity':
                                {'unit': 'm/s', 'min': 0.0, 'max': 35.0},
                                'Lateral Velocity':
                                {'unit': 'm/s', 'min': -10.0, 'max': 10.0},
                                'Steering Wheel':
                                {'unit': 'deg', 'min': -900.0, 'max': 900.0},
                                'Throttle':
                                {'unit': '%', 'min': 0.0, 'max': 100.0},
                                'Brake Pressure':
                                {'unit': 'MPa', 'min': 0.0, 'max': 8.0},
                                'Gear':
                                {'unit': '-', 'min': 0.0, 'max': 10.0},
                                'Engine Speed':
                                {'unit': 'rpm', 'min': 0.0, 'max': 3000.0},
                                'Engine Torque':
                                {'unit': 'N*m', 'min': 0.0, 'max': 50.0},
                                'Side Slip':
                                {'unit': 'deg', 'min': -20.0, 'max': 20.0},
                                'Yaw Rate':
                                {'unit': 'deg/s', 'min': -50.0, 'max': 50.0}}
        self.fig_arr = [None] * len(self.curve_types)  # 储存所有fig图的数组
        self.replay_mode = False
        self.data = None
        self.replay_data = None
        plt.ion()

    def set_replay_mode(self, replay_mode):
        """
        回放模式设置函数。

        Args:
            replay_mode(bool):

        Returns:

        """
        self.replay_mode = replay_mode

    def set_replay_data(self, replay_data):
        """
        导入回放数据

        Args:
            replay_data(ReplayData instance): 回放数据保存类

        Returns:

        """
        self.replay_data = replay_data

    def set_data(self, data):
        """
        导入数据

        Args:
            data:(Data instance): 数据保存类

        Returns:

        """
        self.data = data

    def get_types(self):
        """
        返回数据名列表

        Returns:
            ['Vehicle Speed', 'Acceleration', 'Position X'...]

        """
        return self.curve_types

    def get_data(self, tp):
        """
        返回数据

        Args:
            tp(int): 数据编号

        Returns:
            [0.0, 1.0, 2.0, 3.0, ...]

        """
        if self.replay_mode == 1:
            return self.replay_data.get_data(tp)
        else:
            return self.data.get_data(tp)

    def add_plot(self, fig_idx):
        """
        添加数据图

        Args:
            fig_idx(int): 图编号(等于数据编号)

        Returns:

        """
        fig = plt.figure(num=self.curve_types[fig_idx], figsize = (6,3))
        plt.subplots_adjust(bottom=0.2, left=0.15)
        ax = fig.add_subplot(111)
        ax.set_xlabel('t (s)')
        ax.set_ylabel('%s (%s)' % (self.curve_types[fig_idx],
                                   self.curve_type_info[
                                       self.curve_types[fig_idx]]['unit']))
        ax.plot(0)
        self.fig_arr[fig_idx] = fig
        self.refresh_plot(fig_idx)

        plt.get_current_fig_manager().window.setWindowFlags(
            Qt.WindowStaysOnTopHint)
        plt.get_current_fig_manager().window.show()

    def refresh_plot(self, fig_idx):
        """

        Args:
            fig_idx:

        Returns:

        """
        tp = self.curve_types[fig_idx]
        y_max = self.curve_type_info[tp]['max']
        y_min = self.curve_type_info[tp]['min']
        if not plt.fignum_exists(tp):  # 查找num=tp的图是否存在
            return
        fig = self.fig_arr[fig_idx]
        if fig is None:
            return
        x = self.get_data('Time')
        if len(x) == 0:
            return
        y = self.get_data(tp)
        # 设置横坐标范围(动态)
        if max(x) < 30.0:
            fig.get_axes()[0].set_xbound(*self.auto_ax_bound(0.0, 30.0))
        else:
            fig.get_axes()[0].set_xbound(
                *self.auto_ax_bound(max(x)-30.0, max(x)))
        # 设置纵坐标范围(固定)
        if not(y_max or y_min):
            fig.get_axes()[0].set_ybound(*self.auto_ax_bound(min(y), max(y)))
        else:
            fig.get_axes()[0].set_ybound(*self.auto_ax_bound(y_min, y_max))
        fig.get_axes()[0].get_lines()[0].set_data(x, y)

    def auto_ax_bound(self, a, b):
        """
        调整坐标轴范围

        Args:
            a:
            b:

        Returns:

        """
        spacing=0.0
        if b-a < 1e-3:
            spacing = 0.1
        else:
            spacing = (b-a)*0.1
        return (a-spacing, b+spacing)

    def refresh_all_plot(self):
        """
        刷新所有数据图

        Returns:

        """
        for i in range(len(self.curve_types)):
            self.refresh_plot(i)

    def close(self):
        """
        关闭所有数据图

        Returns:

        """
        for tp in self.curve_types:
            plt.close(tp)


class EvaluationPlotter:
    """
    评价模块plotter.

    Attributes:
        curve_names(list): 保存评价指标名称的列表
        plot_points(int): 数据图默认范围内显示的最多数据个数
        fig_arr(list): 数据图对象列表

    """
    def __init__(self):
        self.curve_names = ['Driving Safety', 'Fuel Economy',
                   'Riding Comfort',  'Travel Efficiency']
        self.plot_points = 0  # 数据图默认范围内显示的最多数据个数
        self.fig_arr = [None] * len(self.curve_names)
        plt.ion()

    def get_types(self):
        """
        返回评价指标名称列表

        Returns:

        """
        return self.curve_names

    def add_plot(self, fig_idx):
        """
        添加指定评价指标数据图

        Args:
            fig_idx(int): 评价指标编号

        Returns:

        """
        fig = plt.figure(num=simulation.evaluator.get_curve_name(fig_idx),figsize=(6,3))
        plt.subplots_adjust(bottom=0.2,left=0.15)
        ax=fig.add_subplot(111)
        ax.set_xlabel('t (s)')
        ax.set_ylabel('%s'%(simulation.evaluator.get_curve_name_and_unit(fig_idx)))
        ax.plot(0)
        self.fig_arr[fig_idx]=fig
        self.refresh_plot(fig_idx)

        plt.get_current_fig_manager().window.setWindowFlags(Qt.WindowStaysOnTopHint)
        plt.get_current_fig_manager().window.show()

    def refresh_plot(self, fig_idx):
        """
        刷新指定评价指标数据图

        Args:
            fig_idx(int): 评价指标编号

        Returns:

        """
        tp = simulation.evaluator.get_curve_name(fig_idx)
        if not plt.fignum_exists(tp):
            return
        fig = self.fig_arr[fig_idx]
        if fig is None:
            return
        # 获取数据
        y = simulation.evaluator.get_curve_data(fig_idx)
        if len(y) == 0:
            return
        x = [float(i) * (simulation.settings.step_length/1000.0) for i in range(len(y))]
        # 设置移动横坐标轴
        if max(x) < 30:
            fig.get_axes()[0].set_xbound(*self.auto_ax_bound(0, 30.0))
        else:
            fig.get_axes()[0].set_xbound(*self.auto_ax_bound(x[-1]-30.0, x[-1]))

        if fig_idx == 0:  # Safety
            string = 'Average: %.2f' % simulation.evaluator.get_report()[-4]
            fig.get_axes()[0].set_title(string)
            fig.get_axes()[0].set_ybound(*self.auto_ax_bound(0, 5))
        elif fig_idx == 1:  # Economy
            string = 'Average: %.2f L/100km' % simulation.evaluator.get_report()[-3]
            fig.get_axes()[0].set_title(string)
            fig.get_axes()[0].set_ybound(*self.auto_ax_bound(0, 5))
        elif fig_idx == 2:  # Comfort
            string = 'Average: %.2f m/$\mathregular{s^2}$' % simulation.evaluator.get_report()[-2]
            fig.get_axes()[0].set_title(string)
            fig.get_axes()[0].set_ybound(*self.auto_ax_bound(0, 5))
        elif fig_idx == 3:  # Efficiency
            string = 'Average: %.2f' % simulation.evaluator.get_report()[-1]
            fig.get_axes()[0].set_title(string)
            fig.get_axes()[0].set_ybound(*self.auto_ax_bound(0, 3))

        fig.get_axes()[0].get_lines()[0].set_data(x, y)

    def auto_ax_bound(self, min_value, max_value):
        """
        调整坐标轴范围

        Args:
            min_value(float): 最小值
            max_value(float): 最大值

        Returns:

        """
        spacing = 0.0
        if max_value-min_value < 1e-3:
            spacing = 0.1
        else:
            spacing = (max_value - min_value) * 0.1
        return min_value - spacing, max_value + spacing

    def refresh_all_plot(self):
        """
        刷新所有评价指标数据图

        Returns:

        """
        for i in range(len(simulation.evaluator.curve_names)):
            self.refresh_plot(i)

    def close(self):
        """
        关闭所有评价指标数据图

        Returns:

        """
        for tp in self.curve_names:
            plt.close(tp)


class MainUI(QtGui.QMainWindow, Ui_MainWindow):
    """
    LasVSim main window.

    Attributes:
        traffic_data(TrafficData instance): 初始交通流数据类
        current_simulation_path(string): 当前仿真项目文件夹路径
        current_configuration_file_path(string): 当前仿真配置文件路径
        SpeedX(int): 渲染速度(每n仿真步渲染一次)
        simulation_load_flag(bool): 仿真项目载入标志位
        StarsEnabled(bool): 评价指标分数显示标志位
        MapEnabled(bool): 小地图显示标志位
        ControlEnabled(bool): 控制量显示标志位
        SensorsEnabled(bool): 传感器探测范围显示标志位
        TrackEnabled(bool): 期望轨迹显示标志位
        VisualizeEnabled(bool): 渲染标志位
        InfoEnabled(bool): 自车状态信息显示标志位
        replay_mode(bool): 回放模式标志位
        image_steering(QImage instance): LasVSim主界面显示的控制量中的方向盘渲染图片
        curveSelect(QComboBox instance): QT组件，用于储存仿真数据显示列表
        plotter(Plotter instance):
        evaluation_plotter(EvaluationPlotter instance):
        eval_curveSelect(QComboBox instance): QT组件，用于储存评价数据显示列表
        slider(QSlider instance): QT滑块组件
        timer(QTimer):
    """
    def __init__(self):
        self.traffic_data = data_module.TrafficData()  # 初始交通流数据保存类
        self.current_simulation_path = None  # 当前仿真项目文件夹路径
        self.current_configuration_file_path = None  # 当前仿真配置文件路径
        self.SpeedX = 1
        # 标志位
        self.simulation_load_flag = False  # 仿真项目载入标志位
        self.StarsEnabled = True
        self.map_view_flag = True
        self.control_view_flag = True
        self.SensorsEnabled = True
        self.TrackEnabled = True
        self.VisualizeEnabled = True
        self.InfoEnabled = True
        self.replay_mode = False
        # 初始化QT窗口
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        font = QFont()
        font.setPixelSize(12)
        self.setFont(font)
        # 设置QT窗口图标
        icon = QIcon()
        icon.addPixmap(QPixmap("Resources/Logos/logo.ico"), QIcon.Normal)
        self.setWindowIcon(icon)
        # 定义QT窗口响应事件
        self.actionZoomIn.triggered.connect(self.zoom_in)
        self.actionZoomOut.triggered.connect(self.zoom_out)
        self.actionZoomReset.triggered.connect(self.zoom_reset)
        self.actionStart.triggered.connect(self.start)
        self.actionPause.triggered.connect(self.pause)
        self.actionStop.triggered.connect(self.stop)
        self.actionSpeed_1x.triggered.connect(self.speed_1x)
        self.actionSpeed_2x.triggered.connect(self.speed_2x)
        self.actionSpeed_4x.triggered.connect(self.speed_4x)
        self.actionMap.triggered.connect(self.on_map_view)
        self.actionControl.triggered.connect(self.on_control_view)
        self.actionNewPlot.triggered.connect(self.on_new_plot)
        self.actionSensors.triggered.connect(self.on_sensors)
        self.actionTrack.triggered.connect(self.on_track)
        self.actionVisualize.triggered.connect(self.on_visualize)
        self.actionInfo.triggered.connect(self.on_info)
        self.actionExit.triggered.connect(self.on_exit)
        self.actionLoad.triggered.connect(self.on_load)
        self.actionExport.triggered.connect(self.on_export)
        self.actionNew.triggered.connect(self.on_new_sim)
        self.actionSettings.triggered.connect(self.on_settings)
        self.actionSetup.triggered.connect(self.on_learner_settings)
        self.actionAbout.triggered.connect(self.on_about)
        self.actionHelp.triggered.connect(self.on_help)
        self.actionSave.triggered.connect(self.on_save_simulation)
        self.actionSaveAs.triggered.connect(self.on_new_sim)
        self.actionImport.triggered.connect(self.on_import)
        self.actionEvaluationPlot.triggered.connect(self.on_evaluation_plot)
        self.actionStars.triggered.connect(self.on_stars)
        self.actionReport.triggered.connect(self.on_report)
        # QT渲染设置
        self.image_steering = QImage('Resources/Rendering/steering.png')
        effect = QGraphicsOpacityEffect()
        effect.setOpacity(0.6)
        self.label_map.setGraphicsEffect(effect)
        # 插入仿真数据显示列表
        self.curveSelect = QComboBox()
        self.toolBarView.insertWidget(self.actionNewPlot, self.curveSelect)
        self.plotter = Plotter()
        for tp in self.plotter.get_types():
            self.curveSelect.addItem(self.tr(tp))
        # 插入评价数据显示列表
        self.evaluation_plotter = EvaluationPlotter()
        self.eval_curveSelect = QComboBox()
        self.toolBarEval.insertWidget(self.actionEvaluationPlot,
                                      self.eval_curveSelect)
        for tp in self.evaluation_plotter.curve_names:
            self.eval_curveSelect.addItem(self.tr(tp))
        # 设置QT中的滑块组件(回放仿真数据中用到)
        self.slider = QSlider(1)
        self.toolBarSim.addWidget(self.slider)
        self.connect(self.slider, QtCore.SIGNAL('valueChanged(int)'),
                     self.on_slider)
        # 启动QT定时器
        self.timer = QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.on_timer)

        self.reset()

    def on_slider(self):
        """
        回放模式下滑块响应事件

        Returns:

        """
        if self.replay_mode:
            self.replay_data.set_frame(self.slider.value())
            self.replay_refresh()

    def reset(self):
        """
        重置LasVSim主界面

        Returns:

        """
        self.SpeedX = True
        self.map_view_flag = True
        self.control_view_flag = True
        self.SensorsEnabled = True
        self.TrackEnabled = True
        self.VisualizeEnabled = True
        self.InfoEnabled = True

        self.actionMap.setChecked(True)
        self.actionControl.setChecked(True)
        self.actionSpeed_1x.setChecked(True)
        self.actionSensors.setChecked(True)
        self.actionTrack.setChecked(True)
        self.actionInfo.setChecked(True)
        self.actionStars.setChecked(True)
        self.actionVisualize.setChecked(True)
        self.actionPause.setEnabled(False)
        self.actionStop.setEnabled(False)
        self.actionPause.setChecked(False)

        self.replay_mode = False
        self.plotter.set_replay_mode(0)
        self.slider.setEnabled(False)

        self.actionReport.setEnabled(False)

        self.StarsEnabled = True
        self.actionStars.setVisible(True)

        self.on_simulation_loaded()

    def on_simulation_loaded(self):
        """
        读取仿真项目响应事件

        Returns:

        """
        # 仿真界面显示控制
        self.label_map.setVisible(self.simulation_load_flag)
        self.infoFrame.setVisible(self.simulation_load_flag)
        self.controlFrame.setVisible(self.simulation_load_flag)
        self.evalFrame.setVisible(self.simulation_load_flag)

        # 按钮开关控制
        self.actionMap.setEnabled(self.simulation_load_flag)
        self.actionControl.setEnabled(self.simulation_load_flag)
        self.actionSensors.setEnabled(self.simulation_load_flag)
        self.actionTrack.setEnabled(self.simulation_load_flag)
        self.actionInfo.setEnabled(self.simulation_load_flag)
        self.actionStars.setEnabled(self.simulation_load_flag)
        self.actionVisualize.setEnabled(self.simulation_load_flag)
        self.actionZoomIn.setEnabled(self.simulation_load_flag)
        self.actionZoomOut.setEnabled(self.simulation_load_flag)
        self.actionZoomReset.setEnabled(self.simulation_load_flag)
        self.actionStart.setEnabled(self.simulation_load_flag)
        self.actionPause.setEnabled(self.simulation_load_flag)
        self.actionSpeed_1x.setEnabled(self.simulation_load_flag)
        self.actionSpeed_2x.setEnabled(self.simulation_load_flag)
        self.actionSpeed_4x.setEnabled(self.simulation_load_flag)
        self.actionNewPlot.setEnabled(self.simulation_load_flag)
        self.actionExport.setEnabled(self.simulation_load_flag)
        self.actionSettings.setEnabled(self.simulation_load_flag)
        self.actionSetup.setEnabled(self.simulation_load_flag)
        self.actionSave.setEnabled(self.simulation_load_flag)
        self.actionSaveAs.setEnabled(self.simulation_load_flag)
        self.actionEvaluationPlot.setEnabled(self.simulation_load_flag)
        self.curveSelect.setEnabled(self.simulation_load_flag)
        self.eval_curveSelect.setEnabled(self.simulation_load_flag)

    # def set_plotter_data(self, data):
    #     """
    #     输入仿真数据
    #
    #     Args:
    #         data:
    #
    #     Returns:
    #
    #     """
    #     self.plotter.set_data(data)

    def on_load(self):
        """
        读取已经存在的仿真项目

        Returns:

        """
        self.current_simulation_path = QFileDialog.getExistingDirectory(
            self, "load simulation", "/")
        if len(self.current_simulation_path) == 0:
            return
        self.actionStart.setEnabled(True)
        self.actionPause.setEnabled(False)
        self.actionStop.setEnabled(False)
        self.actionReport.setEnabled(False)
        self.timer.stop()
        self.current_configuration_file_path = (self.current_simulation_path +
                                                '\simulation setting file.xml')
        settings.load(self.current_configuration_file_path)
        simulation.reset(settings,
                         simulation.init_traffic_distribution_data.load_traffic(
                             self.current_simulation_path))
        self.evaluation_plotter.plot_points = int(
            30000.0 / simulation.settings.step_length)
        self.plotter.plot_points = int(
            30000.0 / simulation.settings.step_length)
        self.plotter.set_data(simulation.data)

        # 关闭replay模式
        self.replay_mode = False
        self.plotter.set_replay_mode(0)
        self.slider.setEnabled(False)
        self.slider.setValue(0)

        simulation.draw()
        self.init_map_view()
        self.refresh_display(simulation.get_canvas())
        self.statusBar().showMessage('Time: %.2f s' % simulation.get_time())

        self.simulation_load_flag = True
        self.on_simulation_loaded()
        self.refresh_display(simulation.get_canvas())

    def on_settings(self):
        """
        仿真设置

        Returns:

        """
        self.timer.stop()
        self.actionPause.setChecked(True)
        dialog = SettingsDialog()
        dialog.init_view(settings)
        ret = dialog.exec_()
        if ret != 1:
            return False
        self.actionStart.setEnabled(True)
        self.actionPause.setEnabled(False)
        self.actionStop.setEnabled(False)
        self.actionReport.setEnabled(False)
        simulation.reset(dialog.settings)
        self.evaluation_plotter.plot_points = int(
            30000.0 / simulation.settings.step_length)
        self.plotter.plot_points = int(
            30000.0 / simulation.settings.step_length)
        self.plotter.set_data(simulation.data)

        # 关闭replay模式
        self.replay_mode=0
        self.plotter.set_replay_mode(0)
        self.slider.setEnabled(False)
        self.slider.setValue(0)

        simulation.draw()
        self.init_map_view()
        self.refresh_display(simulation.get_canvas())
        self.statusBar().showMessage('Time: %.2f s' % simulation.get_time())
        return True

    def on_learner_settings(self):
        """
        环境设置(用于导出供LasVSim-package使用的仿真项目)

        Returns:

        """
        self.timer.stop()
        self.actionPause.setChecked(True)
        dialog = LearnerDialog()
        dialog.init_view(learner_settings)
        ret = dialog.exec_()
        if ret != 1:
            return

    def on_visualize(self):
        """
        渲染开关函数

        Returns:

        """
        simulation.set_visualize(1 - simulation.VisualizeEnabled)
        self.actionVisualize.setChecked(simulation.VisualizeEnabled)
        self.actionTrack.setChecked(simulation.VisualizeEnabled)
        self.actionInfo.setChecked(simulation.VisualizeEnabled)
        self.actionMap.setChecked(simulation.VisualizeEnabled)
        self.actionSensors.setChecked(simulation.VisualizeEnabled)
        self.actionControl.setChecked(simulation.VisualizeEnabled)
        self.actionTrack.setEnabled(simulation.VisualizeEnabled)
        self.actionInfo.setEnabled(simulation.VisualizeEnabled)
        self.actionMap.setEnabled(simulation.VisualizeEnabled)
        self.actionSensors.setEnabled(simulation.VisualizeEnabled)
        self.actionControl.setEnabled(simulation.VisualizeEnabled)
        self.on_track(simulation.VisualizeEnabled)
        self.on_info(simulation.VisualizeEnabled)
        self.on_sensors(simulation.VisualizeEnabled)
        self.on_control_view(simulation.VisualizeEnabled)
        self.on_map_view(simulation.VisualizeEnabled)

        simulation.set_visualize(simulation.VisualizeEnabled)
        #simulation.disp.set_visualize_enabled(simulation.VisualizeEnabled==1)
        #simulation.draw()
        self.refresh_display(simulation.get_canvas())

    def on_report(self):
        """
        生成评价报告,word格式(原华为平台功能，目前暂时弃用)
        Returns:

        """
        # TODO(Chason): 后期待修复的功能
        simulation.evaluator.release()
        subprocess.Popen('generate_report.exe',
                         startupinfo=subprocess.STARTUPINFO())

    def on_new_sim(self):
        """
        新建仿真项目

        Returns:

        """

        self.current_simulation_path = QFileDialog.getSaveFileName(
            self, "new simulation", ".", "LasVSim Simulation")
        if len(self.current_simulation_path) == 0:
            return
        # 第一次启动LasVSim会创建一个新的仿真配置文件
        if self.current_configuration_file_path is None:
            if not self.on_settings():
                return

        # 创建LasVSim工程文件夹
        self.current_simulation_path = self.current_simulation_path + '_LasVSim'
        os.mkdir(self.current_simulation_path)
        self.current_configuration_file_path = (
            self.current_simulation_path + '/simulation setting file.xml')

        self.simulation_load_flag = True
        self.on_simulation_loaded()
        self.on_save()

    def on_save(self):
        """
        保存仿真项目文件
        包括配置文件simulation setting file.xml
        和初始交通流数据simulation traffic data.bin）

        Returns:

        """
        if len(self.current_configuration_file_path) == 0:
            self.on_new_sim()
            return
        if simulation.traffic.traffic_change_flag:
            _logger.info('New traffic data saved')
            simulation.init_traffic_distribution_data = data_module.TrafficData()  # TODO(Chason): 在TrafficData类中加入reset
            simulation.init_traffic_distribution_data.save_traffic(
                simulation.traffic.random_traffic, self.current_simulation_path)

        simulation.settings.save(self.current_configuration_file_path)

        # 关闭replay模式
        self.replay_mode = 0
        self.plotter.set_replay_mode(0)
        self.slider.setEnabled(False)
        self.slider.setValue(0)

    def on_save_as(self):
        """
        将当前仿真项目另存为

        Returns:

        """
        self.on_new_sim()

    def on_save_simulation(self):
        """
        保存当前仿真项目

        Returns:

        """
        self.on_save()

    def on_export(self):
        """
        导出仿真数据文件

        Returns:

        """
        self.stop()
        file_path=QFileDialog.getSaveFileName(self,
                                              "export simulation data",
                                              ".",
                                              "Data File(*.csv)")
        if len(file_path) == 0:
            return
        simulation.data.export_csv(file_path)

    def on_import(self):
        """
        导入仿真数据文件(进入回放模式)

        Returns:

        """
        file_path=QFileDialog.getOpenFileName(self, "load simulation data",
                                              ".", "Data File(*.csv)")
        if len(file_path) == 0:
            return
        if not self.simulation_load_flag:
            self.on_simulation_loaded()
            simulation.reset(settings)
            self.evaluation_plotter.plot_points = int(
                30000.0 / simulation.settings.step_length)
            self.plotter.plot_points = int(
                30000.0 / simulation.settings.step_length)
            self.simulation_load_flag = True
            self.on_simulation_loaded()

        # 按钮开关控制
        self.actionMap.setEnabled(self.simulation_load_flag)
        self.actionControl.setEnabled(self.simulation_load_flag)
        self.actionSensors.setEnabled(self.simulation_load_flag)
        self.actionTrack.setEnabled(self.simulation_load_flag)
        self.actionInfo.setEnabled(self.simulation_load_flag)
        self.actionStars.setEnabled(False)
        self.actionVisualize.setEnabled(False)
        self.actionZoomIn.setEnabled(self.simulation_load_flag)
        self.actionZoomOut.setEnabled(self.simulation_load_flag)
        self.actionZoomReset.setEnabled(self.simulation_load_flag)
        self.actionStart.setEnabled(self.simulation_load_flag)
        self.actionPause.setEnabled(self.simulation_load_flag)
        self.actionSpeed_1x.setEnabled(self.simulation_load_flag)
        self.actionSpeed_2x.setEnabled(self.simulation_load_flag)
        self.actionSpeed_4x.setEnabled(self.simulation_load_flag)
        self.actionNewPlot.setEnabled(self.simulation_load_flag)
        self.actionExport.setEnabled(False)
        self.actionSettings.setEnabled(False)
        self.actionSetup.setEnabled(False)
        self.actionSave.setEnabled(False)
        self.actionSaveAs.setEnabled(False)
        self.actionEvaluationPlot.setEnabled(self.simulation_load_flag)
        self.curveSelect.setEnabled(self.simulation_load_flag)
        self.eval_curveSelect.setEnabled(self.simulation_load_flag)

        self.replay_data = data_module.ReplayData(file_path)
        self.replay_mode = 1
        self.plotter.set_replay_mode(1)
        self.plotter.set_replay_data(self.replay_data)
        self.slider.setEnabled(True)
        self.slider.setRange(0, self.replay_data.frame_count-1)
        self.replay_refresh()

    def get_display_canvas(self):
        """
        返回QT窗口的画布(用于渲染)

        Returns:

        """
        w = self.label_display.width()
        h = self.label_display.height()
        if w % 2 == 0:
            w += 1
        if h % 2 == 0:
            h += 1
        return np.zeros((h, w, 3), dtype='uint8')

    def on_sensors(self, sensor_detection_range_visualizing_flag=None):
        """
        传感器探测范围显示函数
        
        Args:
            sensor_detection_range_visualizing_flag(bool): 

        Returns:

        """
        if sensor_detection_range_visualizing_flag is None:
            self.SensorsEnabled = 1-self.SensorsEnabled
        else:
            self.SensorsEnabled = sensor_detection_range_visualizing_flag
        self.actionSensors.setChecked(self.SensorsEnabled)
        simulation.disp.set_sensor_enabled(self.SensorsEnabled == 1)
        simulation.draw()
        self.refresh_display(simulation.get_canvas())

    def on_info(self, InfoEnabled=None):
        """
        自车位姿显示函数

        Args:
            InfoEnabled(bool):

        Returns:

        """
        if InfoEnabled is None:
            self.InfoEnabled = 1-self.InfoEnabled
        else:
            self.InfoEnabled = InfoEnabled
        self.actionInfo.setChecked(self.InfoEnabled)
        self.infoFrame.setVisible(self.InfoEnabled)
        self.refresh_info_view()

    def on_track(self, TrackEnabled=None):
        """
        期望轨迹显示函数

        Args:
            TrackEnabled(bool):

        Returns:

        """
        if TrackEnabled is None:
            self.TrackEnabled = 1-self.TrackEnabled
        else:
            self.TrackEnabled = TrackEnabled
        # self.actionTrack.setChecked(self.TrackEnabled)
        simulation.disp.set_track_enabled(self.TrackEnabled == 1)
        simulation.draw()
        self.refresh_display(simulation.get_canvas())

    def refresh_display(self, canvas):
        """
        刷新渲染画面

        Args:
            canvas:

        Returns:

        """
        if simulation.VisualizeEnabled != 1 or self.simulation_load_flag != 1:
            return
        h, w, d = canvas.shape
        step = w*d
        data = voidptr(canvas.data, h * step, True)
        img = QImage(data, w, h, step, QImage.Format_RGB888)
        canvas[:] = canvas[:, :, ::-1]
        self.label_display.setPixmap(QPixmap.fromImage(img))
        self.refresh_map_view()
        if self.replay_mode is not 1:
            self.refresh_control_view(*simulation.get_self_car_info())
        else:
            self.refresh_control_view(
                self.replay_data.get_current_state('Steering Wheel'),
                self.replay_data.get_current_state('Throttle'),
                self.replay_data.get_current_state('Brake Pressure'),
                self.replay_data.get_current_state('Gear'))
        self.refresh_info_view()
        self.refresh_plot()
        self.refresh_evaluation_view()

    def refresh_map_view(self):
        """
        刷新小地图

        Returns:

        """
        if self.map_view_flag != 1:
            return
        x, y, v, heading = simulation.get_pos()
        map_view.set_vehicle_pos((x, y, -heading))
        h, w, d = map_view.image.shape
        step = w*d
        data = voidptr(map_view.image.data, h*step, True)
        img = QImage(data, w, h, step, QImage.Format_RGB888)
        map_view.image[:] = map_view.image[:, :, ::-1]
        self.label_map.setPixmap(QPixmap.fromImage(img))

    def zoom_in(self):
        """
        放大仿真画面

        Returns:

        """
        if self.timer.isActive():
            self.timer.stop()
            simulation.disp.zoom('zoom_in')
            simulation.draw()
            self.refresh_display(simulation.get_canvas())
            self.timer.start()
        else:
            simulation.disp.zoom('zoom_in')
            simulation.draw()
            self.refresh_display(simulation.get_canvas())

    def zoom_out(self):
        """
        缩小仿真画面

        Returns:

        """
        if self.timer.isActive():
            self.timer.stop()
            simulation.disp.zoom('zoom_out')
            simulation.draw()
            self.refresh_display(simulation.get_canvas())
            self.timer.start()
        else:
            simulation.disp.zoom('zoom_out')
            simulation.draw()
            self.refresh_display(simulation.get_canvas())

    def zoom_reset(self):
        """
        重置渲染画面比例

        Returns:

        """
        if self.timer.isActive():
            self.timer.stop()
            simulation.disp.zoom('zoom_reset')
            simulation.draw()
            self.refresh_display(simulation.get_canvas())
            self.timer.start()
        else:
            simulation.disp.zoom('zoom_reset')
            simulation.draw()
            self.refresh_display(simulation.get_canvas())

    def on_new_plot(self):
        """
        添加仿真数据图

        Returns:

        """
        self.plotter.add_plot(self.curveSelect.currentIndex())

    def on_evaluation_plot(self):
        """
        添加评价数据图

        Returns:

        """
        if self.replay_mode == 1:
            return
        self.evaluation_plotter.add_plot(self.eval_curveSelect.currentIndex())

    def on_stars(self):
        """
        评价分数显示开关

        Returns:

        """
        self.StarsEnabled = 1 - self.StarsEnabled
        self.actionStars.setChecked(self.StarsEnabled)
        self.evalFrame.setVisible(self.StarsEnabled)

    def refresh_plot(self):
        """
        刷新仿真数据图

        Returns:

        """
        self.plotter.refresh_all_plot()
        if self.replay_mode is not True:
            self.evaluation_plotter.refresh_all_plot()
        else:
            self.evaluation_plotter.close()

    def on_timer(self):
        """
        QT定时器响应事件(仿真一步)

        Returns:

        """
        if self.replay_mode is not 1:
            self.sim_step()
            self.statusBar().showMessage('Time: %.2f s'%simulation.get_time())
        else:
            self.replay_step()

    def replay_step(self):
        """
        回放模式下的单步更新函数

        Returns:

        """
        if self.replay_data.step(self.SpeedX):
            self.replay_refresh()
        else:
            self.pause()

    def replay_refresh(self):
        """
        刷新回放画面

        Returns:

        """
        x, y, v, heading = self.replay_data.get_self_pos()
        simulation.mission_update((x, y, v, -heading+90))
        simulation.agent.dynamic.ego_x = x
        simulation.agent.dynamic.ego_y = y
        simulation.agent.dynamic.ego_vel = v
        simulation.agent.dynamic.ego_heading = heading
        simulation.veh_info = self.replay_data.get_other_vehicles()
        light_status = self.replay_data.get_light_status()
        trajectories = self.replay_data.get_current_trajectory()
        simulation.disp.set_data([], light_status, (x, y, v, heading), [],
                                 simulation.veh_info)
        simulation.disp.set_info(dict(t=self.replay_data.get_current_time(),
                                      rotation=0, winker=0), trajectories)
        simulation.disp.set_pos(*simulation.map.get_display_pos((x, y), heading))
        simulation.draw()
        self.refresh_display(simulation.get_canvas())
        self.slider.setSliderPosition(self.replay_data.frame_index)
        self.statusBar().showMessage('[Replay Mode] Time: %.2f s' %
                                     self.replay_data.get_current_time())
        self.actionReport.setEnabled(False)

    def sim_step(self):
        """
        普通模式下的单步更新函数

        Returns:

        """
        if simulation.sim_step_internal(self.SpeedX) == 1:
            self.refresh_display(simulation.get_canvas())

    def start(self):
        """
        仿真开始

        Returns:

        """
        if simulation.stopped and self.replay_mode != 1:
            simulation.reset(settings)
            self.evaluation_plotter.plot_points = int(
                30000.0 / simulation.settings.step_length)
            self.plotter.plot_points = int(
                30000.0 / simulation.settings.step_length)
            self.plotter.set_data(simulation.data)
        self.actionStart.setEnabled(False)
        self.actionPause.setEnabled(True)
        self.actionStop.setEnabled(True)
        self.actionReport.setEnabled(False)
        self.actionPause.setChecked(False)
        if self.replay_mode == 1:
            self.replay_data.set_frame(0)
        self.timer.start()

    def pause(self):
        """
        仿真暂停

        Returns:

        """
        if self.timer.isActive():
            self.timer.stop()
            self.actionPause.setChecked(True)
        else:
            self.timer.start()
            self.actionPause.setChecked(False)

    def stop(self):
        """
        仿真终止

        Returns:

        """
        self.actionStart.setEnabled(True)
        self.actionPause.setEnabled(False)
        self.actionStop.setEnabled(False)
        self.timer.stop()
        if self.replay_mode == 0:
            self.actionReport.setEnabled(True)
            simulation.stop()

    def speed_1x(self):
        """
        一倍速渲染

        Returns:

        """
        self.SpeedX = 1
        self.actionSpeed_1x.setChecked(True)
        self.actionSpeed_2x.setChecked(False)
        self.actionSpeed_4x.setChecked(False)

    def speed_2x(self):
        """
        二倍速渲染

        Returns:

        """
        self.SpeedX = 2
        self.actionSpeed_1x.setChecked(False)
        self.actionSpeed_2x.setChecked(True)
        self.actionSpeed_4x.setChecked(False)

    def speed_4x(self):
        """
        四倍速渲染

        Returns:

        """
        self.SpeedX = 4
        self.actionSpeed_1x.setChecked(False)
        self.actionSpeed_2x.setChecked(False)
        self.actionSpeed_4x.setChecked(True)

    def on_map_view(self, map_view_enabled=None):
        """
        小地图显示

        Args:
            map_view_enabled(bool):

        Returns:

        """
        if map_view_enabled is None:
            self.map_view_flag = 1 - self.map_view_flag
        else:
            self.map_view_flag = map_view_enabled
        self.actionMap.setChecked(self.map_view_flag)
        self.label_map.setVisible(self.map_view_flag)

    def on_control_view(self, control_view_enable=None):
        """
        控制量显示

        Args:
            control_view_enable(bool):

        Returns:

        """
        if control_view_enable is None:
            self.control_view_flag = 1 - self.control_view_flag
        else:
            self.control_view_flag = control_view_enable
        self.actionControl.setChecked(self.control_view_flag)
        self.controlFrame.setVisible(self.control_view_flag)

    def init_map_view(self):
        """
        初始化小地图

        Returns:

        """
        map_view.__init__(simulation.settings.map)
        mission = simulation.agent.mission
        map_view.set_route(simulation.map.get_display_route([mission.points[0],
                                                            mission.points[-1]],
                                                            mission.cross_list))
        x, y, v, heading = mission.pos
        # a = -a+90
        map_view.set_vehicle_pos((x, y, -heading))
        h, w, d = map_view.image.shape
        step = w*d
        data = voidptr(map_view.image.data, h*step, True)
        img = QImage(data, w, h, step, QImage.Format_RGB888)
        map_view.image[:] = map_view.image[:, :, ::-1]
        self.label_map.setPixmap(QPixmap.fromImage(img))

    def set_steering(self, ang):
        """
        设置方向盘转角显示量

        Args:
            ang(float): deg

        Returns:

        """
        self.text_steering.setText(str('%d'%ang+'°'))#.encode('utf-8').decode('utf-8')
        m=QMatrix()
        m.rotate(-ang)
        self.label_steering.setPixmap(QPixmap.fromImage(self.image_steering.transformed(m,Qt.SmoothTransformation)))

    def set_throttle(self, throttle):
        """
        设置油门开度显示量

        Args:
            throttle(float): %

        Returns:

        """
        self.text_acc.setText('%d%%' % throttle)
        w = self.label_acc.width()
        h = self.label_acc.height()
        img = QImage(QSize(w, h), QImage.Format_RGB888)
        img.fill(QColor(180, 250, 180))
        p = QPainter(img)
        p.setPen(Qt.green)
        p.setBrush(QBrush(Qt.green, Qt.SolidPattern))
        p.drawRect(QRectF(0, 0, w * throttle / 100, h))
        p.end()
        self.label_acc.setPixmap(QPixmap.fromImage(img))

    def set_brake(self, brake):
        """
        设置制动压力显示量
        Args:
            brake(float): MPa

        Returns:

        """
        self.text_brake.setText('%.1f MPa' % brake)
        brake_pct = brake/8.0*100
        w = self.label_brake.width()
        h = self.label_brake.height()
        img = QImage(QSize(w, h), QImage.Format_RGB888)
        img.fill(QColor(250, 180, 180))
        p = QPainter(img)
        p.setPen(Qt.red)
        p.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        p.drawRect(QRectF(0, 0, w*brake_pct/100, h))
        p.end()
        self.label_brake.setPixmap(QPixmap.fromImage(img))

    def set_gear(self, gear):
        """
        设置传动比显示量

        Args:
            gear(float):

        Returns:

        """
        if gear == 0:
            str_gear = '-'
        else:
            str_gear = '%.2f' % gear
        # TODO(Chason):回放模式会默认以下条件触发，后续改正
        # if simulation.get_dynamic_type() == 'CVT Car':
        #     str_gear = '-'
        self.text_gear.setText(str_gear)

    def refresh_control_view(self, steering, throttle, brake, gear):
        """
        刷新显示控制量

        Args:
            steering(float): deg
            throttle(float): %
            brake(float): MPa
            gear(float):

        Returns:

        """
        if self.control_view_flag != 1:
            return
        self.set_steering(steering)
        self.set_throttle(throttle)
        self.set_brake(brake)
        self.set_gear(gear)

    def refresh_info_view(self):
        """
        刷新自车位姿显示量

        Returns:

        """
        if self.InfoEnabled != 1:
            return
        x, y, v, heading = simulation.get_pos()
        heading = degree_fix(-heading + 90)
        if v < 0.1 and simulation.get_time() < 1:
            mission = 'start'
        else:
            mission = simulation.agent.mission.get_description()
        self.text_mission.setText(mission)
        self.text_speed.setText('%.1f km/h' % (v*3.6))
        self.text_x.setText('%.1f m' % x)
        self.text_y.setText('%.1f m' % y)
        self.text_direction.setText(str('%.1f °' % heading))#.encode('utf-8').decode('utf-8')

    def get_evaluation_stars(self, n):
        """
        显示评价分数

        Args:
            n(int): 分值. 0~5

        Returns:

        """
        return ('★'*int(n)+'☆'*(5-int(n)))

    def refresh_evaluation_view(self):
        """
        刷新评价分数显示

        Returns:

        """
        if self.replay_mode == 1:
            self.evalFrame.setVisible(False)
            return
        else:
            self.evalFrame.setVisible(self.StarsEnabled)
        if self.StarsEnabled != 1:
            return
        safety, economy, comfort, efficiency = (simulation.evaluator.get_report())[0:4]
        self.labelSecurity.setText(self.get_evaluation_stars(safety))
        self.labelEconomy.setText(self.get_evaluation_stars(economy))
        self.labelComfortability.setText(self.get_evaluation_stars(comfort))
        self.labelSpeed.setText(self.get_evaluation_stars(efficiency))

    def on_exit(self):
        """
        推出程序

        Returns:

        """
        self.close()

    def on_help(self):
        """
        打开帮助文件

        Returns:

        """
        os.startfile('Help.pdf')
        # self.on_about()

    def on_about(self):
        """
        显示软件相关信息

        Returns:

        """
        msg = '\nLasVSim-gui 0.2.1_alpha\n\n'
        msg += 'Developed by :\n'
        msg += 'The Intelligent Driving Laboratory(iDLab), ' \
               'Tsinghua University\n'
        msg += '\n2019.'
        QMessageBox.about(self, 'About this software : ', msg)

    def closeEvent(self, *args, **kwargs):
        """
        主界面关闭响应事件

        Args:
            *args:
            **kwargs:

        Returns:

        """
        self.plotter.close()
        self.evaluation_plotter.close()

    def resizeEvent(self, event):
        """
        主界面尺寸变化响应事件

        Args:
            event:

        Returns:

        """
        self.label_display.resize(self.centralWidget().size())
        if self.simulation_load_flag:
            simulation.reset_canvas(self.get_display_canvas())
        else:
            simulation.init_canvas(self.get_display_canvas())
        self.refresh_display(simulation.get_canvas())

        x = self.label_map.x()
        y = self.centralWidget().height()-self.label_map.height()-5
        self.label_map.move(QPoint(x,y))

        x1,y1=self.controlFrame.x(),self.controlFrame.y()
        x1=self.centralWidget().width()-self.controlFrame.width()-5
        y1=self.centralWidget().height()-self.controlFrame.height()-5
        self.controlFrame.move(QPoint(x1,y1))

        x2,y2=self.evalFrame.x(),self.evalFrame.y()
        x2=self.centralWidget().width()-self.evalFrame.width()-5
        self.evalFrame.move(QPoint(x2,y2))


if __name__ == "__main__":
    """
   Program Entry Point 
   """
    freeze_support()

    settings = Settings()
    settings.load('Library/default_simulation_setting.xml')    # load default settings
    learner_settings = Settings()
    learner_settings.load('Library/default_simulation_setting.xml')

    app = QtGui.QApplication(sys.argv)

    window = MainUI()    # create window
    main_window = window
    simulation = Simulation(canvas=window.get_display_canvas()) # create simulation object

    window.show()   # show window

    sys.exit(app.exec_())
