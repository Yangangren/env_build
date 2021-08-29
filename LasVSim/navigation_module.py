# coding=utf-8

"""
Mission Edit Dialog of Autonomous Car Simulation System
Author: Li Bing
Date: 2017-8-23
"""

from PyQt4.Qt import *
from PyQt4 import QtCore, QtGui,uic
from map_module import *
from data_structures import *


class MapDialog(QtGui.QDialog):
    """
    Map Dialog Class
    """
    def __init__(self, map, parent=None):
        QtGui.QDialog.__init__(self, parent)
        uic.loadUi('QT UI/PickInMap.ui',self)
        font=QFont()
        font.setPixelSize(12)

        icon = QIcon()
        icon.addPixmap(QPixmap("Resources/Logos/navigation.ico"), QIcon.Normal)
        self.setWindowIcon(icon)

        self.setFont(font)
        self.points = []
        self.pick = None
        self.map_type = map
        self.map = Map(map)
        self.buttonOK.clicked.connect(self.on_ok)
        self.buttonReset.clicked.connect(self.on_reset)
        # self.buttonCancel.clicked.connect(self.on_cancel)
        self.buttonSource.clicked.connect(self.on_source)
        self.buttonLoop.clicked.connect(self.on_loop)
        self.buttonLoop.setEnabled(False)
        self.mission=None
        self.mission_type = 'One Way'

        if self.map_type == MAPS[0]:
            img = QImage('Resources/Tools/urbanroad_map.png')
            self.label_map.setPixmap(QPixmap.fromImage(img))
        elif self.map_type == MAPS[1]:
            img = QImage('Resources/Rendering/highway_map.png')
            self.label_map.setPixmap(QPixmap.fromImage(img))
            self.buttonReset.setEnabled(False)
            self.buttonSource.setEnabled(False)
            self.high_way_route()

    def on_source(self):
        self.pick='Source'
        self.setCursor(Qt.CrossCursor)

    def on_ok(self):
        if self.mission is not None:
            self.done(1)
        else:
            self.done(0)

    def on_loop(self):
        self.mission_type = 'Loop'
        self.buttonLoop.setEnabled(False)
        self.buttonSource.setEnabled(False)
        self.points.append(self.map.get_nearest_lane(self.points[0][0],
                                                     self.points[0][1],
                                                     'Source'))
        self.refresh()

    def on_reset(self):
        self.points = []
        self.buttonLoop.setEnabled(False)
        self.buttonSource.setEnabled(True)
        self.mission_type = 'One Way'
        self.refresh()

    def mouseReleaseEvent(self, mouse_event):
        if self.pick is None:
            return
        elif self.pick == 'Source':
            x0, y0 = mouse_event.x(), mouse_event.y()
            x1 = x0 - self.label_map.x()
            y1 = y0 - self.label_map.y()
            if (0 <= x1 <= self.label_map.width() and
                            0 <= y1 <= self.label_map.height()):
                x = (float(x1)/self.label_map.width()-0.5)*2*MAP_MAX_X
                y = (0.5-float(y1)/self.label_map.width())*2*MAP_MAX_Y

                self.y = y
                self.x = x

                if len(self.points) == 0:
                    self.points.append(self.map.get_nearest_lane(self.x, self.y,
                                                                 'Source'))
                else:
                    self.points.append(self.map.get_nearest_lane(self.x, self.y,
                                                                 'Target'))
                self.refresh()
            self.setCursor(Qt.ArrowCursor)
            self.pick = None

        if len(self.points) > 1:
            self.buttonLoop.setEnabled(True)

    def high_way_route(self):
        self.points = []
        self.points.append(self.map.get_nearest_lane(-876.35, -5.74,
                                                     'Source'))

        self.points.append(self.map.get_nearest_lane(876.35, -5.74,
                                                         'Target'))
        self.refresh()

    def geo2pix(self,p):
        x, y = p
        x, y = x/MAP_MAX_X/2+0.5, 0.5-y/MAP_MAX_Y/2
        w = self.label_map.width()
        h = self.label_map.height()
        x, y = int(x*w), int(y*h)
        return x, y

    def refresh(self):
        """计算全局最短路径同时显示在小地图上

        --
        """
        if len(self.points) == 0:
            self.label_position_source.setText('-')
            self.label_dir_source.setText('-')
            self.label_position_target.setText('-')
            self.label_dir_target.setText('-')
        elif len(self.points) == 1:
            self.label_position_source.setText(
                '(%.1f,%.1f)' % (self.points[0][:2]))
            self.label_dir_source.setText(self.points[0][2])
        else:
            self.label_position_target.setText('(%.1f,%.1f)'%(self.points[-1][:2]))
            self.label_dir_target.setText(self.points[-1][2])

        if self.map_type == MAPS[0]:
            img = QImage('Resources/Tools/urbanroad_map.png')  # Draw map background.
        elif self.map_type == MAPS[1]:
            img = QImage('Resources/Rendering/highway_map.png')
            self.label_position_source.setText('(-876.35, -5.74)')
            self.label_dir_source.setText('East')
            self.label_position_target.setText('(876.35, -5.74)')
            self.label_dir_target.setText('East')

        painter = QPainter(img)

        if len(self.points) >= 2:
            painter.setPen(QPen(Qt.darkYellow, 3, Qt.SolidLine))
            points = []
            for i in range(len(self.points)):
                a = 360-self.map.direction_to_angle(self.points[i][2][0])
                if a >= 270:
                    a -= 360
                points.append([self.points[i][0], self.points[i][1], 0, a])
            self.mission = points
            if self.map_type == MAPS[1]:
                self.mission = [[-876.35, -5.74, 20.0, 90.0],
                           [876.35, -5.74, 22.22, 90.0]]
            mission = Mission(self.map, points)
            if not mission.has_path:
                self.points.pop()
                points.pop()
            route = self.map.get_display_route(points, mission.cross_list)
            for i in range(len(route)-1):
                x1, y1 = self.geo2pix(route[i])
                x2, y2 = self.geo2pix(route[i+1])
                painter.drawLine(x1, y1, x2, y2)

        painter.setPen(Qt.black)
        for i in range(len(self.points)):
            painter.setBrush(Qt.green)
            painter.drawEllipse(QPoint(*self.geo2pix(self.points[i][:2])),4,4)

        painter.end()
        self.label_map.setPixmap(QPixmap.fromImage(img))
