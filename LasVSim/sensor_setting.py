# coding=utf-8

"""
Sensor Edit Dialog of Autonomous Car Simulation System
Author: Li Bing
Date: 2017-8-23
"""

from PyQt4.Qt import *
from PyQt4 import QtCore, QtGui,uic
from simulator import *
# from Modules import *
from sensor_module import *
import copy


SENSOR_LIBRARY_PATH='Library/sensor_library.csv'


class SensorLibrary:
    """
    Sensor Library: all sensor types and parameters
    """
    def __init__(self):
        self.sensors = []
        with open(SENSOR_LIBRARY_PATH) as f:
            attrs = f.readline().split(',')
            for line in f:
                self.sensors.append(dict(zip(attrs,line.split(','))))

    def types(self):
        return [self.sensors[i]['Name'] for i in range(len(self.sensors))]

    def get_info(self, i):
        return self.sensors[i]


sensor_libray = SensorLibrary()
types=sensor_libray.types()

def get_float(s):
    """
    convert string to float value
    """
    try:
        return float(str(s))
    except:
        return 0.0

class SensorDialog(QtGui.QDialog):
    """
    Sensor Edit Dialog Class
    """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        uic.loadUi('QT UI/SensorEdit.ui',self)
        font=QFont()
        font.setPixelSize(12)
        self.setFont(font)
        self.tableWidget.itemSelectionChanged.connect(self.on_table_clicked)
        self.deleteButton.clicked.connect(self.on_delete)
        self.addButton.clicked.connect(self.on_add)
        self.updateButton.clicked.connect(self.on_update)
        self.okButton.clicked.connect(self.on_ok)
        self.cancelButton.clicked.connect(self.on_cancel)
        self.comboType.currentIndexChanged.connect(self.on_type_change)

    def on_cancel(self):
        self.done(0)

    def on_ok(self):
        self.done(1)

    def on_type_change(self):
        iType = self.comboType.currentIndex()
        s = sensor_libray.get_info(iType)
        self.editAngle.setText(str(s['Angle']))
        self.editRadius.setText(str(s['Radius']))
        self.editX.setText(str(s['Installation_Lat']))
        self.editY.setText(str(s['Installation_Long']))
        self.editHeading.setText(str(s['Orientation']))
        self.editAP.setText(str(s['Accuracy_Location']))
        self.editAV.setText(str(s['Accuracy_Vel']))
        self.editAY.setText(str(s['Accuracy_Yaw']))
        self.editAW.setText(str(s['Accuracy_Width']))
        self.editAL.setText(str(s['Accuracy_Length']))
        self.editAH.setText(str(s['Accuracy_Height']))
        self.editAR.setText(str(s['Accuracy_Radius']))

    def on_add(self):
        s = SensorInfo()
        s.type = self.comboType.currentIndex()
        s.detection_angle = get_float(self.editAngle.text())
        s.detection_range = get_float(self.editRadius.text())
        s.installation_lateral_bias = get_float(self.editX.text())
        s.installation_longitudinal_bias = get_float(self.editY.text())
        s.installation_orientation_angle = get_float(self.editHeading.text())
        s.accuracy_location=get_float(self.editAP.text())
        s.accuracy_velocity=get_float(self.editAV.text())
        s.accuracy_heading=get_float(self.editAY.text())
        s.accuracy_width=get_float(self.editAW.text())
        s.accuracy_length=get_float(self.editAL.text())
        s.accuracy_height = get_float(self.editAH.text())
        s.accuracy_radius = get_float(self.editAR.text())
        self.sensors.append(s)
        self.refresh_table()
        self.refresh_image()

    def on_update(self):
        iRow = self.tableWidget.currentRow()
        if iRow < 0 or iRow >= len(self.sensors):
            return
        s = self.sensors[iRow]
        s.type = self.comboType.currentIndex()
        s.detection_angle=get_float(self.editAngle.text())
        s.detection_range=get_float(self.editRadius.text())
        if s.detection_range > 200.0:  # sumo最大返回自车周围200米内的车辆，因此传感器范围不能超过200m
            s.detection_range = 200.0
            self.editRadius.setText('200.0')
        s.installation_lateral_bias=get_float(self.editX.text())
        s.installation_longitudinal_bias=get_float(self.editY.text())
        s.installation_orientation_angle=get_float(self.editHeading.text())
        s.accuracy_location=get_float(self.editAP.text())
        s.accuracy_velocity=get_float(self.editAV.text())
        s.accuracy_heading=get_float(self.editAY.text())
        s.accuracy_width=get_float(self.editAW.text())
        s.accuracy_length=get_float(self.editAL.text())
        s.accuracy_height = get_float(self.editAH.text())
        s.accuracy_radius = get_float(self.editAR.text())
        # print(s.accuracy_location)

        self.tableWidget.setItem(iRow, 0, QTableWidgetItem(types[int(s.type)]))
        self.tableWidget.setItem(iRow, 1, QTableWidgetItem(str(s.detection_range)))
        self.tableWidget.setItem(iRow, 2, QTableWidgetItem(str(s.detection_angle)))
        self.tableWidget.setItem(iRow, 3, QTableWidgetItem(str(s.installation_orientation_angle)))

        self.refresh_image()

    def on_delete(self):
        iRow=self.tableWidget.currentRow()
        if iRow>=0 and iRow<len(self.sensors):
            self.sensors.remove(self.sensors[iRow])
        # self.refresh_table()
        self.tableWidget.removeRow(iRow)
        self.refresh_image()

    def on_table_clicked(self):
        # self.refresh_table()
        self.refresh_image()
        self.update_form()

    def init_view(self,sensors):
        # print('Input to Sensor Win: '),
        # for i in range(len(sensors)):
        #     print(sensors[i].accuracy_location),
        #     print(' ')
        # print('')
        self.car_image=QImage('Resources/Rendering/Car.png')
        for t in types:
            self.comboType.addItem(self.tr(t))

        self.sensors=[]
        for s in sensors:
            self.sensors.append(copy.copy(s))

        self.refresh_image()
        self.refresh_table()

    def geo2pix(self,geo_pt):
        gx,gy=geo_pt
        w=self.imgLabel.size().width()
        h=self.imgLabel.size().height()
        px=(gx+self.geo_x)/self.geo_width*w
        py=(1-(gy+self.geo_y)/self.geo_length)*h
        return px,py

    def get_car_rect(self):
        car_length = 4.5
        car_center2head = 3.3
        car_width = 1.8
        x1, y1 = -car_width/2, car_center2head
        x2, y2 = +car_width/2, car_center2head - car_length
        return QRectF(QPointF(*self.geo2pix((x1, y1))),
                      QPointF(*self.geo2pix((x2, y2))))

    def refresh_table(self):
        self.tableWidget.clearContents()
        for i in range(len(self.sensors)):
            self.tableWidget.insertRow(i)
            self.tableWidget.setItem(i, 0, QTableWidgetItem(
                types[int(self.sensors[i].type)]))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(
                str(self.sensors[i].detection_range)))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(
                str(self.sensors[i].detection_angle)))
            self.tableWidget.setItem(i, 3, QTableWidgetItem(
                str(self.sensors[i].installation_orientation_angle)))
        self.update_form()

    def refresh_image(self):
        self.get_display_range()
        img = QImage(self.imgLabel.size(), QImage.Format_RGB888)
        p = QPainter(img)
        p.fillRect(img.rect(), QBrush(Qt.lightGray))
        h = self.imgLabel.size().height()

        for i in range(len(self.sensors)):
            self.draw_sector(p, self.sensors[i],
                             i == self.tableWidget.currentRow())
            pass
        p.drawImage(self.get_car_rect(), self.car_image)
        p.end()
        self.imgLabel.setPixmap(QPixmap.fromImage(img))

    def draw_sector(self, p, s, selected=False):
        """Draw each sensor's detection region."""
        if selected:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(0, 255, 0, 100)))
        else:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(255, 255, 255, 50)))
        x0, y0 = s.installation_lateral_bias, s.installation_longitudinal_bias
        x1, y1 = x0 - s.detection_range, y0- s.detection_range
        x2, y2 = x0 + s.detection_range, y0 + s.detection_range
        rect = QRectF(QPointF(*self.geo2pix((x1,y1))),
                      QPointF(*self.geo2pix((x2, y2))))
        p.drawPie(rect, (int(s.installation_orientation_angle-s.detection_angle/2)+90)
                  * 16, int(s.detection_angle)*16)

    def update_form(self):
        iRow = self.tableWidget.currentRow()
        if iRow < 0 or iRow >= len(self.sensors):
            return
        s=self.sensors[iRow]
        self.comboType.setCurrentIndex(int(s.type))
        self.editAngle.setText(str(s.detection_angle))
        self.editRadius.setText(str(s.detection_range))
        self.editX.setText(str(s.installation_lateral_bias))
        self.editY.setText(str(s.installation_longitudinal_bias))
        self.editHeading.setText(str(s.installation_orientation_angle))
        self.editAP.setText(str(s.accuracy_location))
        self.editAV.setText(str(s.accuracy_velocity))
        self.editAY.setText(str(s.accuracy_heading))
        self.editAW.setText(str(s.accuracy_width))
        self.editAL.setText(str(s.accuracy_length))
        self.editAH.setText(str(s.accuracy_height))
        self.editAR.setText(str(s.accuracy_radius))

    def get_display_range(self):
        y1, y2 = -50.0, 50.0
        for s in self.sensors:
            r = s.detection_range
            angle_span = s.detection_angle
            angle = s.installation_orientation_angle
            if abs(angle)<angle_span:
                y2=max(y2,r)
            if abs(angle-180)<angle_span:
                y1=min(y1,-r)
        y1=int(y1)/10*10-10
        y2=int(y2)/10*10+10
        self.geo_length=float(y2-y1)
        self.geo_width=self.geo_length*self.imgLabel.size().width()/self.imgLabel.size().height()
        self.geo_x=self.geo_width/2
        self.geo_y=float(-y1)