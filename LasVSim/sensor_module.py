from ctypes import *
from _ctypes import FreeLibrary
# from SensorSetting import *
# from TrafficModule import self.other_car_num
from math import fabs
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
import time
import winsound
#from LasVSim.AOE_estimator_for_LasVsim.main import AOE
SENSORS_MODEL_PATH = 'Modules/Sensors.dll'

class SensorInfo(Structure):
    """Sensor structure for C/C++ interface
    """
    _fields_ = [("id", c_int),
                ("type", c_int),
                ("installation_lateral_bias", c_float),
                ("installation_longitudinal_bias", c_float),
                ("installation_orientation_angle", c_float),
                ("detection_angle",c_float),
                ("detection_range", c_float),
                ("accuracy_location", c_float),
                ("accuracy_heading", c_float),
                ("accuracy_velocity", c_float),
                ("accuracy_width", c_float),
                ("accuracy_length", c_float),
                ("accuracy_height", c_float),
                ("accuracy_radius", c_float)]


class NoiseInfo(Structure): #for different type
    """NoiseInfo structure for C/C++ interface
        """
    _fields_ =[
        ("car_location", c_float),
        ("car_heading", c_float),
        ("car_velocity", c_float),
        ("car_size", c_float),
        ("truck_location", c_float),
        ("truck_heading", c_float),
        ("truck_velocity", c_float),
        ("truck_size", c_float),
        ("motor_location", c_float),
        ("motor_heading", c_float),
        ("motor_velocity", c_float),
        ("motor_size", c_float),
        ("bike_location", c_float),
        ("bike_heading", c_float),
        ("bike_velocity", c_float),
        ("bike_size", c_float),
        ("ped_location", c_float),
        ("ped_heading", c_float),
        ("ped_velocity", c_float),
        ("ped_size", c_float)
    ]
class ObjectInfo(Structure):
    """Other car Struct for C/C++ interface.
    """
    _fields_ = [("object_id_real", c_int),
                ("object_type_real", c_int),
                ("object_x_real", c_float),
                ("object_y_real", c_float),
                ("object_heading_real", c_float),
                ("object_width_real", c_float),
                ("object_length_real", c_float),
                ("object_height_real", c_float),
                ("object_radius_real", c_float),
                ("object_velocity_real", c_float),
                ("object_vx_real", c_float),
                ("object_vy_real", c_float)
                ]
class ObjectInfoFromSensor(Structure):
    """Detected car struct for C/C++ interface.
    """
    _fields_ = [("object_id_detected", c_int),
                ("object_type_detected", c_int),
                ("object_x_detected", c_float),
                ("object_y_detected", c_float),
                ("object_heading_detected", c_float),
                ("object_yaw_rate_detected", c_float),
                ("object_width_detected", c_float),
                ("object_length_detected", c_float),
                ("object_height_detected", c_float),
                ("object_radius_detected", c_float),
                ("object_velocity_detected", c_float),
                ("object_vx_detected", c_float),
                ("object_vy_detected", c_float),
                ("IsMan", c_int)
                ]
class TrafficSignInfo(Structure):
    _fields_ = [
        # static
        ("TS_type", c_int),
        ("TS_length", c_float),
        ("TS_height", c_float),
        ("TS_value", c_int),
        # dynamic
        ("TS_x", c_float),
        ("TS_y", c_float),
        ("TS_z", c_float),
        ("TS_heading", c_float),
        ("TS_range", c_float)]
class TrafficSignInfoFromSensor(Structure):
    _fields_ = [
        # static
        ("TS_type_detected", c_int),
        ("TS_length_detected", c_float),
        ("TS_height_detected", c_float),
        ("TS_value_detected", c_int),
        # dynamic
        ("TS_x_detected", c_float),
        ("TS_y_detected", c_float),
        ("TS_z_detected", c_float),
        ("TS_heading_detected", c_float),
        ("TS_range_detected", c_float)]
class ManInfo(Structure):
    _fields_=[
        #static
        ("man_length", c_float),
        ("man_width", c_float),
        #dynamic
        ("man_x", c_float),
        ("man_y", c_float),
        ("man_vx",c_float),
        ("man_vy", c_float),
        ("man_heading", c_float)
    ]
class ManInfoFromSensor(Structure):
    _fields_=[
        ("id_detected",c_int),
        #static
        ("length_detected", c_float),
        ("width_detected", c_float),
        #dynamic
        ("x_detected", c_float),
        ("y_detected", c_float),
        ("vx_detected",c_float),
        ("vy_detected", c_float),
        ("heading_detected", c_float)
    ]
class DetectionError(Structure):
    _fields_ = [
        ("id", c_int),
        ("sensor_id", c_int),
        ("object_type", c_int),
        ("Ex", c_float),
        ("Ey", c_float),
        ("Etheta", c_float),
        ("Ew", c_float),
        ("El", c_float),
        ("Evx", c_float),
        ("Evy", c_float)]

class theta_record:
    def __init__(self):
        self.round=0
        self.previous_theta=0
        self.threshold=150
        self.initflag=0
    def update(self,new_theta):
        if (self.initflag==0):
            self.previous_theta=new_theta
            self.initflag=1
        if (abs(new_theta-self.previous_theta)>self.threshold):
            if self.previous_theta<=0:
                self.round=self.round-1
            else:
                self.round=self.round+1
        self.previous_theta=new_theta
        return new_theta+360*self.round

class Detect_Result(Structure):
    _fields_ =[
        ("detected_car_num",c_int),
        ("detected_man_num", c_int),
        ("detected_TS_num", c_int),
        ("measurement_num", c_int)]

class Outer_Noise_setting(Structure):
    _fields_ = [
        ("lidar", c_int),
        ("camera", c_int),
        ]
class Sensors(object):
    """
    Sensor Modal for Autonomous Car Simulation System
    a interface with ctypes for the dynamic library
    """
    def __init__(self,step_length, sensor_info=None, path=None):
        self.__lasvsim_version = 'package'
        self.__step_length = step_length
        module_path = os.path.dirname(__file__)
        self.dll = CDLL(module_path.replace('\\', '/') + '/' + SENSORS_MODEL_PATH)
        self.sensor_info = sensor_info
        self.outer_noise_setting=Outer_Noise_setting()
        if self.__lasvsim_version == 'package':
            self.setSensor(self.sensor_info)
        from LasVSim.traffic_module import MAX_TRAFFIC
        self.other_car_num = MAX_TRAFFIC
        self.traffic_sign_number = 0  # maximum traffic sign number
        self.man_number=0 #maximum pedestrian number
        self.other_car_arr = ObjectInfo * self.other_car_num
        self.detect_car_arr = ObjectInfoFromSensor * self.other_car_num
        self.traffic_sign_arr = TrafficSignInfo * self.traffic_sign_number
        self.detection_err_arr= DetectionError * self.other_car_num
        self.detect_traffic_sign_arr=TrafficSignInfoFromSensor * self.traffic_sign_number
        self.detect_man_arr=ManInfoFromSensor*self.man_number
        self.detected_object_num=0
        self.detect_man_num=0
        self.measurement_num=0

        #建立对象
        self.other_cars = self.other_car_arr()
        self.detection_err=self.detection_err_arr()
        self.detect_cars = self.detect_car_arr()
        self.detect_cars_GT=self.detect_car_arr()
        self.traffic_sign = self.traffic_sign_arr()
        self.detect_traffic_sign=self.detect_traffic_sign_arr()
        self.detect_man=self.detect_man_arr()
        self.detect_man_GT=self.detect_man_arr()
        self.detect_result =Detect_Result()
        self.theta_records = [theta_record() for i in range(self.other_car_num)]

        for i in range(len(self.other_cars)):  # initialize other_cars position
            self.other_cars[i].object_x_real = 99999
            self.other_cars[i].object_y_real = 99999
        ##这边可以建立KF


    def __del__(self):
        # FreeLibrary(self.dll._handle)
        del self.dll

    def setVehicleModel(self, vehicle_model):
        self.vehicle_model = vehicle_model

    def setSensor(self, sensors):
        self.sensors = sensors
        # for i in range(len(self.sensors)):#ADD BY TKM 0730
        #     self.sensors[i].id=i
        #a,b,c=input("input R,phi,R rate std:")
        winsound.Beep(600, 1000)
        # d,e=input("input acceleration, angle acceleration")
        noise_arr = NoiseInfo * len(sensors)
        self.noise=noise_arr()
        noise_fields = [
            "car_location",
            "car_heading",
            "car_velocity",
            "car_size",
            "truck_location",
            "truck_heading",
            "truck_velocity",
            "truck_size",
            "motor_location",
            "motor_heading",
            "motor_velocity",
            "motor_size",
            "bike_location",
            "bike_heading",
            "bike_velocity",
            "bike_size",
            "ped_location",
            "ped_heading",
            "ped_velocity",
            "ped_size"]
        noise_attr=["Accuracy_Location","Accuracy_Yaw","Accuracy_Vel","Accuracy_Width"]
        #"Accuracy_Length", "Accuracy_Height" currently no use
        lidar=pd.read_csv(os.getcwd()+'\LasVSim\Library\lidar_error_for_different_object_type.csv')
        camera=pd.read_csv(os.getcwd()+'\LasVSim\Library\camera_error_for_different_object_type.csv')
        tmp=0 # int(input("lidar use other setting (yes:1) (no:0)?"))
        if (tmp==1):
            lidar_use_outer_setting=True
        else:
            lidar_use_outer_setting = False
        tmp=0 # int(input("camera use other setting (yes:1) (no:0)?"))
        if (tmp==1):
            camera_use_outer_setting=True
        else:
            camera_use_outer_setting = False

        self.outer_noise_setting.lidar = 0
        self.outer_noise_setting.camera = 0
        #build noise
        for i in range(len(sensors)):
            cnt=0
            if ((3<sensors[i].type and sensors[i].type<9) and lidar_use_outer_setting):

                #lidar
                self.outer_noise_setting.lidar=1
                print("lidar yes",self.outer_noise_setting.lidar)
                for j in range(len(noise_fields)):
                    exec('self.noise[i].%s=%f'%(noise_fields[j],lidar[noise_attr[cnt%4]][np.floor(cnt/4)]))
                    cnt+=1
                    #print(cnt)
            elif (sensors[i].type==9 and camera_use_outer_setting):
                print("camera yes")
                #camera
                for j in range(len(noise_fields)):
                    self.outer_noise_setting.camera = 1
                    exec ('self.noise[i].%s=%f' % (noise_fields[j], camera[noise_attr[cnt % 4]][np.floor(cnt / 4)]))
                    cnt += 1
                    #print(cnt)
            else:
                #others
                for j in range(len(noise_fields)):
                    exec ('self.noise[i].%s=9999' %noise_fields[j])
        #debug
        # for i in range(len(noise_fields)):
        #     exec('print(self.noise[1].%s)'%(noise_fields[i]))
        fusion_mode=1 # int(input("input fusion mode:"))
        output_mode=0 # int(input("output mode:"))
        if fusion_mode==2:
            Q_acc=0      #int(input("input acceleration variance"))
            Q_yawrate=0  # int(input("input angular acceleration variance"))
        else:
            Q_acc =500
            Q_yawrate=800
        self.dll.SetSensors(len(sensors), byref(sensors),byref(self.outer_noise_setting),byref(self.noise),c_int(output_mode),c_int(fusion_mode),c_float(Q_acc),c_float(Q_yawrate))
        #urban c_float(500), c_float(800)
        # Display mode:0:none,1:brief,2:detail,3:all data;
        # Fusion on:0: Ground Truth, 1: Object Detection, 2: Object Tracking
        print('\nsensor configured')

    def update(self, pos, vehicles):
        t1 = datetime.datetime.now().microsecond
        t3 = time.mktime(datetime.datetime.now().timetuple())
        veh_type = {'car_1': 3, 'car_2': 3, 'car_3': 3, 'car_4':3, 'car_5':3,'truck_1': 100,
                         'motorbike_1': 200, 'motorbike_2': 200, 'motorbike_3': 200}
        x, y, velocity, a = pos
        self.veh_route = []
        i = 0
        for veh in vehicles:
            self.other_cars[i].object_id_real = i
            self.other_cars[i].object_type_real = veh_type[veh['type']]
            self.other_cars[i].object_x_real = veh['x']
            self.other_cars[i].object_y_real = veh['y']
            self.other_cars[i].object_heading_real = veh['phi']
            self.other_cars[i].object_width_real = veh['w']
            self.other_cars[i].object_length_real = veh['l']
            self.other_cars[i].object_height_real = 1.5
            self.other_cars[i].object_radius_real = 3.0
            self.other_cars[i].object_velocity_real = veh['v']
            #temperary solution
            self.other_cars[i].object_vx_real=veh['v']*math.cos(veh['phi']*math.pi/180)
            self.other_cars[i].object_vy_real =veh['v']*math.sin(veh['phi']*math.pi/180)
            self.veh_route.append(veh['route'])
            i = i+1

        nearby_TS_number=0
        traffic_sign = []
        for i in range(len(traffic_sign)):
            if (np.sqrt((traffic_sign[i].TS_x-x)**2+(traffic_sign[i].TS_y-y)**2)<100):
                self.traffic_sign[nearby_TS_number].TS_type = traffic_sign[i].TS_type
                self.traffic_sign[nearby_TS_number].TS_length = traffic_sign[i].TS_length
                self.traffic_sign[nearby_TS_number].TS_height = traffic_sign[i].TS_height
                self.traffic_sign[nearby_TS_number].TS_value = traffic_sign[i].TS_value
                self.traffic_sign[nearby_TS_number].TS_x = traffic_sign[i].TS_x
                self.traffic_sign[nearby_TS_number].TS_y = traffic_sign[i].TS_y
                self.traffic_sign[nearby_TS_number].TS_z = traffic_sign[i].TS_z
                self.traffic_sign[nearby_TS_number].TS_heading = traffic_sign[i].TS_heading
                self.traffic_sign[nearby_TS_number].TS_range = traffic_sign[i].TS_range
                nearby_TS_number+=1

        self.dll.Detect.argtypes = (c_float, c_float, c_float, c_float, c_int,c_int,
                                    POINTER(self.other_car_arr),POINTER(self.traffic_sign_arr),
                                    POINTER(self.detection_err_arr),
                                    POINTER(self.detect_car_arr),POINTER(self.detect_car_arr),
                                    POINTER(self.detect_man_arr),POINTER(self.detect_man_arr),
                                    POINTER(self.detect_traffic_sign_arr)
                                    ,POINTER(Detect_Result))
        self.dll.Detect.restype = c_int
        self.dll.Detect(c_float(x), c_float(y), c_float(a), c_float(velocity),  # add velocity by TKM
                        c_int(len(vehicles)), c_int(nearby_TS_number),
                        pointer(self.other_cars),
                        pointer(self.traffic_sign),
                        pointer(self.detection_err),
                        pointer(self.detect_cars),
                        pointer(self.detect_cars_GT),
                        pointer(self.detect_man),
                        pointer(self.detect_man_GT),
                        pointer(self.detect_traffic_sign),
                        pointer(self.detect_result))
        self.detected_object_num=self.detect_result.detected_car_num
        #print("object number", self.detected_object_num)
        #print("man num in C++:",self.detect_result.detected_man_num)
        self.detect_man_num = self.detect_result.detected_man_num
        #print("man num in python:", self.detect_man_num)
        self.detected_TS_num=self.detect_result.detected_TS_num
        self.measurement_num=self.detect_result.measurement_num

        for i in range(len(vehicles)):  # reset other_cars' position
            self.other_cars[i].object_x_real = 99999
            self.other_cars[i].object_y_real = 99999
        for i in range(nearby_TS_number):
            self.traffic_sign[i].TS_x = 99999
            self.traffic_sign[i].TS_y = 99999
        t2 = datetime.datetime.now().microsecond
        t4 = time.mktime(datetime.datetime.now().timetuple())
        strTime = 'time use:%dms' % ((t4 - t3) * 1000 + (t2 - t1) / 1000)
        Time=np.array([((t4 - t3) * 1000 + (t2 - t1) / 1000)])

        return self.detect_cars_GT,self.detect_man_GT
    def getVisibleVehicles(self):
        visible_vehicles = []
        visible_man=[]
        visible_traffic_sign=[]
        for i in range(self.detected_object_num):
            vehicle = self.detect_cars[i]
            id = vehicle.object_id_detected
            type = vehicle.object_type_detected
            x = vehicle.object_x_detected
            y = vehicle.object_y_detected
            a = vehicle.object_heading_detected
            w = vehicle.object_width_detected
            l = vehicle.object_length_detected
            height = vehicle.object_height_detected
            radius = vehicle.object_radius_detected
            v = vehicle.object_velocity_detected
            route = self.veh_route[id]
            # if width < 1:
            #     width = 1.0
            # if length < 1:
            #     length = 1.0

            visible_vehicles.append(dict(type=id,
                                            x=x,
                                            y=y,
                                            phi=a,
                                            v=v,
                                            l=l,
                                            w=w,
                                            route=route))
        for i in range(self.detect_man_num):
            man = self.detect_man[i]
            ##print("man:",man)
            id = man.id_detected
            x = man.x_detected
            y = man.y_detected
            vx=man.vx_detected
            vy=man.vy_detected
            v=np.sqrt(vx**2+vy**2)
            heading = man.heading_detected
            width = man.width_detected
            length = man.length_detected
            visible_man.append((id,x,y,v,heading,width,length))
        for i in range(self.detected_TS_num):
            TS=self.detect_traffic_sign[i]
            type=TS.TS_type_detected
            value=TS.TS_value_detected
            length=TS.TS_length_detected
            height=TS.TS_height_detected
            width=0.1
            x=TS.TS_x_detected
            y=TS.TS_y_detected
            z=TS.TS_z_detected
            heading=TS.TS_heading_detected
            Range=TS.TS_range_detected
            visible_traffic_sign.append(( x, y, heading, width, length))
        #print(visible_man)
        return visible_vehicles

    def getFusionResult(self):
        vehicles_fusion_result={}
        man_fusion_result={}
        #print("object number in fusion result",self.detected_object_num)
        for i in range(self.detected_object_num):
            vehicle = self.detect_cars[i]
            id = vehicle.object_id_detected
            #print(id)
            type = vehicle.object_type_detected
            x = vehicle.object_x_detected
            y = vehicle.object_y_detected
            heading = self.theta_records[id].update(vehicle.object_heading_detected)
            width = vehicle.object_width_detected
            length = vehicle.object_length_detected
            v = vehicle.object_velocity_detected
            vx=vehicle.object_vx_detected
            vy = vehicle.object_vy_detected
            yawrate=vehicle.object_yaw_rate_detected
            #真值多了个type，非行人类多了omega
            vehicles_fusion_result[str(id)]=(type,x, y, heading, width, length,vx,vy,yawrate)
        ##print("detected_man:",self.detect_man_num)
        for i in range(self.detect_man_num):
            man = self.detect_man[i]
            ##print("man:",man)
            id = man.id_detected
            type=300
            x = man.x_detected
            y = man.y_detected
            vx=man.vx_detected
            vy=man.vy_detected
            v=np.sqrt(vx**2+vy**2)
            heading = self.theta_records[id].update(man.heading_detected)
            width = man.width_detected
            length = man.length_detected
            man_fusion_result[str(id)]=(type,x,y,heading,width,length,vx,vy)
        return vehicles_fusion_result, man_fusion_result
    def getVisibleVehiclesGT(self):
        visible_vehiclesGT = {}
        visible_manGT={}
        detection_keys=[]
        for i in range(self.detected_object_num):
            vehicle = self.detect_cars_GT[i]
            id = vehicle.object_id_detected
            type = vehicle.object_type_detected
            x = vehicle.object_x_detected
            y = vehicle.object_y_detected
            #heading=vehicle.object_heading_detected
            heading = self.theta_records[id].update(vehicle.object_heading_detected)
            width = vehicle.object_width_detected
            length = vehicle.object_length_detected
            v = vehicle.object_velocity_detected
            vx=vehicle.object_vx_detected
            vy = vehicle.object_vy_detected
            #真值多了个type
            visible_vehiclesGT[str(id)]=(type,x, y, heading, width, length,vx,vy)
            detection_keys.append(str(id))
        ##print("detected_man:",self.detect_man_num)
        for i in range(self.detect_man_num):
            man = self.detect_man_GT[i]
            ##print("man:",man)
            id = man.id_detected
            type=300
            x = man.x_detected
            y = man.y_detected
            vx=man.vx_detected
            vy=man.vy_detected
            v=np.sqrt(vx**2+vy**2)
            heading = self.theta_records[id].update(man.heading_detected)
            width = man.width_detected
            length = man.length_detected
            visible_manGT[str(id)]=(type,x,y,heading,width,length,vx,vy)
            detection_keys.append(str(id))
        return visible_vehiclesGT,visible_manGT,detection_keys

    def getDetectPair(self):
        detect_pair = []
        #print("z_num",self.measurement_num)
        for i in range(self.measurement_num):
            tmp = self.detection_err[i]
            detect_pair.append((tmp.id, tmp.sensor_id,tmp.object_type,tmp.Ex,tmp.Ey,tmp.Etheta,tmp.Ew,tmp.El,tmp.Evx,tmp.Evy))
        return detect_pair
if __name__ == "__main__":
    #build testing samples
    plt.ion()
    VEHICLE_COUNT = 1001
    sensor_N=3
    sensor_info = (SensorInfo * sensor_N)()
    for i in range(sensor_N):
        if i == 0:
            sensor_info[0].id = c_int(0)
            sensor_info[0].type = c_int(9)
            sensor_info[0].installation_lateral_bias = c_float(0.0)
            sensor_info[0].installation_longitudinal_bias = c_float(1.65) #car length 3.3
            sensor_info[0].installation_orientation_angle = c_float(0.0)
            sensor_info[0].detection_angle = c_float(360.0)
            sensor_info[0].detection_range = c_float(60.0)
            sensor_info[0].accuracy_location = c_float(0.7)
            sensor_info[0].accuracy_heading = c_float(8.0)
            sensor_info[0].accuracy_velocity = c_float(1.4)
            sensor_info[0].accuracy_width = c_float(0.2)
            sensor_info[0].accuracy_length = c_float(0.3)
            sensor_info[0].accuracy_height = c_float(0.15)
            sensor_info[0].accuracy_radius = c_float(0.2)
        elif i == 1:
            sensor_info[1].id = c_int(1)
            sensor_info[1].type = c_int(9)
            sensor_info[1].installation_lateral_bias = c_float(0.0)
            sensor_info[1].installation_longitudinal_bias = c_float(0.0)
            sensor_info[1].installation_orientation_angle = c_float(0.0)
            sensor_info[1].detection_angle = c_float(360.0)
            sensor_info[1].detection_range = c_float(50.0)
            sensor_info[1].accuracy_location = c_float(0.5)
            sensor_info[1].accuracy_heading = c_float(2.0)
            sensor_info[1].accuracy_velocity = c_float(1.0)
            sensor_info[1].accuracy_width = c_float(0.1)
            sensor_info[1].accuracy_length = c_float(0.1)
            sensor_info[1].accuracy_height = c_float(0.1)
            sensor_info[1].accuracy_radius = c_float(0.1)
        elif i == 2:
            sensor_info[2].id = c_int(2)
            sensor_info[2].type = c_int(4)
            sensor_info[2].installation_lateral_bias = c_float(0.0)
            sensor_info[2].installation_longitudinal_bias = c_float(1.65)
            sensor_info[2].installation_orientation_angle = c_float(0.0)
            sensor_info[2].detection_angle = c_float(38.0)
            sensor_info[2].detection_range = c_float(80.0)
            sensor_info[2].accuracy_location = c_float(1.0)
            sensor_info[2].accuracy_heading = c_float(4.0)
            sensor_info[2].accuracy_velocity = c_float(3.0)
            sensor_info[2].accuracy_width = c_float(0.1)
            sensor_info[2].accuracy_length = c_float(0.14)
            sensor_info[2].accuracy_height = c_float(0.1)
            sensor_info[2].accuracy_radius = c_float(0.1)
    # ego_pos = [0.0, 0.0, 0.0, 90-60.0]
    #x,y,v,a
    ego_pos = [0.0, 0.0, 0.0,-45.0]

    traffic_sign = (TrafficSignInfo * 10)()
    for i in range(10):
        traffic_sign[i].TS_type = i
        traffic_sign[i].TS_z = i + 1
    # traffic_sign=[{'TS_type':0,'TS_length':1,'TS_height':2,'TS_value':3,'TS_x':4,'TS_y':5,'TS_z':6,
    #                'TS_heading':7,'TS_range':8}]

    cars = [{'type': 0, 'id':1,'x': -21.7254, 'y': 10.0620, 'angle': -45.0, 'v': 10.0,'render': True},
            {'type': 1, 'id':2,'x': 20, 'y': -20, 'angle': 70.0, 'v': 20.0, 'render': True},
            {'type': 2, 'id':3,'x': 10, 'y': -10, 'angle': -90.0, 'v': 30.0,'render': True},
            #{'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0,'render': True},
            #{'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0,'render': True},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0},
            # {'type': 0, 'x': -21.7254, 'y': 10.0620, 'angle': -90.0, 'v': 0.0},
            # {'type': 1, 'x': 59.8791, 'y': 155.7381, 'angle': -90.0, 'v': 0.0},
            # {'type': 2, 'x': 85.1766, 'y': 122.6166, 'angle': -90.0, 'v': 0.0},
            # {'type': 3, 'x': 23.4503, 'y': 63.9505, 'angle': -113.0, 'v': 0.0},
            # {'type': 4, 'x': 35.8858, 'y': 62.7002, 'angle': 0.0, 'v': 0.0}
            ]


    class VehicleModels(object):
        """Vehicle model class.

            Read vehicle model information file and load these information into
            simulation.

            Attributes:
                __info: A dict containing all vehicle models' information.
                __type_array: A list containing vehicle type's id.
        """
        __type_array = [0, 1, 2, 3, 4, 100, 1000]

        def __init__(self, model_path):
            self.__info = dict()
            with open(model_path) as f:
                line = f.readline()
                line = f.readline()
                while len(line) > 0:
                    data = line.split(',')
                    type = int(data[1])
                    if type not in self.__type_array:
                        line = f.readline()
                        continue
                    h = float(data[7])
                    w = float(data[8])
                    img_path = 'Resources/Rendering/%d.png' % type
                    x = float(data[4])
                    y = float(data[2])
                    self.__info[type] = (h, w, x, y, img_path)
                    line = f.readline()

        def get_types(self):
            return self.__type_array

        def get_vehicle(self, type):
            if not self.__info.__contains__(type):
                type = 0
            return self.__info[type]


    vehicle_models = VehicleModels('Library/vehicle_model_library.csv')
    sensors = Sensors(0.05)
    sensors.setVehicleModel(vehicle_models)
    sensors.setSensor(sensors=sensor_info)
    recur_times=20

    # create blank picture
    fig, ax = plt.subplots()


    for i in range(recur_times):
        #constant velocity model
        for j in range(len(cars)):
            cars[j]['x'] =cars[j]['x']+cars[j]['v'] * math.cos(cars[j]['angle'] / 180 * np.pi)*0.05
            cars[j]['y'] =cars[j]['y']+cars[j]['v'] * math.sin(cars[j]['angle'] / 180 * np.pi)*0.05
        sensors.update(pos=ego_pos, vehicles=cars, traffic_sign=traffic_sign)
        visible_vehicles,visible_man,visible_traffic_sign=sensors.getVisibleVehicles()
        visible_vehiclesGT,visible_manGT=sensors.getVisibleVehiclesGT()

        for vehicle in visible_vehiclesGT:
            print(vehicle)
        for man in visible_manGT:
            print(man)
        #vehicle (id, x, y, v, heading, width, length)
        #man (id,x,y,v,heading,width,length)
        #traffic ( x, y, heading, width, length)
        """draw ego car"""
        #plot ego car
        if 1:
            a, l, w = ego_pos[3], 4.8, 2
            rad = a * math.pi / 180
            half_diag_len = math.sqrt(l ** 2 + w ** 2) / 2
            init_rad = math.atan( w/ l)
            x = ego_pos[0] -half_diag_len * math.cos(init_rad + rad)
            y = ego_pos[1] -half_diag_len * math.sin(init_rad + rad)
            #print("x",x) #x of rect
            #print("y",y) #y of rect
            rect = mpatches.Rectangle((x, y),
                                      l,w, angle=a, linewidth=1, edgecolor='r', facecolor='none')
            #Add the patch to the Axes
            ax.add_patch(rect)
        #plot sensors
        for i in range(sensor_N):
            install_l = math.sqrt(
                sensor_info[i].installation_longitudinal_bias ** 2 + sensor_info[i].installation_lateral_bias ** 2)
            aaa = math.atan2(sensor_info[i].installation_lateral_bias, sensor_info[i].installation_longitudinal_bias)
            # Wedge(center, r, theta1, theta2, width=None, **kwargs)[source]
            wedge = mpatches.Wedge(
                ((ego_pos[0]+ install_l * math.cos(ego_pos[3] * math.pi / 180 + aaa)),
                 (ego_pos[1] + install_l * math.sin(ego_pos[3] * math.pi / 180 + aaa))),
                sensor_info[i].detection_range,
                ego_pos[3]+ sensor_info[i].installation_orientation_angle - 0.5 * sensor_info[i].detection_angle,
                ego_pos[3]+ sensor_info[i].installation_orientation_angle + 0.5 * sensor_info[i].detection_angle,
                edgecolor='b', facecolor='none')
            # wedge = mpatches.Wedge((install_l*math.cos(ego_pos[3]*math.pi/180),install_l*math.sin(ego_pos[3]*math.pi/180)),
            #                        sensor_info[i].detection_range, ego_pos[3]+sensor_info[i].installation_orientation_angle-0.5*sensor_info[i].detection_angle,
            #                        ego_pos[3]+sensor_info[i].installation_orientation_angle+0.5*sensor_info[i].detection_angle, edgecolor='b', facecolor='none')
            ax.add_patch(wedge)
        #plt.plot(0,0, 'x')

        #plot object
        if 1:
            #print(len(cars))
            for j in range(len(cars)):
                w=sensors.other_cars[j].object_width_real
                l = sensors.other_cars[j].object_length_real
                a = sensors.other_cars[j].object_heading_real
                deg = a
                rad = deg * math.pi / 180 # deg2rad
                half_diag_len = math.sqrt(l ** 2 + w ** 2) / 2  #half diagnal length
                init_rad = math.atan(w/ l)  #initial diagnal's deg
                x = cars[j]['x'] - half_diag_len * math.cos(init_rad + rad) #new x coordinate(LB side of rect),deg=(init diagnal's deg+yaw-90)
                y = cars[j]['y'] - half_diag_len * math.sin(init_rad + rad)  #new y coordinate(RU side of rect)
                plt.plot(cars[j]['x'] ,cars[j]['y'], 'x')
                # print("x", vehicles[i].veh_x)
                # print("y", vehicles[i].veh_y)
                # print("heading",vehicles[i].veh_heading)
                # print("diag_deg",(init_rad + rad)*180/math.pi)
                rect = mpatches.Rectangle((x, y),
                                          l,w, angle=deg, linewidth=1, edgecolor='g', facecolor='g')  #build rect
                # Add the patch to the Axes
                ax.add_patch(rect)  #plot object
        #plot bounding box
        if sensors.detected_object_num != 0:
            #print("sensor_detect_num: ",sensors.detected_object_num)
            for i in range(sensors.detected_object_num):
                w = visible_vehicles[i][5]
                #print("debug:(w): ",w)
                l = visible_vehicles[i][6]
                a = visible_vehicles[i][4]
                deg = a
                rad = deg * math.pi / 180 # deg2rad
                half_diag_len = math.sqrt(l ** 2 + w ** 2) / 2  #half diagnal length
                init_rad = math.atan(w / l)  #initial diagnal's deg
                #print("debug:(x): ", visible_vehicles[i][1] )
                #print("debug:(y): ", visible_vehicles[i][2])
                x = visible_vehicles[i][1] - half_diag_len * math.cos(init_rad + rad)  #new x coordinate(LB side of rect),deg=(init diagnal's deg+yaw-90)
                y = visible_vehicles[i][2] - half_diag_len * math.sin(init_rad + rad)  #new y coordinate(RU side of rect)
                rect = mpatches.Rectangle((x, y),
                                          l,w, angle=deg, linewidth=1.2, edgecolor='r', facecolor='none')  #build rect
                # Add the patch to the Axes
                ax.add_patch(rect)  #plot
        ax.set_xlabel("XG")
        ax.set_ylabel("YG")
        ax.set_xlim(-80, 80)
        ax.set_ylim(-80, 80)
        ax.set_aspect('equal')  # x,y axixs are equal
        plt.pause(1)
        plt.cla()

        # if 1:
        #     for i in range(sensors.detected_object_num):
        #         w=cars[sensor]
        #         w=visible_vehicles[i][5]
        #         l = visible_vehicles[i][6]
        #         a = visible_vehicles[i][4]
        #         deg = a - 90  #minus 90 deg for correction
        #         rad = deg * math.pi / 180 # deg2rad
        #         half_diag_len = math.sqrt(l ** 2 + w ** 2) / 2  #half diagnal length
        #         init_rad = math.atan(l / w)  #initial diagnal's deg
        #         x = visible_vehicles[i][1] - half_diag_len * math.cos(init_rad + rad)  #new x coordinate(LB side of rect),deg=(init diagnal's deg+yaw-90)
        #         y = visible_vehicles[i][2] - half_diag_len * math.sin(init_rad + rad)  #new y coordinate(RU side of rect)
        #
        #         # print("x", vehicles[i].veh_x)
        #         # print("y", vehicles[i].veh_y)
        #         # print("heading",vehicles[i].veh_heading)
        #         # print("diag_deg",(init_rad + rad)*180/math.pi)
        #         rect = mpatches.Rectangle((x, y),
        #                                   w, l, angle=deg, linewidth=1, edgecolor='g', facecolor='g')  #build rect
        #         # Add the patch to the Axes
        #         ax.add_patch(rect)  #plot object






