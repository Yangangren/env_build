# coding=utf-8
from ctypes import *
import os
EVALUATION_LIB_PATH = 'Modules/EvaluationModel.dll'


class Evaluator(object):
    curve_names = ['Driving Safety', 'Fuel Economy',
                   'Riding Comfort',  'Travel Efficiency']
    curve_names_and_unit = ['Driving Safety', 'Fuel Economy g/s',
                            'Total Acceleration m/$\mathregular{s^2}$',
                            'Speed Ratio']

    def __init__(self):
        module_path = os.path.dirname(__file__)
        print('=======',EVALUATION_LIB_PATH)
        print('=======',module_path)
        self.dll = CDLL(module_path.replace('\\', '/') + '/' + EVALUATION_LIB_PATH)
        self.dll.init()
        self.data = []
        self.released = False

    def __del__(self):
        if not self.released:
            self.dll.release()
            pass

    def release(self):
        try:
            if not self.released:
                self.released=True
                self.dll.release()
        except:
            pass

    def update(self, input, veh_info):
        self.set_data(input, veh_info)
        self.data.append(self.get_data())

    def set_data(self, data, veh_info):
        ((x, y, heading), (plan_x, plan_y), (lane_x, lane_y),
         (front_x, front_y), isInCrossing, isChangeLane,
         delta, alpha, sigma,
         v_lon, v_lat, a,
         d, omega, fuel, a_lon, a_lat,
         front_speed, speed, yaw_rate, steering_wheel,
         engine_speed, engine_torque, gear,
         traffic_density, speed_limit,
         total_fuel_consumption, total_distance_travelled) = data
        heading = -heading + 90
        # print('Lateral acc:  ', a_lat)
        # print('Longitudinal acc:  ', a_lon)
        # print('____')
        # print('Input lon_acc:  ', a_lon)
        # print('Input lat_acc:  ', a_lat)

        self.dll.update(c_float(x), c_float(y),
                        c_float(plan_x), c_float(plan_y),
                        c_float(lane_x), c_float(lane_y),
                        c_float(front_x), c_float(front_y),
                        c_bool(isChangeLane), c_bool(isInCrossing),
                        c_float(delta), c_float(alpha), c_float(sigma),
                        c_float(v_lon), c_float(v_lat), c_float(a), c_float(d),
                        c_float(omega), c_float(a_lon), c_float(a_lat),
                        c_float(speed), c_float(front_speed), c_float(yaw_rate),
                        c_float(steering_wheel), c_float(engine_speed),
                        c_float(engine_torque), c_float(gear), c_float(fuel),
                        c_float(traffic_density), c_float(speed_limit),
                        byref(veh_info), c_float(heading),
                        c_float(total_fuel_consumption),
                        c_float(total_distance_travelled))

    def get_curve_data(self, index):
        if len(self.data) == 0:
            return []
        return zip(*self.data)[index+4]

    def get_curve_name(self, index):
        return self.curve_names[index]

    def get_curve_name_and_unit(self, index):
        return self.curve_names_and_unit[index]

    def get_data(self):
        data = (c_float*12)()  # Safety, Economy, Comfort, Efficency
        self.dll.get_data(byref(data))
        #data[0] = 5.0 - data[0]
        #data[4] = 5.0 - data[4]
        #data[2] = 5.0 - data[2]
        #data[5] = min(100, data[5])
        return list(data)

    def get_report(self):
        """返回评价结果：[Safety, Economy, Comfort, Efficency]"""
        if len(self.data) == 0:
            return [0]*12
        else:
            return self.data[-1]
