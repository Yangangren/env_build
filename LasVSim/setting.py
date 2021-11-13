from LasVSim.data_structures import *

"""controller and vehicle dynamic modules Setting Value"""
current_path = os.path.dirname(__file__)
plan_horizon=30
history_len = 1
step_length = 50
setting_path = 'C:/Users/DuanJingliang/Desktop/PI_DSAC/PI_dsac_20200218/LasVSim/Scenario/Ramp2Highway_endtoend/'
# setting_path = './LasVSim/Scenario/Ramp2Highway_endtoend/+str(i)'
# setting_path = './LasVSim/Scenario/Highway_endtoend/'
CONTROLLER_MODEL_PATH='C:/Users/DuanJingliang/Desktop/PI_DSAC/PI_dsac_20200218/LasVSim/Modules/Controller.dll'
CVT_MODEL_FILE_PATH = "C:/Users/DuanJingliang/Desktop/PI_DSAC/PI_dsac_20200218/LasVSim/Modules/CarModel_CVT.dll"
AMT_MODEL_FILE_PATH = "C:/Users/DuanJingliang/Desktop/PI_DSAC/PI_dsac_20200218/LasVSim/Modules/CarModel_AMT.dll"
TRUCK_MODEL_FILE_PATH = "C:/Users/DuanJingliang/Desktop/PI_DSAC/PI_dsac_20200218/LasVSim/Modules/CarModel_Truck.dll"
ROUTER_MAX_ERR = 1e-6

"""Simulation Setting Value"""

DECISION_OUTPUT_TYPE = ['spatio-temporal trajectory',
                        'acceleration(m/s^2), front wheel angle(deg)',
                        'eng_torque(N*m), steering(deg), brake(Mpa)']  # 不能更改顺序
DYNAMIC_OUTPUT = 2
KINEMATIC_OUTPUT = 1
SPATIO_TEMPORAL_TRAJECTORY = 0
DEFAULT_SETTING_FILE = './LasVSim/Scenario/Ramp2Highway_endtoend'
TRAFFIC_TYPE = ['No Traffic', 'Mixed Traffic', 'Vehicle Only Traffic'] # 不能更改顺序
NO_TRAFFIC = 0
MIXED_TRAFFIC = 1
VEHICLE_ONLY_TRAFFIC = 2
TRAFFIC_DENSITY = ['Sparse', 'Middle', 'Dense'] # 不能更改顺序
SPARSE = 0
MIDDLE = 1
DENSE = 2
HISTORY_TRAFFIC_SETTING = ['Traffic Type', 'Traffic Density',
                           'Map']  # 上次仿真的交通流配置
RANDOM_TRAFFIC = {}  # 随机交通流分布信息

FILE_TYPE = ['C/C++ DLL', 'Python Module'] # 不能更改顺序

MAPS = ['Map1_Urban Road', 'Map2_Highway', 'Map3_Shanghai Anting',
        'Map4_Beijing Changping', 'Map5_Mcity', 'Map3_Highway_v2']  # 不能更改顺序


"""scenario test Setting Value"""
model_type = 'CVT Car'
carparameter=CarParameter()
carparameter.LX_AXLE=2.33 # 轴距，m
carparameter.LX_CG_SU=1.1165 # 悬上质量质心至前轴距离，m
carparameter.M_SU=236 # 悬上质量，kg
carparameter.IZZ_SU=1536 # 转动惯量，kg*m^2
carparameter.A=2.22 # 迎风面积，m^2
carparameter.CFx=0.3 # 空气动力学侧偏角为零度时的纵向空气阻力系数
carparameter.AV_ENGINE_IDLE=750 # 怠速转速，rpm
carparameter.IENG=0.21 # 曲轴转动惯量，kg*m^2
carparameter.TAU=0.2 # 发动机-变速箱输入轴 时间常数，s
carparameter.R_GEAR_TR1=2.25 # 最低档变速箱传动比
carparameter.R_GEAR_FD=2.86# 主减速器传动比
carparameter.BRAK_COEF=800 # 液压缸变矩系数,Nm/(MPa)
carparameter.Steer_FACTOR=17.5 # 转向传动比
carparameter.M_US=236 # 簧下质量，kg
carparameter.RRE=0.3 # 车轮有效滚动半径，m
carparameter.CF=-40000  # 前轮侧偏刚度，N/rad
carparameter.CR=-70000 # 后轮侧偏刚度，N/rad
carparameter.ROLL_RESISTANCE=0.02# 滚动阻力系数

# model_type  = 'Truck'
# carparameter=CarParameter()
# carparameter.LX_AXLE=5.0 # 轴距，m
# carparameter.LX_CG_SU=1.11 # 悬上质量质心至前轴距离，m
# carparameter.M_SU=4455.0  # 悬上质量，kg
# carparameter.IZZ_SU= 34802.6 # 转动惯量，kg*m^2
# carparameter.A=6.8 # 迎风面积，m^2
# carparameter.CFx=0.69 # 空气动力学侧偏角为零度时的纵向空气阻力系数
# carparameter.AV_ENGINE_IDLE=725.0 # 怠速转速，rpm
# carparameter.IENG=1.4 # 曲轴转动惯量，kg*m^2
# carparameter.TAU=0.5 # 发动机-变速箱输入轴 时间常数，s
# carparameter.R_GEAR_TR1=7.59 # 最低档变速箱传动比
# carparameter.R_GEAR_FD=5.0# 主减速器传动比
# carparameter.BRAK_COEF=40000.0 / 7.0 # 液压缸变矩系数,Nm/(MPa)
# carparameter.Steer_FACTOR=25.0 # 转向传动比
# carparameter.M_US= 1305.0 # 簧下质量，kg
# carparameter.RRE=0.51 # 车轮有效滚动半径，m
# carparameter.CF=-319703.0  # 前轮侧偏刚度，N/rad
# carparameter.CR=-97687.0 # 后轮侧偏刚度，N/rad
# carparameter.ROLL_RESISTANCE=0.0041# 滚动阻力系数




