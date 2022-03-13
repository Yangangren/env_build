import os
import csv

class Data_IDC:
    """

    保存IDC每一步的仿真数据

    """
    def __init__(self):
        self.file = None  # 保存数据的文件
        self.file_path = None
        self.n = 0

    def new_file(self, path, file_name):
        if not os.path.exists(path):
            os.makedirs(path)
        self.file_path = path + '/' + file_name + '.csv'
        self.file = open(self.file_path, 'a')
        N = 150  # N>视野内交通参与者数目
        headers = ['time'] + ['Ego_v_x', 'ego_v_y', 'ego_r', 'ego_x', 'ego_y', 'ego_phi'] + ['steer', 'acc'] +\
                  ['Track_x', 'track_y', 'track_phi', 'track_v'] + ['phase', 'task', 'path_indexs', 'path_value'] + \
                  ['Sur_x', 'sur_y', 'sur_v', 'sur_phi', 'sur_l', 'sur_w', 'sur_route', 'sur_type'] * N
        with open(self.file_path, 'w', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def write(self, time, ego, action, track, light, task, path_index, path_values, others):
        data_0 = [time]
        data_1 = [light, task, path_index, path_values]
        data = [data_0, ego, action, track, data_1, others]
        step_data = []
        for i in data:
            step_data.extend(i)
        f = open(self.file_path, 'a', encoding='utf8', newline='')
        writer = csv.writer(f)
        writer.writerow(step_data)

    def read(self, file_path):
        # 仅输出csv中第n行数据（跳过表头）
        # 默认从第一行输出,循环一次输出下一行
        self.file_path = file_path
        data_step = None
        f = open(self.file_path, 'r')
        reader = csv.reader(f)
        head = next(reader)
        for index, info in enumerate(reader):
            if index == self.n:
                data_step = info
        self.n += 1
        return data_step

    def split_data(self, data_step):
        # 拆分该行的各类信息
        time = data_step[0]
        ego = data_step[1: 7]
        action = data_step[7: 9]
        track = data_step[9: 13]
        phase = data_step[13]
        task = data_step[14]
        path_index = data_step[15]
        path_values = data_step[16]
        sur = data_step[17:]
        return time, ego, action, track, phase, task, path_index, path_values, sur

    def split_ego(self, data_step):
        ego_vx = float(data_step[1])
        ego_vy = float(data_step[2])
        ego_r = float(data_step[3])
        ego_x = float(data_step[4])
        ego_y = float(data_step[5])
        ego_phi = float(data_step[6])
        return ego_vx, ego_vy, ego_r, ego_x, ego_y, ego_phi

    def split_sur(self, sur_data, i):
        # i表示该行sur信息中第i个交通参与者
        # 循环执行--连续输出该行的交通参与者信息
        per_sur_dim = 8
        # number_sur = len(sur_data)/per_sur_dim
        # for i in range(number_sur):
        sur_x = float(sur_data[i * per_sur_dim])
        sur_y = float(sur_data[i * per_sur_dim + 1])
        sur_v = float(sur_data[i * per_sur_dim + 2])
        sur_phi = float(sur_data[i * per_sur_dim + 3])
        sur_l = float(sur_data[i * per_sur_dim + 4])
        sur_w = float(sur_data[i * per_sur_dim + 5])
        sur_route = sur_data[i * per_sur_dim + 6]
        sur_type = sur_data[i * per_sur_dim + 7]
        # 放在draw_vehicle ？
        return sur_x, sur_y, sur_v, sur_phi, sur_l, sur_w, sur_route, sur_type

def test_csv():
    sim_data = Data_IDC()
    # path = 'D:/codecode/AAAmine/Toyota_ryg/env_build/hierarchical_decision/data_results'
    # name = 'simulation_data'
    # sim_data.new_file(path, name)
    # sim_data.write(0, None)
    file_path = 'D:/codecode/AAAmine/Toyota_ryg/env_build/hierarchical_decision/data_results/2022-03-11-09-49-03/episode0.csv'
    n = 2
    for _ in range(n):
        # 输出前n行（跳过表头）n>=1
        data_step = sim_data.read(file_path)
        time, ego, action, track, phase, task, path_index, path_values, sur = sim_data.split_data(data_step)
        print(time, ego, action, track, phase, task, path_index, path_values)
    ego_vx, ego_vy, ego_r, ego_x, ego_y, ego_phi = sim_data.split_ego(data_step)
    print(ego_vx, ego_vy, ego_r, ego_x, ego_y, ego_phi)
    m = 2
    for i in range(m):
        # 输出第n行数据中第1-m个交通参与者的信息
        sur_x, sur_y, sur_v, sur_phi, sur_l, sur_w, sur_route, sur_type = sim_data.split_sur(sur, i)
        print(sur_x, sur_y, sur_v, sur_phi, sur_l, sur_w, sur_route, sur_type)


if __name__ == '__main__':
    test_csv()