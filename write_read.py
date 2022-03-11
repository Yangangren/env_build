import os
import csv

class Data_IDC:
    """

    保存IDC每一步的仿真数据

    """
    def __init__(self):
        self.file = None  # 保存数据的文件
        self.file_path = None

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

    def read(self, file_path, n):
        self.file_path = file_path
        f = open(self.file_path, 'r')
        reader = csv.reader(f)
        head = next(reader)
        for index, info in enumerate(reader):
            if index == n:
                print(info)


def test_csv():
    sim_data = Data_IDC()
    # path = 'D:/codecode/AAAmine/Toyota_ryg/env_build/hierarchical_decision/data_results'
    # name = 'simulation_data'
    # sim_data.new_file(path, name)
    # sim_data.write(0, None)

    file_path = 'D:/codecode/AAAmine/Toyota_ryg/env_build/hierarchical_decision/data_results/2022-03-10-22-44-32/episode0.csv'
    sim_data.read(file_path, 0)


if __name__ == '__main__':
    test_csv()