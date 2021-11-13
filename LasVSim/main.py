# coding=utf-8
import os,sys
base_dir=os.path.dirname(__file__)
sys.path.append(base_dir)
import LasVSim.lasvsim

def check_import():
    print('import success')

if __name__ == "__main__":
    lasvsim.create_simulation()

    while lasvsim.sim_step_internal():
        pass

    lasvsim.export_simulation_data('C:/Users/Chason/Desktop/test.csv')
    print('done')