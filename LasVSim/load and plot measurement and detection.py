import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
file_place=r'D:\Seafile\硕士毕业论文\第四章实验\chapter4_experiment1_tracking'
err=np.load(file_place+'\err.npy',allow_pickle=True)
detection_GT=np.load(file_place+'\detecton.npy', allow_pickle=True)
fusion_result=np.load(file_place+r'\result.npy', allow_pickle=True)
fontdictCH = {'family': 'SimSun',
        'weight': 'normal',
        'size': 16,
        }

#plt.rcParams['figure.figsize'] = (12.0, 6.0)
detection_GT_integrated=pd.DataFrame()
err_integrated=pd.DataFrame()
measurement_integrated=pd.DataFrame()
fusion_result_integrated=pd.DataFrame()
fusion_error_integrated=pd.DataFrame()

#画那部车
id2plot=108
time_upper_limit=3000
time_lower_limit=1900
for t in range(len(detection_GT)):
    if (t<time_lower_limit):
        continue
    if (t>time_upper_limit):
        continue
    if (str(id2plot) in detection_GT[t][0]):
        veh=detection_GT[t][0][str(id2plot)]
        # type, x, y, heading, width, length, vx, vy)
        tmp = pd.DataFrame({'time': t, 'id':id2plot,'type':veh[0],'x':veh[1],'y':veh[2],'theta':veh[3],'w':veh[4],
                            'l':veh[5],'vx':veh[6],'vy':veh[7]}, index=[0])
        detection_GT_integrated=detection_GT_integrated.append(tmp, ignore_index=True)
        # 整合融合结果
        veh_fusion=fusion_result[t][0][str(id2plot)]
        tmp_fusion = pd.DataFrame(
            {'time': t, 'id': id2plot, 'type': veh_fusion[0], 'x': veh_fusion[1], 'y': veh_fusion[2], 'theta': veh_fusion[3], 'w': veh_fusion[4],
             'l': veh_fusion[5], 'vx': veh_fusion[6], 'vy': veh_fusion[7]}, index=[0])
        fusion_result_integrated = fusion_result_integrated.append(tmp_fusion, ignore_index=True)
        #计算融合误差
        tmp_fusion_err=pd.DataFrame(
            {'time': t, 'id': id2plot, 'object_type': veh[0], 'Ex': tmp['x']-tmp_fusion['x'], 'Ey': tmp['y']-tmp_fusion['y'],
             'Etheta': tmp['theta']-tmp_fusion['theta'], 'Ew': tmp['w']-tmp_fusion['w'],
             'El': tmp['l']-tmp_fusion['l'], 'Evx': tmp['vx']-tmp_fusion['vx'], 'Evy': tmp['vy']-tmp_fusion['vy']}, index=[0])
        fusion_error_integrated=fusion_error_integrated.append(tmp_fusion_err)
    if (str(id2plot) in detection_GT[t][1]):
        man=detection_GT[t][1][str(id2plot)]
        # type, x, y, heading, width, length, vx, vy)
        tmp = pd.DataFrame({'time': t, 'id':id2plot,'type':man[0],'x':man[1],'y':man[2],'theta':man[3],'w':man[4],
                            'l':man[5],'vx':man[6],'vy':man[7]}, index=[0])
        detection_GT_integrated=detection_GT_integrated.append(tmp, ignore_index=True)

        # 整合融合结果
        man_fusion = fusion_result[t][1][str(id2plot)]
        tmp_fusion = pd.DataFrame(
            {'time': t, 'id': id2plot, 'type': man_fusion[0], 'x': man_fusion[1], 'y': man_fusion[2], 'theta': man_fusion[3], 'w': man_fusion[4],
             'l': man_fusion[5], 'vx': man_fusion[6], 'vy': man_fusion[7]}, index=[0])
        fusion_result_integrated = fusion_result_integrated.append(tmp_fusion, ignore_index=True)
        # 计算融合误差
        tmp_fusion_err = pd.DataFrame(
            {'time': t, 'id': id2plot, 'object_type': man[0], 'Ex': tmp['x'] - tmp_fusion['x'],
             'Ey': tmp['y'] - tmp_fusion['y'],
             'Etheta': tmp['theta'] - tmp_fusion['theta'], 'Ew': tmp['w'] - tmp_fusion['w'],
             'El': tmp['l'] - tmp_fusion['l'], 'Evx': tmp['vx'] - tmp_fusion['vx'],
             'Evy': tmp['vy'] - tmp_fusion['vy']}, index=[0])
        fusion_error_integrated = fusion_error_integrated.append(tmp_fusion_err)
for t in range(len(err)):
    if (t<time_lower_limit):
        continue
    if (t>time_upper_limit):
        continue
    for pair in err[t]:
        # 指定哪一辆车
        if (pair[0]==id2plot):
            #print(t)
            #不知道为啥GT比err领先一位
            try:
                Index = detection_GT_integrated[detection_GT_integrated['time'] == t].index.tolist()[0]
            except:
                Index = detection_GT_integrated[detection_GT_integrated['time'] == (t + 1)].index.tolist()[0]
            tmp=pd.DataFrame({'time': t, 'id': pair[0], 'sensor id': pair[1],'object_type':pair[2],'Ex':pair[3],'Ey':pair[4],
                              'Etheta':pair[5],'Ew':pair[6],'El':pair[7],'Evx':pair[8],'Evy':pair[9]},index=[0])
            err_integrated=err_integrated.append(tmp, ignore_index=True)
            measurement_tmp=pd.DataFrame({'time': t, 'id': pair[0],'sensor id': pair[1],'object_type':pair[2],'x': detection_GT_integrated['x'][Index] + pair[3],
                                          'y': detection_GT_integrated['y'][Index] + pair[4], 'theta': detection_GT_integrated['theta'][Index] + pair[5],
                                          'w': detection_GT_integrated['w'][Index] + pair[6], 'l': detection_GT_integrated['l'][Index] + pair[7],
                                          'vx': detection_GT_integrated['vx'][Index] + pair[8], 'vy': detection_GT_integrated['vy'][Index] + pair[9]}, index=[0])
            measurement_integrated=measurement_integrated.append(measurement_tmp, ignore_index=True)


#markers = {"0": ".", "1": "x",'2':'^','3':'s'}
def plot_discountinous(x,y,data,linewidth=1,color='darkviolet'):
    X=np.array(data[x])
    Y=np.array(data[y])
    Xinterval=[]
    l=0
    for r in range(len(X)):
        try: vr=X[r+1]
        except:#最后一步
            Xinterval.append([l, r])
        if vr-X[r]>1:
            Xinterval.append([l, r])
            l=r+1
    for interval in Xinterval:
        L=interval[0]
        R=interval[1]
        plt.plot(X[L:R],Y[L:R],linewidth=linewidth, color=color)
    return
def plot_fusion_error(x,y,err_data,fusion_err_data,y_label,x_label=r"时间 [秒]",linewidth=1, color='darkviolet'):
    # 画误差和滤波结果
    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.17, 0.12, 0.8, 0.86])
    sns.scatterplot(x=x, y=y, hue='sensor id', data=err_data, palette="bright", style='sensor id')
    plot_discountinous(x=x, y=y, data=fusion_err_data, linewidth=linewidth, color=color)
    xticks=[i for i in range(int(time_lower_limit/200),int(fusion_err_data['time'].iloc[-1]/200)+1)]
    plt.xticks([tick*200 for tick in xticks],[tick*10 for tick in xticks])
    ax1.set_ylabel(y_label, fontdict=fontdictCH)
    ax1.set_xlabel(x_label, fontdict=fontdictCH)
    plt.savefig(y_label+'.png', dpi=600)
    plt.show()

def plot_state(state,xlabel,ylabel=r"时间 [$\times 0.05$秒]"):
    #画状态
    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.17, 0.12, 0.8, 0.86])
    #sns.lineplot(x="time", y="Ex", data=err_integrated,palette="bright")
    sns.scatterplot(x="time", y=state, data=measurement_integrated, hue='sensor id', palette="bright", style='sensor id')
    #sns.lineplot(x='time', y=state, data=detection_GT_integrated, linewidth=2, color='r')
    plot_discountinous(x='time', y=state, data=fusion_result_integrated, linewidth=1, color='darkviolet')
    ax1.set_ylabel(xlabel, fontdict=fontdictCH)
    ax1.set_xlabel(ylabel, fontdict=fontdictCH)
    plt.show()
#画传感器感知时间
f1 = plt.figure(1)
ax1 = f1.add_axes([0.17, 0.12, 0.8, 0.86])
sns.scatterplot(x="time", y="sensor id", hue='sensor id', data=err_integrated, palette="bright", style='sensor id')
plt.yticks([0,1,2,3],['0','1','2','3'])
ax1.set_ylabel('传感器编号', fontdict=fontdictCH)
ax1.set_xlabel(r"时间 [$\times 0.05$秒]", fontdict=fontdictCH)
plt.show()
#画误差和滤波结果
plot_fusion_error(x='time',y='Ex',err_data=err_integrated,y_label='$E_x$',fusion_err_data=fusion_error_integrated)
plot_fusion_error(x='time',y='Ey',err_data=err_integrated,y_label='$E_y$',fusion_err_data=fusion_error_integrated)
plot_fusion_error(x='time',y='Etheta',err_data=err_integrated,y_label=r'$E_{\theta}$',fusion_err_data=fusion_error_integrated)
plot_fusion_error(x='time',y='Ew',err_data=err_integrated,y_label='$E_w$',fusion_err_data=fusion_error_integrated)
plot_fusion_error(x='time',y='El',err_data=err_integrated,y_label='$E_l$',fusion_err_data=fusion_error_integrated)
plot_fusion_error(x='time',y='Evx',err_data=err_integrated,y_label='$E_{vx}$',fusion_err_data=fusion_error_integrated)
plot_fusion_error(x='time',y='Evy',err_data=err_integrated,y_label='$E_{vy}$',fusion_err_data=fusion_error_integrated)
#
# #画状态
plot_state('x','$X_G$轴位置')
plot_state('y','$Y_G$轴位置')
plot_state('theta','朝向')
plot_state('w','宽度')
plot_state('l','长度')
plot_state('vx','$X_G$轴方向速度')
plot_state('vy','$Y_G$轴方向速度')


# f1 = plt.figure(1)
# ax1 = f1.add_axes([0.17, 0.12, 0.8, 0.86])
# #sns.lineplot(x="time", y="Ex", data=err_integrated,palette="bright")
# sns.scatterplot(x="time", y="x", data=measurement_integrated, hue='sensor id', palette="bright", style='sensor id')
# #sns.lineplot(x='time', y='x', data=detection_GT_integrated, linewidth=2, color='r')
# sns.lineplot(x='time', y='x', data=fusion_result_integrated, linewidth=1, color='y',markers='--')
# ax1.set_ylabel('X轴位置', fontdict=fontdictCH)
# ax1.set_xlabel(r"时间 [$\times 0.05$秒]", fontdict=fontdictCH)
# plt.show()
#
#
# f1 = plt.figure(1)
# ax1 = f1.add_axes([0.17, 0.12, 0.8, 0.86])
# #sns.lineplot(x="time", y="Ex", data=err_integrated,palette="bright")
# sns.scatterplot(x="time", y="y", data=measurement_integrated, hue='sensor id', palette="bright", style='sensor id')
# sns.lineplot(x='time', y='y', data=detection_GT_integrated, linewidth=2, color='r')
# ax1.set_ylabel('Y轴位置', fontdict=fontdictCH)
# ax1.set_xlabel(r"时间 [$\times 0.05$秒]", fontdict=fontdictCH)
# plt.show()
#
# f1 = plt.figure(1)
# ax1 = f1.add_axes([0.17, 0.12, 0.8, 0.86])
# #sns.lineplot(x="time", y="Ex", data=err_integrated,palette="bright")
# sns.scatterplot(x="time", y="theta", data=measurement_integrated, hue='sensor id', palette="bright", style='sensor id')
# sns.lineplot(x='time', y='theta', data=detection_GT_integrated, linewidth=2, color='r')
# ax1.set_ylabel('朝向', fontdict=fontdictCH)
# ax1.set_xlabel(r"时间 [$\times 0.05$秒]", fontdict=fontdictCH)
# plt.show()
#
# f1 = plt.figure(1)
# ax1 = f1.add_axes([0.17, 0.12, 0.8, 0.86])
# #sns.lineplot(x="time", y="Ex", data=err_integrated,palette="bright")
# sns.scatterplot(x="time", y="w", data=measurement_integrated, hue='sensor id', palette="bright", style='sensor id')
# sns.lineplot(x='time', y='w', data=detection_GT_integrated, linewidth=2, color='r')
# ax1.set_ylabel('宽度', fontdict=fontdictCH)
# ax1.set_xlabel(r"时间 [$\times 0.05$秒]", fontdict=fontdictCH)
# plt.show()
#
# f1 = plt.figure(1)
# ax1 = f1.add_axes([0.17, 0.12, 0.8, 0.86])
# #sns.lineplot(x="time", y="Ex", data=err_integrated,palette="bright")
# sns.scatterplot(x="time", y="l", data=measurement_integrated, hue='sensor id', palette="bright", style='sensor id')
# sns.lineplot(x='time', y='l', data=detection_GT_integrated, linewidth=2, color='r')
# sns.lineplot(x='time', y='l', data=fusion_result_integrated, linewidth=1, color='y')
# ax1.set_ylabel('长度', fontdict=fontdictCH)
# ax1.set_xlabel(r"时间 [$\times 0.05$秒]", fontdict=fontdictCH)
# plt.show()
#
# f1 = plt.figure(1)
# ax1 = f1.add_axes([0.17, 0.12, 0.8, 0.86])
# #sns.lineplot(x="time", y="Ex", data=err_integrated,palette="bright")
# sns.scatterplot(x="time", y="vx", data=measurement_integrated, hue='sensor id', palette="bright", style='sensor id')
# sns.lineplot(x='time', y='vx', data=detection_GT_integrated, linewidth=2, color='r')
# ax1.set_ylabel('X轴方向速度', fontdict=fontdictCH)
# ax1.set_xlabel(r"时间 [$\times 0.05$秒]", fontdict=fontdictCH)
# plt.show()
#
# f1 = plt.figure(1)
# ax1 = f1.add_axes([0.17, 0.12, 0.8, 0.86])
# #sns.lineplot(x="time", y="Ex", data=err_integrated,palette="bright")
# sns.scatterplot(x="time", y="vy", data=measurement_integrated, hue='sensor id', palette="bright", style='sensor id')
# sns.lineplot(x='time', y='vy', data=detection_GT_integrated, linewidth=2, color='r')
# ax1.set_ylabel('Y轴方向速度', fontdict=fontdictCH)
# ax1.set_xlabel(r"时间 [$\times 0.05$秒]", fontdict=fontdictCH)
# plt.show()
