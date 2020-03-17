#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.pyplot as pyplt
from matplotlib import gridspec

def get_ds_infos():
    """
    Read the file includes data subject information.
    
    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]
    
    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes 
    """ 

    dss = pd.read_csv("./../data/data_subjects_info.csv")
    print("[INFO] -- Data subjects' information is imported.")
    
    return dss

def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.
    
    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t+".x",t+".y",t+".z"])
        else:
            dt_list.append([t+".roll", t+".pitch", t+".yaw"])

    return dt_list


def creat_time_series(dt_list, act_labels, trial_codes, mode="mag", labeled=True):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.

    Returns:
        It returns a time-series of sensor data.
    
    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

    if labeled:
        dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 
    else:
        dataset = np.zeros((0,num_data_cols))
        
    ds_list = get_ds_infos()
    
    print("[INFO] -- Creating Time-Series")
    for sub_id in ds_list["code"]:
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                fname = './../data/A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                vals = np.zeros((len(raw_data), num_data_cols))
                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        
                    else:
                        vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                    vals = vals[:,:num_data_cols]
                if labeled:
                    lbls = np.array([[act_id,
                            sub_id-1,
                            ds_list["weight"][sub_id-1],
                            ds_list["height"][sub_id-1],
                            ds_list["age"][sub_id-1],
                            ds_list["gender"][sub_id-1],
                            trial          
                           ]]*len(raw_data))
                    vals = np.concatenate((vals, lbls), axis=1)
                dataset = np.append(dataset,vals, axis=0)
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]
            
    if labeled:
        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]
    
    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset
#________________________________

def creat_dataset():
    ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
    TRIAL_CODES = {
        ACT_LABELS[0]:[1,2,11],
        ACT_LABELS[1]:[3,4,12],
        ACT_LABELS[2]:[7,8,15],
        ACT_LABELS[3]:[9,16],
        ACT_LABELS[4]:[6,14],
        ACT_LABELS[5]:[5,13]
    }
    ## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
    ## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
    sdt = ["attitude", "userAcceleration"]
    print("[INFO] -- Selected sensor data types: "+str(sdt))    
    act_labels = ACT_LABELS [0:6]
    print("[INFO] -- Selected activites: "+str(act_labels))    
    trial_codes = [TRIAL_CODES[act] for act in act_labels]
    dt_list = set_data_types(sdt)
    dataset = creat_time_series(dt_list, act_labels, trial_codes, mode="raw", labeled=True)
    print("[INFO] -- Shape of time-Series dataset:"+str(dataset.shape))    
    return(dataset)

def drow_data_multi(data):
    data.plot()
    plt.xlabel('Second', fontsize=18)
    plt.ylabel('Value', fontsize=16)
    lgnd=plt.legend()
    fig = pyplt.gcf()
    fig.set_size_inches(18, 8)
    plt.show()

def drow_data_single(X_test):
    plt.rcParams["font.family"] = 'DejaVu Serif'
    plt.rcParams['text.usetex'] = True

    Text_size = 56

    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['xtick.labelsize'] = Text_size
    plt.rcParams['ytick.labelsize'] = Text_size
    plt.rcParams['legend.fontsize'] = Text_size
    plt.rcParams['axes.titlesize']=Text_size
    plt.rcParams['axes.labelsize']=Text_size
    plt.rcParams['figure.figsize'] = (24.0, 12.0)
    plt.rcParams['font.size'] = Text_size
    #################################################

    _ = plt.plot(X_test[0], '-o')

    plt.grid()
    plt.show()

def drow_matrix_cor(M_pairwise):
    Text_size = 30

    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['xtick.labelsize'] = Text_size
    plt.rcParams['ytick.labelsize'] = Text_size
    plt.rcParams['legend.fontsize'] = Text_size
    plt.rcParams['axes.titlesize']=Text_size
    plt.rcParams['axes.labelsize']=Text_size
    plt.rcParams['figure.figsize'] = (12.0, 12.0)
    plt.rcParams['font.size'] = Text_size
    #################################################

    _ = plt.imshow(M_pairwise)

    _ = plt.colorbar()

    plt.show()
    
def drow_data_cluster(X_test, prediction_vector, List_of_x, M_pairwise, T):    
    Text_size = 56

    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['xtick.labelsize'] = Text_size
    plt.rcParams['ytick.labelsize'] = Text_size
    plt.rcParams['legend.fontsize'] = Text_size
    plt.rcParams['axes.titlesize']=Text_size
    plt.rcParams['axes.labelsize']=Text_size
    plt.rcParams['figure.figsize'] = (24.0, 12.0)
    plt.rcParams['font.size'] = Text_size
    #################################################
    color = ['orange', 'green', 'red', 'yelow', 'blue']

    _ = plt.plot(X_test[0], '-')
    
    for t in np.unique(prediction_vector):
        ind = np.where(prediction_vector == t)
        _ = plt.plot(List_of_x[ind]+T, X_test[0][2*T:X_test[0].shape[0]-T][ind], 'o', color = color[t], label = 'Type ' + str(t + 1))



    plt.grid()
    plt.legend(loc = 'best')
    # plt.savefig('./results/'+series_name+'_claster_vector.png', bbox_inches='tight')
    plt.show()
    
def drow_phase_trajectory(List_of_All):
    index = 0

    _, _, List_of_points_plus, List_of_points_minus, List_of_t_plus, List_of_t_minus, line_point, ress = List_of_All[index]
    Text_size = 30

    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['xtick.labelsize'] = Text_size
    plt.rcParams['ytick.labelsize'] = Text_size
    plt.rcParams['legend.fontsize'] = Text_size
    plt.rcParams['axes.titlesize']=Text_size
    plt.rcParams['axes.labelsize']=Text_size
    plt.rcParams['figure.figsize'] = (12.0, 12.0)
    plt.rcParams['font.size'] = Text_size
    #################################################

    _ = plt.plot(ress[:, 0], ress[:, 1], '-o', color = 'blue')

    for point in List_of_points_plus:
        _ = plt.plot(point[0], point[1], '*', color = 'orange')
    for point in List_of_points_minus:
        _ = plt.plot(point[0], point[1], '*', color = 'red')

    x_line = np.array([-.25, .25])
    k = line_point[1]/line_point[0]
    y_line = k*x_line

    _ = plt.plot(x_line, y_line, '--', color = 'black')

    # plt.xlabel('Time $x$, $sec$')
    # plt.ylabel('Time $y$, $sec$')
    # plt.savefig('./results/'+series_name+'_full.png', bbox_inches='tight')
    plt.show()

def drow_big(X_test, M_pairwise, prediction_vector, T, resss, discrete):
    multicolor = True
    if multicolor:
        color = ['orange', 'green', 'red', 'yelow', 'blue']
    else:
        color = ['black', 'black', 'black', 'black', 'black']
    marker = ['^', 's', 'v', 'D', 'P']
    plt.rcParams["font.family"] = 'DejaVu Serif'
    plt.rcParams['text.usetex'] = True

    Text_size = 24

    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 4
    plt.rcParams['xtick.labelsize'] = Text_size
    plt.rcParams['ytick.labelsize'] = Text_size
    plt.rcParams['legend.fontsize'] = Text_size
    plt.rcParams['axes.titlesize']=Text_size
    plt.rcParams['axes.labelsize']=Text_size
    plt.rcParams['figure.figsize'] = (16.0, 12.0)
    plt.rcParams['font.size'] = Text_size
    plt.rcParams["legend.labelspacing"] = 0.1
    plt.rcParams["legend.handletextpad"] = 0.1
    #################################################



    fig = plt.figure();

    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1])

    ax1 = fig.add_subplot(gs[0]);
    ax2 = fig.add_subplot(gs[1]);
    ax3 = fig.add_subplot(gs[2]);
    ax4 = fig.add_subplot(gs[3]);

    #------___1___------
    _ = ax1.plot(X_test[0], '-', color = 'black', label = 'Time series')
    ax1.legend(loc = 'best')

    #------___2___------
    im = ax2.imshow(M_pairwise, cmap='gray')
    fig.colorbar(im, ax=ax2)


    #------___3___------
    _ = ax3.plot(X_test[0], '-', color = 'black', label = 'Time series')
    for t in np.unique(prediction_vector):
        ind = np.where(prediction_vector == t)
        _ = ax3.plot(np.arange(2*T, X_test[0].shape[0] - T)[ind], X_test[0][2*T:X_test[0].shape[0]-T][ind], linewidth = 0, marker = marker[t], color = color[t], label = 'Type ' + str(t + 1))
    ax3.legend(loc = 'best')

    #------___4___------
    for t in np.unique(prediction_vector):
        ind = np.where(prediction_vector == t)
        ind = [x//discrete for x in ind]
        ind = np.unique(ind)
        _ = ax4.plot(resss[:, 0][ind], resss[:, 1][ind], linewidth = 0, marker = marker[t], color = color[t], label = 'Type ' + str(t + 1))
    ax4.grid()
    ax4.legend(loc = 'best')
    # ax4.set_xlabel('First principal component')
    # ax4.set_ylabel('Second principal component')

    plt.show()
    
def drow_with_segments(X_test, List_of_All, List_of_point, prediction_vector, T, List_of_x):  
    
    Text_size = 56

    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['xtick.labelsize'] = Text_size
    plt.rcParams['ytick.labelsize'] = Text_size
    plt.rcParams['legend.fontsize'] = Text_size
    plt.rcParams['axes.titlesize']=Text_size
    plt.rcParams['axes.labelsize']=Text_size
    plt.rcParams['figure.figsize'] = (24.0, 12.0)
    plt.rcParams['font.size'] = Text_size
    #################################################
    color = ['orange', 'green', 'red', 'yelow', 'blue']

    _ = plt.plot(X_test[0], '-')

    for t in np.unique(prediction_vector):
    #     _ = plt.plot(List_of_x[0] + T, 0, color = color[t], label = 'Type ' + str(t + 1))
        ind = List_of_point[t] + T
        for x in (List_of_x + T)[ind]:
            _ = plt.axvline(x = x, color = color[t])


    for t in np.unique(prediction_vector):
        ind = np.where(prediction_vector == t)
        _ = plt.plot(List_of_x[ind]+T, X_test[0][2*T:X_test[0].shape[0]-T][ind], 'o', color = color[t], label = 'Type ' + str(t + 1))



    plt.grid()
    plt.legend(loc = 'best')
    plt.xlabel('Time $t$, $sec$')
    plt.ylabel('Acceleration $x$, $m/sec^2$')
    plt.show()

