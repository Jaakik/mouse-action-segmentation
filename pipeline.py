import pandas as pd
import numpy as np
import os
from helpers import *

FRAME_WIDTH_TOP = 1024
FRAME_HEIGHT_TOP = 570

body_parts = 7
fps = 30

RESIDENT_COLOR = 'lawngreen'
INTRUDER_COLOR = 'skyblue'

INT_CONSTANT = 2



PLOT_MOUSE_START_END = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4),
                        (3, 5), (4, 6), (5, 6), (1, 2)]


class_to_color = {'other': 'white', 'attack': 'red', 'mount': 'green',
                  'investigation': 'orange'}

def pipeline(data, train=False, task=1, behavior=-1, intpol=-1):
    ''' The data parameter is the raw imported dataset '''

    # Obtaining the correct output paths
    if task<4:
        outpath = "data/task_" + str(task) + "/"
    elif task==4:
        outpath = "data/testFeat/"

    # Extracting the data and the labels (if training set)
    print("Extracting features")
    data, labels, annotators = extract_features(data, train)
    # centering the coordinates around the middle
    # data = center_origin(data, FRAME_WIDTH_TOP / 2, FRAME_HEIGHT_TOP / 2)

    if train:
        data, labels = augment(data, labels, annotators, task, behavior)

    # (normalize maybe?)
    #
    # print("Standardizing")
    # data = standardize(data, FRAME_WIDTH_TOP, FRAME_HEIGHT_TOP)


    # Interpolating if we're dealing with training data
    if train & (len(labels) != 0) & intpol>0:
        print("Interpolating")
        data, labels = interpolate_frame(data, labels, intpol)

    # augmenting data with bounded box intersection area (adding one column)
    print("Augmenting with bounding box")
    # data = bounding_box(data)

    # augmenting data with center of mass data, egocentric distance, mouse extension
    print("Augmenting with center of mass")
    data = center_of_mass(data)

    print("Augmenting with speed data")
    data = speed(data, INT_CONSTANT+1)

    print("Writing Files")
    if task<4:
        outFeat = outpath+"featuresInt"+str(intpol)+"/"
        outGT = outpath+"groundTruthInt"+str(intpol)+"/"
        if not os.path.exists(outGT):
            os.makedirs(outGT)
    elif task == 4:
        outFeat = outpath
    if not os.path.exists(outFeat):
        os.makedirs(outFeat)

    for seq_id in tqdm(list(data.keys())):
        np.save(outFeat+seq_id+".npy", data[seq_id])
        if task<4:
            np.savetxt(outGT+seq_id+".txt", labels[seq_id].astype(int), fmt ='%.0f')

    return data