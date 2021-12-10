import pandas as pd
import numpy as np
import os
from helpers import *

FRAME_WIDTH_TOP = 1024
FRAME_HEIGHT_TOP = 570

RESIDENT_COLOR = 'lawngreen'
INTRUDER_COLOR = 'skyblue'

INT_CONSTANT = 2

PLOT_MOUSE_START_END = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4),
                        (3, 5), (4, 6), (5, 6), (1, 2)]

class_to_color = {'other': 'white', 'attack': 'red', 'mount': 'green',
                  'investigation': 'orange'}

def pipeline(data, train=False):
    ''' The data parameter is the raw imported dataset '''


    sequence_names = list(data["sequences"].keys())


    # Extracting the data and the labels (if training set)
    extr_data, extr_label = extract_features(data, train)

    # centering the coordinates around the middle
    ctr_data = center_origin(extr_data, FRAME_WIDTH_TOP/2, FRAME_HEIGHT_TOP/2)

    # Interpolating if we're dealing with training data
    if train & (len(extr_label) == 0):
        data, labels = interpolate_frame(ctr_data, extr_label, INT_CONSTANT)
    else:
        data = ctr_data

    # augmenting data with bounded box intersection area (adding one column)
    data = bounding_box(data)

    # augmenting data with center of mass data, egocentric distance, mouse extension
    data = center_of_mass(data)


    return data