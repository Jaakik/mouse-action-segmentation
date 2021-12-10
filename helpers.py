import numpy as np
import pandas as pd


def extract_features(data, train=False):
    '''
    Extracting the data and reshaping it. It returns a dictionary indexed by seq_id, the value of which corresponds
     to the flattened data, therefore an (n_frames x 28) matrix at this point. Labels are only extracted if we're
     training data, otherwise the annotation dictionary will be empty '''

    data_dict = {}
    anno_dict = {}
    sequence_train_names = list(data["sequences"].keys())
    for sequence_key in sequence_train_names :
        single_sequence = data["sequences"][sequence_key]
        # features Dimensions: (# frames) x (mouse ID) x (x, y coordinate) x (body part)
        keypoints = single_sequence["keypoints"]
        n_frames = keypoints.shape[0]
        data_dict[sequence_key] = np.reshape(keypoints, (n_frames, 28))
        if train:
            anno_dict[sequence_key] = data["sequences"][sequence_key]['annotations']
    return data_dict, anno_dict


def interpolate_frame(data, label_dict, n):
    sequence_train_names = list(data.keys())
    for sequence_key in sequence_train_names:
        single_sequence = pd.DataFrame(data[sequence_key])
        single_sequence['annotations'] = label_dict[sequence_key]

        new_index = pd.RangeIndex(len(single_sequence) * (n + 1))
        single_sequence_ext = pd.DataFrame(np.nan, index=new_index, columns=single_sequence.columns)
        ids = np.arange(len(single_sequence)) * (n + 1)
        single_sequence_ext.loc[ids] = single_sequence.values
        single_sequence_ext.drop(single_sequence_ext.index[-n:], inplace=True)
        single_sequence_ext['annotations'].interpolate('backfill', inplace=True)
        single_sequence_ext.interpolate('quadratic', inplace=True)

        data[sequence_key] = single_sequence_ext.loc[:, single_sequence_ext.columns!='annotations'].to_numpy()
        label_dict[sequence_key] = single_sequence_ext.loc[:, single_sequence_ext.columns=='annotations'].to_numpy().flatten()
    return data, label_dict


def bounding_box(data):
    ''' Adds a feature which corresponds to the area of intersection between the mice based on the two rectangles
    that are defined by their position in the space '''
    sequence_train_names = list(data.keys())
    for sequence_key in sequence_train_names :
        single_sequence = data[sequence_key].copy()
        out = np.zeros((single_sequence.shape[0], 29))

        mouse1_x = single_sequence[:, 0:7]
        mouse1_y = single_sequence[:, 7:14]
        mouse2_x = single_sequence[:, 14:21]
        mouse2_y = single_sequence[:, 21:28]

        x_min_1 = np.min(mouse1_x, axis=1)
        y_min_1 = np.min(mouse1_y, axis=1)
        x_max_1 = np.max(mouse1_x, axis=1)
        y_max_1 = np.min(mouse1_y, axis=1)

        x_min_2 = np.min(mouse2_x, axis=1)
        y_min_2 = np.min(mouse2_y, axis=1)
        x_max_2 = np.max(mouse2_x, axis=1)
        y_max_2 = np.min(mouse2_y, axis=1)

        dx = np.minimum(x_max_1, x_max_2) - np.maximum(x_min_1, x_min_2)
        dy = np.minimum(y_max_1, y_max_2) - np.maximum(y_min_1, y_min_2)
        int_area = np.abs(dx*dy)
        int_area[(dx<0) & (dy<0)] = 0
        out[:, 0:28] = single_sequence[:, 0:28]
        out[:, 28] = int_area
        data[sequence_key] = out
    return data


def center_of_mass(data):
    sequence_train_names = data.keys()
    for sequence_key in sequence_train_names:
        single_sequence = data[sequence_key]
        shape = single_sequence.shape
        data[sequence_key] = np.zeros((shape[0], 36))
        # keypoints = single_sequence["keypoints"]
        n_frames = single_sequence.shape[0]
        mouse1_x = single_sequence[:, 0:7]
        mouse1_y = single_sequence[:, 7:14]
        mouse2_x = single_sequence[:, 14:21]
        mouse2_y = single_sequence[:, 21:28]
        mouse1_center = np.vstack([np.average(mouse1_x, axis=1), np.average(mouse1_y, axis=1)]).T
        mouse2_center = np.vstack([np.average(mouse2_x, axis=1), np.average(mouse2_y, axis=1)]).T
        mouse1_dispersion = np.average(
            np.sqrt(np.square(mouse1_x - mouse1_center[:, 0:1]) + np.square(mouse1_y - mouse1_center[:, 1:2])), axis=1)
        mouse2_dispersion = np.average(
            np.sqrt(np.square(mouse2_x - mouse2_center[:, 0:1]) + np.square(mouse2_y - mouse2_center[:, 1:2])), axis=1)
        distance_between_centers = np.sqrt(np.sum(np.square(mouse1_center - mouse2_center), axis=1))
        data[sequence_key][:, 0:29] = single_sequence[:, 0:29]
        data[sequence_key][:, 29:31] = mouse1_center
        data[sequence_key][:, 31:33] = mouse2_center
        data[sequence_key][:, 33] = mouse1_dispersion
        data[sequence_key][:, 34] = mouse2_dispersion
        data[sequence_key][:, 35] = distance_between_centers
    return data


def get_percentage(sequence_key, num_to_text, data, vocabulary):
    anno_seq = num_to_text(data['sequences'][sequence_key]['annotations'])
    counts = {k: np.mean(np.array(anno_seq) == k) for k in vocabulary}
    return counts


def center_origin(data_dict, x_offset, y_offset):
    center_dict = {}
    for seq_name in data_dict.keys():
        center_dict[seq_name] = np.zeros(data_dict[seq_name].shape)
        center_dict[seq_name][0:7] = data_dict[seq_name][0:7] - x_offset
        center_dict[seq_name][7:14] = data_dict[seq_name][7:14] - y_offset
        center_dict[seq_name][14:21] = data_dict[seq_name][14:21] - x_offset
        center_dict[seq_name][21:28] = data_dict[seq_name][21:28] - y_offset
    return center_dict





