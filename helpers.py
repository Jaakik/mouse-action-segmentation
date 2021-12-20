

import copy

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import torch
import random

body_parts = 7
fps = 30

aug_num = {1: 0, 2: 3, 3: 4}


def extract_features(data, train=False):
    '''
    Extracting the data and reshaping it. It returns a dictionary indexed by seq_id, the value of which corresponds
     to the flattened data, therefore an (n_frames x 28) matrix at this point. Labels are only extracted if we're
     training data, otherwise the annotation dictionary will be empty '''

    data_dict = {}
    anno_dict = {}
    label_dict = {}

    sequence_train_names = list(data["sequences"].keys())
    for sequence_key in tqdm(sequence_train_names):
        single_sequence = data["sequences"][sequence_key]
        # features Dimensions: (# frames) x (mouse ID) x (x, y coordinate) x (body part)
        keypoints = single_sequence["keypoints"]
        n_frames = keypoints.shape[0]
        data_dict[sequence_key] = np.reshape(keypoints, (n_frames, 28))
        anno_dict[sequence_key] = data["sequences"][sequence_key]['annotator_id']
        if train:
            label_dict[sequence_key] = data["sequences"][sequence_key]['annotations']
    return data_dict, label_dict, anno_dict

def anno_id_counts(dataset):
  all_annotator_ids = [dataset["sequences"][k]['annotator_id'] for k in dataset["sequences"]]
  unique_annotator_ids, annotator_id_counts = np.unique(all_annotator_ids, return_counts=True)
  for uaid, aic in zip(unique_annotator_ids, annotator_id_counts):
      print(f"Annotator id: {uaid} |  Number of sequences: {aic}")


def interpolate_frame(data, label_dict, n):
    sequence_train_names = list(data.keys())
    for sequence_key in tqdm(sequence_train_names):
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

def seed_everything(seed):
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)

def bounding_box(data):
    ''' Adds a feature which corresponds to the area of intersection between the mice based on the two rectangles
    that are defined by their position in the space '''
    sequence_train_names = list(data.keys())
    for sequence_key in tqdm(sequence_train_names):
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
        int_area[(dx<0) | (dy<0)] = 0
        out[:, 0:28] = single_sequence[:, 0:28]
        out[:, 28] = int_area
        data[sequence_key] = out
    return data


def center_of_mass(data):
    sequence_train_names = data.keys()
    for sequence_key in tqdm(sequence_train_names):
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
    for seq_name in tqdm(data_dict.keys()):
        center_dict[seq_name] = np.zeros(data_dict[seq_name].shape)
        center_dict[seq_name][0:7] = data_dict[seq_name][0:7] - x_offset
        center_dict[seq_name][7:14] = data_dict[seq_name][7:14] - y_offset
        center_dict[seq_name][14:21] = data_dict[seq_name][14:21] - x_offset
        center_dict[seq_name][21:28] = data_dict[seq_name][21:28] - y_offset
    return center_dict

def standardize(data_dict, x_max, y_max):
    standard_dict = {}
    for seq_name in tqdm(data_dict.keys()):
        standard_dict[seq_name] = np.zeros(data_dict[seq_name].shape)
        standard_dict[seq_name][0:7] = data_dict[seq_name][0:7]/x_max
        standard_dict[seq_name][7:14] = data_dict[seq_name][7:14]/y_max
        standard_dict[seq_name][14:21] = data_dict[seq_name][14:21]/x_max
        standard_dict[seq_name][21:28] = data_dict[seq_name][21:28]/y_max
    return standard_dict

def speed(data_dict):
    for seq_id in tqdm(list(data_dict.keys())):
        keypoints = data_dict[seq_id]
        shape = keypoints.shape
        n_frames = shape[0]
        if shape[1]!=36:
            keypoints = keypoints[:, 0:36]
            shape = keypoints.shape
        for frame in range(n_frames):
            mouse1_xy = np.vstack([keypoints[frame][0:7], keypoints[frame][7:14]]).T
            mouse2_xy = np.vstack([keypoints[frame][14:21], keypoints[frame][21:28]]).T
            upper_indices = np.triu_indices(body_parts, 1)
            intra1_d = cdist(mouse1_xy, mouse1_xy, 'euclidean')[upper_indices]
            intra2_d = cdist(mouse2_xy, mouse2_xy, 'euclidean')[upper_indices]
            extra_d = cdist(mouse1_xy, mouse2_xy, 'euclidean')[upper_indices]

            if (frame < n_frames-1):
                mouse1_xy_fl = np.vstack([keypoints[frame+1][0:7], keypoints[frame][7:14]]).T
                mouse2_xy_fl = np.vstack([keypoints[frame+1][14:21], keypoints[frame][21:28]]).T
                joints_velocity_1 = np.diagonal(cdist(mouse1_xy_fl, mouse1_xy, 'euclidean'))*fps
                joints_velocity_2 = np.diagonal(cdist(mouse2_xy_fl, mouse2_xy, 'euclidean'))*fps

            if (frame == 0):
                feature_f = np.hstack([joints_velocity_1, joints_velocity_2, intra1_d, intra2_d, extra_d])
            else:
                feat_new = np.hstack([joints_velocity_1, joints_velocity_2, intra1_d, intra2_d, extra_d])
                feature_f = np.vstack([feature_f, feat_new])
        data_dict[seq_id] = np.zeros((shape[0], shape[1]+feature_f.shape[1]))
        data_dict[seq_id][:, 0:36] = keypoints
        data_dict[seq_id][:, 36:] = feature_f
        del feature_f
    return data_dict

def augment_fn(X):
    new_X = copy.copy(X)
    mouse1xy = X[:, 0:14].reshape((X.shape[0], 2, 7))
    mouse2xy = X[:, 14:28].reshape((X.shape[0], 2, 7))

    # mirroring
    mirror_x = torch.randint(low=0, high=2, size=(X.shape[0], 1))
    mirror_x[mirror_x == 0] = -1

    mouse1xy[:, 0, :] = mirror_x * mouse1xy[:, 0, :]
    mouse2xy[:, 0, :] = mirror_x * mouse2xy[:, 0, :]

    mirror_y = torch.randint(low=0, high=2, size=(X.shape[0], 1))
    mirror_y[mirror_y == 0] = -1

    mouse1xy[:, 1, :] = mirror_y * mouse1xy[:, 1, :]
    mouse2xy[:, 1, :] = mirror_y * mouse2xy[:, 1, :]

    # rotation
    angle = (torch.rand(X.shape[0]) - 0.5) * (np.pi * 2)
    c, s = torch.cos(angle), torch.sin(angle)
    rot = torch.stack([c, -s, s, c])
    rot = rot.T
    rot = rot.reshape((-1, 2, 2))
    mouse1xy = torch.einsum('lij,ljk->lik', rot, torch.tensor(mouse1xy, dtype=torch.float32))
    mouse2xy = torch.einsum('lij,ljk->lik', rot, torch.tensor(mouse2xy, dtype=torch.float32))

    # shift
    shift_x = (torch.rand(X.shape[0], 1) - 0.5) * 2 * 0.25
    shift_y = (torch.rand(X.shape[0], 1) - 0.5) * 2 * 0.25
    mouse1xy[:, 0, :] = mouse1xy[:, 0, :] + shift_x
    mouse1xy[:, 1, :] = mouse1xy[:, 1, :] + shift_y
    mouse2xy[:, 0, :] = mouse2xy[:, 0, :] + shift_x
    mouse2xy[:, 1, :] = mouse2xy[:, 1, :] + shift_y

    mouse1xy = mouse1xy.reshape((-1, 14))
    mouse2xy = mouse2xy.reshape((-1, 14))

    new_X[:, 0:14] = mouse1xy
    new_X[:, 14:28] = mouse2xy

    # Gaussian noise
    mu, sigma = 0, 0.2
    noise = torch.empty(X.shape).normal_(mean=mu, std=sigma)
    new_X = new_X + np.array(noise)
    return new_X


def augment(data, labels, annotators, task, behavior=-1):

    seed_everything(2021)
    keys = list(data.keys())
    if task==2:
        rand_aug = random.sample(keys, 70%len(keys))
        for seq_id in keys:
            for a in range(aug_num[task]-1):
                if ((a == aug_num[task]-2) & (seq_id not in rand_aug)): continue
                new_name = seq_id + "aug" + str(a+1) + "anno" +str(annotators[seq_id])
                data[new_name] = augment_fn(data[seq_id])
                labels[new_name] = labels[seq_id]
            new_name = seq_id + "aug" + str(0) + "anno" + str(annotators[seq_id])
            data[new_name] = data[seq_id]
            labels[new_name] = labels[seq_id]
            del data[seq_id]
            del labels[seq_id]
    if task == 3:
        a_num = int(np.floor(10 / len(data)))
        rand_aug = random.sample(keys, 10 % len(keys))
        for seq_id in keys:
            for a in range(a_num):
                if ((a == a_num - 1) & (seq_id not in rand_aug)): continue
                new_name = seq_id + "aug" + str(a+1) + "beh" + str(behavior)
                data[new_name] = augment_fn(data[seq_id])
                labels[new_name] = labels[seq_id]
            new_name = seq_id + "aug" + str(0) + "beh" + str(behavior)
            data[new_name] = data[seq_id]
            labels[new_name] = labels[seq_id]
            del data[seq_id]
            del labels[seq_id]
    return data, labels