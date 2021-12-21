#!/usr/bin/python2.7

import torch
import numpy as np
import random


class BatchGenerator(object):
    def __init__(self, num_classes_1, num_classes_3, actions_dict_1,actions_dict_3, gt_path_1, gt_path_2, gt_path_3,
                 features_path_1, features_path_2, features_path_3, sample_rate):
        self.index = 0

        self.list_of_examples_1 = list()
        self.list_of_examples_2 = list()
        self.list_of_examples_3 = list()

        self.num_classes_1 = num_classes_1
        self.num_classes_3 = num_classes_3

        self.actions_dict_1 = actions_dict_1
        self.actions_dict_3 = actions_dict_3

        self.gt_path_1 = gt_path_1
        self.gt_path_2 = gt_path_2
        self.gt_path_3 = gt_path_3

        self.features_path_1 = features_path_1
        self.features_path_2 = features_path_2
        self.features_path_3 = features_path_3

        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples_1)
        random.shuffle(self.list_of_examples_2)
        random.shuffle(self.list_of_examples_3)

    def has_next(self):
        if self.index < len(self.list_of_examples_1):
            return True
        return False

    def read_data(self, vid_list_file_1, vid_list_file_2, vid_list_file_3):
        file_ptr = open(vid_list_file_1, 'r')
        self.list_of_examples_1 = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples_1)

        file_ptr = open(vid_list_file_2, 'r')
        self.list_of_examples_2 = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples_2)

        file_ptr = open(vid_list_file_3, 'r')
        self.list_of_examples_3 = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples_3)

    # applying a list of classes of 70n for both
    def next_batch(self, batch_size):
        batch_1 = self.list_of_examples_1[self.index:self.index + batch_size]
        batch_2 = self.list_of_examples_2[self.index:self.index + batch_size]
        batch_3 = self.list_of_examples_3[self.index:self.index + batch_size]

        self.index += batch_size

        batch_input_1 = []
        batch_target_1 = []
        batch_input_2 = []
        batch_target_2 = []
        batch_input_3 = []
        batch_target_3 = []
        beh = "beh_"
        ann = "ann_"

        for vid in batch_1:
            features = np.load(self.features_path_1 + vid.split('.')[0] + '.npy').T
            file_ptr = open(self.gt_path_1 + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict_1[content[i]]
            batch_input_1.append(features[:, ::self.sample_rate])
            batch_target_1.append(classes[::self.sample_rate])

        for vid in batch_2:
            # identifier at the end of each video
            ann += vid.split('.')[0][-1]
            features = np.load(self.features_path_2 + vid.split('.')[0] + '.npy').T
            file_ptr = open(self.gt_path_2 + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict_1[content[i]]
            batch_input_2.append(features[:, ::self.sample_rate])
            batch_target_2.append(classes[::self.sample_rate])

        for vid in batch_3:
            # identifier at the end of each video
            beh += vid.split('.')[0][-1]
            features = np.load(self.features_path_3 + vid.split('.')[0] + '.npy').T
            file_ptr = open(self.gt_path_3 + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict_3[content[i]]
            batch_input_3.append(features[:, ::self.sample_rate])
            batch_target_3.append(classes[::self.sample_rate])

        length_of_sequences_1 = list(map(len, batch_target_1))
        batch_input_tensor_1 = torch.zeros(len(batch_input_1), np.shape(batch_input_1[0])[0],
                                           max(length_of_sequences_1), dtype=torch.float)
        batch_target_tensor_1 = torch.ones(len(batch_input_1), max(length_of_sequences_1), dtype=torch.long) * (-100)
        mask_1 = torch.zeros(len(batch_input_1), self.num_classes_1, max(length_of_sequences_1), dtype=torch.float)
        for i in range(len(batch_input_1)):
            batch_input_tensor_1[i, :, :np.shape(batch_input_1[i])[1]] = torch.from_numpy(batch_input_1[i])
            batch_target_tensor_1[i, :np.shape(batch_target_1[i])[0]] = torch.from_numpy(batch_target_1[i])
            mask_1[i, :, :np.shape(batch_target_1[i])[0]] = torch.ones(self.num_classes_1, np.shape(batch_target_1[i])[0])

        ##################################
        length_of_sequences_2 = list(map(len, batch_target_2))
        batch_input_tensor_2 = torch.zeros(len(batch_input_2), np.shape(batch_input_2[0])[0],
                                           max(length_of_sequences_2), dtype=torch.float)
        batch_target_tensor_2 = torch.ones(len(batch_input_2), max(length_of_sequences_2), dtype=torch.long) * (-100)
        mask_2 = torch.zeros(len(batch_input_2), self.num_classes_1, max(length_of_sequences_2), dtype=torch.float)
        for i in range(len(batch_input_2)):
            batch_input_tensor_2[i, :, :np.shape(batch_input_2[i])[1]] = torch.from_numpy(batch_input_2[i])
            batch_target_tensor_2[i, :np.shape(batch_target_2[i])[0]] = torch.from_numpy(batch_target_2[i])
            mask_2[i, :, :np.shape(batch_target_2[i])[0]] = torch.ones(self.num_classes_1, np.shape(batch_target_2[i])[0])
        ##################################

        length_of_sequences_3 = list(map(len, batch_target_3))
        batch_input_tensor_3 = torch.zeros(len(batch_input_3), np.shape(batch_input_3[0])[0],
                                           max(length_of_sequences_3),
                                           dtype=torch.float)
        batch_target_tensor_3 = torch.ones(len(batch_input_3), max(length_of_sequences_3), dtype=torch.long) * (-100)
        mask_3 = torch.zeros(len(batch_input_3), self.num_classes_3, max(length_of_sequences_3), dtype=torch.float)
        for i in range(len(batch_input_3)):
            batch_input_tensor_3[i, :, :np.shape(batch_input_3[i])[1]] = torch.from_numpy(batch_input_3[i])
            batch_target_tensor_3[i, :np.shape(batch_target_3[i])[0]] = torch.from_numpy(batch_target_3[i])
            mask_3[i, :, :np.shape(batch_target_3[i])[0]] = torch.ones(self.num_classes_3, np.shape(batch_target_3[i])[0])

        return batch_input_tensor_1, batch_target_tensor_1, mask_1, batch_input_tensor_2, batch_target_tensor_2, mask_2, batch_input_tensor_3, batch_target_tensor_3, mask_3, beh, ann
