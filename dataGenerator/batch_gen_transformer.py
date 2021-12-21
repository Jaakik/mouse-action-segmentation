'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''

import torch
import random
import numpy as np
from scipy.stats import truncnorm
import torch.nn.functional as TF
import torch.nn as nn

class TimeWarpLayer(nn.Module):
    def __init__(self):
        super(TimeWarpLayer, self).__init__()

    def forward(self, x, grid, mode='bilinear'):
        '''
        :type&shape x: (cuda.)FloatTensor, (N, D, T)
        :type&shape grid: (cuda.)FloatTensor, (N, T, 2)
        :type&mode: bilinear or nearest
        :rtype&shape: (cuda.)FloatTensor, (N, D, T)
        '''
        assert len(x.shape) == 3
        assert len(grid.shape) == 3
        assert grid.shape[-1] == 2
        x_4dviews = list(x.shape[:2]) + [1] + list(x.shape[2:])
        grid_4dviews = list(grid.shape[:1]) + [1] + list(grid.shape[1:])
        out = TF.grid_sample(input=x.view(x_4dviews), grid=grid.view(grid_4dviews), mode=mode, align_corners=True).view(x.shape)
        return out


class GridSampler():
    def __init__(self, N_grid, low=1, high=5):  # high=5
        N_primary = 100 * N_grid
        assert N_primary % N_grid == 0
        self.N_grid = N_grid
        self.N_primary = N_primary
        self.low = low
        self.high = high

    def sample(self, batchsize=1):
        num_centers = np.random.randint(low=self.low, high=self.high)
        lower, upper = 0, 1
        mu, sigma = np.random.rand(num_centers), 1 / (num_centers * 1.5)  # * 1.5
        TN = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        vals = TN.rvs(size=(self.N_primary, num_centers))
        grid = np.sort(
            np.random.choice(vals.reshape(-1), size=self.N_primary, replace=False))  # pick one center for each primary
        grid = (grid[::int(self.N_primary / self.N_grid)] * 2 - 1).reshape(1, self.N_grid, 1)  # range [-1, 1)
        grid = np.tile(grid, (batchsize, 1, 1))
        grid = np.concatenate([grid, np.zeros_like(grid)], axis=-1)
        return grid


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

        self.timewarp_layer = TimeWarpLayer()

    def reset(self):
        self.index = 0
        self.my_shuffle()

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        self.gts = [self.gt_path + vid for vid in self.list_of_examples]
        self.features = [self.features_path + vid.split('.')[0] + '.npy' for vid in self.list_of_examples]
        self.my_shuffle()

    def my_shuffle(self):
        # shuffle list_of_examples, gts, features with the same order
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.list_of_examples)
        random.seed(randnum)
        random.shuffle(self.gts)
        random.seed(randnum)
        random.shuffle(self.features)


    def warp_video(self, batch_input_tensor, batch_target_tensor):
        '''
        :param batch_input_tensor: (bs, C_in, L_in)
        :param batch_target_tensor: (bs, L_in)
        :return: warped input and target
        '''
        bs, _, T = batch_input_tensor.shape
        grid_sampler = GridSampler(T)
        grid = grid_sampler.sample(bs)
        grid = torch.from_numpy(grid).float()

        warped_batch_input_tensor = self.timewarp_layer(batch_input_tensor, grid, mode='bilinear')
        batch_target_tensor = batch_target_tensor.unsqueeze(1).float()
        warped_batch_target_tensor = self.timewarp_layer(batch_target_tensor, grid, mode='nearest')  # no bilinear for label!
        warped_batch_target_tensor = warped_batch_target_tensor.squeeze(1).long()  # obtain the same shape

        return warped_batch_input_tensor, warped_batch_target_tensor

    def merge(self, bg, suffix):
        '''
        merge two batch generator. I.E
        BatchGenerator a;
        BatchGenerator b;
        a.merge(b, suffix='@1')
        :param bg:
        :param suffix: identify the video
        :return:
        '''

        self.list_of_examples += [vid + suffix for vid in bg.list_of_examples]
        self.gts += bg.gts
        self.features += bg.features

        print('Merge! Dataset length:{}'.format(len(self.list_of_examples)))


    def next_batch(self, batch_size, if_warp=False): # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]

        self.index += batch_size

        batch_input = []
        batch_target = []
        for idx, vid in enumerate(batch):
            features = np.load(batch_features[idx])
            file_ptr = open(batch_gts[idx], 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]

            feature = features[:, ::self.sample_rate]
            target = classes[::self.sample_rate]
            batch_input.append(feature)
            batch_target.append(target)

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)  # bs, C_in, L_in
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            if if_warp:
                warped_input, warped_target = self.warp_video(torch.from_numpy(batch_input[i]).unsqueeze(0), torch.from_numpy(batch_target[i]).unsqueeze(0))
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]], batch_target_tensor[i, :np.shape(batch_target[i])[0]] = warped_input.squeeze(0), warped_target.squeeze(0)
            else:
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask, batch


