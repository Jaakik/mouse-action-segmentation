#!/usr/bin/python2.7

import torch
from tcn_multitask import Trainer
from batch_gen_multitask import BatchGenerator
import os
import argparse
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')

parser.add_argument('--features_dim', default='2048', type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)


parser.add_argument('--num_f_maps', default='64', type=int)

# Need input
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_layers_PG', type=int)
parser.add_argument('--num_layers_R', type=int)
parser.add_argument('--num_R', type=int)

args = parser.parse_args()

num_epochs = args.num_epochs
features_dim = 113
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

sample_rate = 1

vid_list_file_1 = "data/task_1/splits/train_1.split.bundle"
vid_list_file_tst_1 = "data/task_1/splits/test.split.bundle"
features_path_1 = "data/task_1/features/"
gt_path_1 = "data/task_1/groundTruth/"
mapping_file_1 = "data/task_1/mapping.txt"

vid_list_file_3 = "data/task_3/splits/train_3.split.bundle"
vid_list_file_tst_3 = "data/task_3/splits/test.split.bundle"
features_path_3 = "data/task_3/features/"
gt_path_3 = "data/task_3/groundTruth/"
mapping_file_3 = "data/task_3/mapping.txt"


vid_list_file_2 = "data/task_2/splits/train_2.split.bundle"
vid_list_file_tst_2 = "data/task_2/splits/test.split.bundle"
features_path_2 = "data/task_2/features/"
gt_path_2 = "data/task_2/groundTruth/"

model_dir = "./models/"+args.dataset+"/split_"+args.split
results_dir = "./results/"+args.dataset+"/split_"+args.split

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr_1 = open(mapping_file_1, 'r')
actions = file_ptr_1.read().split('\n')[:-1]
file_ptr_1.close()
actions_dict_1 = dict()
for a in actions:
    actions_dict_1[a.split()[1]] = int(a.split()[0])
num_classes_1 = len(actions_dict_1)

file_ptr_3 = open(mapping_file_3, 'r')
actions = file_ptr_3.read().split('\n')[:-1]
file_ptr_3.close()
actions_dict_3 = dict()
for a in actions:
    actions_dict_3[a.split()[1]] = int(a.split()[0])

num_classes_3 = len(actions_dict_3)
print(num_classes_3)

trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes_1=num_classes_1,num_classes_3=num_classes_3)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes_1,num_classes_3, actions_dict_1,actions_dict_3, gt_path_1,gt_path_2,gt_path_3, features_path_1,features_path_2,features_path_3, sample_rate)
    batch_gen.read_data(vid_list_file_1, vid_list_file_2,vid_list_file_3)

    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path_1, vid_list_file_tst_1, num_epochs, actions_dict_1, device, sample_rate)

