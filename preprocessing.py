# python3 preprocessing.py --task n --intpol i

import pandas as pd
import numpy as np
import os
from pathlib import Path
from pipeline import pipeline
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='1', type=int)
    parser.add_argument('--intpol', default='0', type=int)
    args = parser.parse_args()
    task = args.task
    intpol = args.intpol

    if task<4:
        task_dir = 'data/task_' + str(task) + '/'
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        train = np.load(task_dir + 'train.npy', allow_pickle=True).item()
    elif task == 4:
        test = np.load('data/test.npy', allow_pickle=True).item()


    if task<3:
        tr_data = pipeline(train, train=True, task=task, behavior=-1, intpol=intpol)
    elif task == 3:
        for behavior in train.keys():
            print(behavior)
            b = int(behavior.split('-')[-1])
            pipeline(train[behavior], train=True, task=task, behavior=b, intpol=intpol)
            print("\n")
    elif task == 4:
        te_data = pipeline(test, train=False, task=task, behavior=-1, intpol=0)
    if task<4:
        files = os.listdir(task_dir + "groundTruthInt" + str(intpol) + "/")
        split_dir = task_dir + "splits/"
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        bundle = split_dir + "trainInt" + str(intpol) + ".bundle"
        Path(bundle).touch()
        with open(bundle, "w") as out:
            for file in files:
                out.writelines(file + "\n")
