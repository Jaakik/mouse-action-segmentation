import pandas as pd
import numpy as np
import os
from pipeline import pipeline
from helpers import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='1', type=int)
    args = parser.parse_args()
    task = args.task
    train = np.load('data/task_'+ str(task) +'/train.npy', allow_pickle=True).item()
    test = np.load('data/task_'+ str(task) +'/test.npy', allow_pickle=True).item()
    sample_submission = np.load('data/task_'+ str(task) + '/sample_submission.npy', allow_pickle=True).item()

    # class_to_number = {s: i for i, s in enumerate(train['vocabulary'])}
    # number_to_class = {i: s for i, s in enumerate(train['vocabulary'])}
    # vocabulary = train['vocabulary']

    if task<3:
        tr_data = pipeline(train, train=True, task=task, behavior=-1)
    else:
        for behavior in train.keys():
            print(behavior)
            b=int(behavior.split('-')[-1])
            pipeline(train[behavior], train=True, task=task, behavior=b)

