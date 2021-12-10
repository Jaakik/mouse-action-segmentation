import pandas as pd
import numpy as np
import os
from pipeline import pipeline


if __name__ == "__main__":
    train = np.load('data/train.npy', allow_pickle=True).item()
    test = np.load('data/test.npy', allow_pickle=True).item()
    sample_submission = np.load('data/sample_submission.npy', allow_pickle=True).item()

    class_to_number = {s: i for i, s in enumerate(train['vocabulary'])}
    number_to_class = {i: s for i, s in enumerate(train['vocabulary'])}
    vocabulary = train['vocabulary']

    tr_data = pipeline(train, train=True)

