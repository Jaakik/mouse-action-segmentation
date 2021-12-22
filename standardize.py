import argparse
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='1', type=int)
    parser.add_argument('--intpol', default='0', type=int)
    args = parser.parse_args()
    intpol = args.intpol
    task = args.task
    if task < 3:
        inPath = "data/task_" + str(task) + "/featuresInt" + str(intpol) + "/"
        outPath = "data/task_" + str(task) + "/featuresStdInt" + str(intpol) + "/"
    elif task==4:
        inPath = "data/testFeat/"
        outPath = "data/testFeatStd/"
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    files = os.listdir(inPath)
    for file in tqdm(files):
        seq_id = file.split("/")[-1].split(".")[0]
        features = np.load(inPath + file, allow_pickle=True)
        # features = np.delete(features, 28, axis=1)
        features = features - np.min(features, axis=0)
        features = np.divide(features, np.max(features, axis=0))
        features = np.where(features!=np.nan, features, 0)
        features = np.where(np.isnan(features), 0, features)
        np.save(outPath+file, features)
