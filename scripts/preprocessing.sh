#!/bin/bash
# Usage:
#   sh preprocessing.sh n
# where n is the number of interpolated frames in between each frame.
# This script generates the necessary feature files that can be fed into the model. Make sure that before running the
# Following folder structure is maintained:
#  .
#  ├── data
#  │     ├── task_1
#  │     │     ├── splits
#  │     │     │     └── ...
#  │     │     └── train.npy
#  │     ├── task_2
#  │     │     ├── splits
#  │     │     │     └── ...
#  │     │     └── train.npy
#  │     ├── task_3
#  │     │     ├── splits
#  │     │     │     └── ...
#  │     │     └── train.npy
#  │     └── test.npy
#  └── ...
#
#
# Where the train.npy and test.npy files are the dataset files for the corresponding tasks that can be found at the following links:
#     task 1 - https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications/problems/mabe-task-1-classical-classification/dataset_files?unique_download_uri=3289&challenge_id=755
#     task 2 - https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications/problems/mabe-task-2-annotation-style-transfer/dataset_files?unique_download_uri=3292&challenge_id=756
#     task 3 - https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications/problems/mabe-task-3-learning-new-behavior/dataset_files?unique_download_uri=3295&challenge_id=757
#     test   - https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications/problems/mabe-task-1-classical-classification/dataset_files?unique_download_uri=3300&challenge_id=755


intpol=$1
echo "Preprocessing task 1"
python3 preprocessing.py --task 1 --intpol $intpol
echo "Standardizing task 1"
python3 standardize.py --task 1 --intpol $intpol

echo
echo "Preprocessing task 2"
python3 preprocessing.py --task 2 --intpol $intpol
echo "Standardizing task 2"
python3 standardize.py --task 2 --intpol $intpol

echo
echo "Preprocessing task 3"
python3 preprocessing.py --task 3 --intpol $intpol
echo "Standardizing task 3"
python3 standardize.py --task 3 --intpol $intpol

echo
echo "Preprocessing testing data"
python3 preprocessing.py --task 4
echo "Standardizing testing data"
python3 standardize.py --task 4