# Mouse Action Segmentation

Most of the information and data can be found under the following challenge on [AiCrowd](https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications)

## Data and preprocessing:
The project consists of 3 tasts, each of which requires a slightly different dataset. Before beginning it is necessary 
to download the necessary dataset, and arrange it in the following structure under the `data` directory:
```directory structure
.
├── data
│     ├── task_1
│     │     ├── splits
│     │     │     └── ...
│     │     ├── mapping.txt
│     │     └── train.npy
│     ├── task_2
│     │     ├── splits
│     │     │     └── ...
│     │     ├── mapping.txt
│     │     └── train.npy
│     ├── task_3
│     │     ├── splits
│     │     │     └── ...
│     │     ├── mapping.txt
│     │     └── train.npy
│     └── test.npy
└── ...              
```
In the above structure, the train.npy are different for each task and the test.npy is the same for all. Those can be found 
at the following links:
- [Task 1](https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications/problems/mabe-task-1-classical-classification/dataset_files?unique_download_uri=3289&challenge_id=755)
- [Task 2](https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications/problems/mabe-task-2-annotation-style-transfer/dataset_files?unique_download_uri=3292&challenge_id=756)
- [Task 3](https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications/problems/mabe-task-2-annotation-style-transfer/dataset_files?unique_download_uri=3292&challenge_id=756)
- [Test](https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications/problems/mabe-task-1-classical-classification/dataset_files?unique_download_uri=3300&challenge_id=755)

Make sure that a splits directory is present under each task directory. 

Once this directory structure is created, it is possible to run the preprocessing script from the working directory using the following command:
```
sh ./scripts/preprocessing.sh n
```
Where `n` is an integer variable representing the amount of interpolation between the frames. Use `n=0` for no interpolation.

Following the execution of the script, the data will be ready for training, and present under the correct directory structure. 
Under each task, three directories will be created which are described below:
- `featuresIntn` - Directory containing raw feature files, in `.npy` format
- `featuresStdIntn` - Directory containing standardized (all columns between 0 and 1) feature files, in the `.npy` format
- `groundTruthIntn` - Directory containing labels for the given sequence

where n is the interpolation number. For tasks 2&3 some of the file names may be additionally annotated with the 
augmentation #, the annotator number and the behavior number, depending on the task.

## Utils
The preprocessing script executes the correct forms of data augmentation specific to each task, as described in the project paper.
This makes use of the `preprocessing.py` script, which in turns calls `pipeline.py`. While those two scripts take care 
of the correct file manipulation, all of the data augmentation implementations can be found in the `helpers.py` file. 
Functions from this file are called from withing the `pipeline.py` script.

# Training & Prediciton 
Run run.py to generate the predictions using ridge regression with fine tuned hyper parameters. 

## Requirements
* Python >= 3.8
* numpy 










