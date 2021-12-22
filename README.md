# mouse action segmentation

You can download the challenge data following the instructions in [AiCrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files)

# Data:
Create a data folder and place the test.csv and train.csv files following this projce folde structure :
```directory structure
root ── main 
      ├─ data ─── test.csv
      │       ├─ train.csv
      │                    
```
# Utils:
Please checkout data_utils.py and train_utils.py for all the helper functions that were involved in data processing and training. 

# Implementations:
Please checkout implementations.py for the implementation of the cost functions that were used in this project

# Training & Prediciton 
Run run.py to generate the predictions using ridge regression with fine tuned hyper parameters. 

## Requirements
* Python >= 3.8
* numpy 










