#!/bin/bash

intpol=0
python3 preprocessing.py --task 1 --intpol $intpol
python3 standardize.py --task 1 --intpol $intpol

python3 preprocessing.py --task 2 --intpol $intpol
python3 standardize.py --task 2 --intpol $intpol

python3 preprocessing.py --task 3 --intpol $intpol
python3 standardize.py --task 3 --intpol $intpol

python3 preprocessing.py --task 4 --intpol $intpol
python3 standardize.py --task 4 --intpol $intpol
