#!/bin/bash

python3 run_tcn.py --action=train   \
                --num_epochs=100 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --intpol 0
