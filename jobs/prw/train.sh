#!/bin/bash

config_name="prw"
config_path="../../configs/cgps/${config_name}.py" 
python -u ../../tools/train.py $config_path >train_log.txt 2>&1
