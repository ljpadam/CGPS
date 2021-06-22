#!/bin/bash

config_name="cuhk"
config_path="../../configs/cgps/${config_name}.py" 

python -u ../../tools/train.py ${config_path} >train_log.txt 2>&1 
