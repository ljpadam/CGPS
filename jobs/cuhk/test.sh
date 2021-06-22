#!/bin/bash
config_name="cuhk"
config_path="../../configs/cgps/${config_name}.py" 
python -u ../../tools/test_personsearch.py $config_path  cuhk.pth --eval bbox >>result.txt 2>&1
