#!/bin/bash
../../tools/dist_test.sh ../../configs/cgps/prw.py prw.pth 1 --out results_1000.pkl >log_tmp.txt 2>&1 
python ../../tools/test_results_prw.py >result.txt 2>&1
