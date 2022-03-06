#!/bin/bash

# AASIST.py     IRSE50.py    Res2Net50V2.py  ResNet34V2.py
# ECAPATDNN.py  __pycache__  Res2NeXt50.py   WavLMLarge.py 

python test_asv_cm.py --gpu 5 --norm nonorm --model ECAPATDNN 
python test_asv_cm.py --gpu 5 --norm nonorm --model ResNet34V2
python test_asv_cm.py --gpu 5 --norm nonorm --model Res2Net50V2 
python test_asv_cm.py --gpu 5 --norm nonorm --model Res2NeXt50
