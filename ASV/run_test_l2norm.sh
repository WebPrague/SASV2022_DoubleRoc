#!/bin/bash

# AASIST.py     IRSE50.py    Res2Net50V2.py  ResNet34V2.py
# ECAPATDNN.py  __pycache__  Res2NeXt50.py   WavLMLarge.py 

python test_asv_cm.py --gpu 5 --norm l2norm --model ECAPATDNN 
python test_asv_cm.py --gpu 5 --norm l2norm --model ResNet34V2
python test_asv_cm.py --gpu 5 --norm l2norm --model Res2Net50V2 
python test_asv_cm.py --gpu 5 --norm l2norm --model Res2NeXt50
