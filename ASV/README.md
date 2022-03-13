# 0x00 ASV
This directory provides code related to extracting speaker embeddings. 
We used some public pre-trained models and some models we trained ourselves on the Voxceleb2 dataset to extract speaker embeddings.
We also provide some models that include the use of the WavLM model as a feature extractor, and then fine-tuning on the Voxceleb2 dataset.
```bash
./run_spk_ebd_extract.sh
```
# 0x02 Test different ASV models and AASIST(CM) pre-train model

```bash
./run_test_nonorm.sh
```
```bash
./run_test_nonorm.sh
```

# 0x03 Result
# NoNorm Score Fusion

| ASV Model         | DEV SASV EER |  DEV SV EER   |  DEV SPF EER  | EVAL SASV EER |  EVAL SV EER   |   EAVL SPF EER | 
| ---------------   | :---------   | :---------    |  :------------| :---------    |  :---------    |  :------------ | 
| ECAPA-TDNN        |     1.08%    |     1.82%     |     0.13%     |     1.40%     |     1.53%      |       1.22%    |  
| SE-ResNet-34      |     0.54%    |     1.01%     |     0.09%     |     0.76%     |     0.74%      |       0.78%    |  
| SE-Res2Net-50     |     0.13%    |     0.16%     |     0.07%     |     0.55%     |     0.32%      |       0.74%    |  
| Res2NeXt-50       |     0.20%    |     0.54%     |     0.07%     |     0.66%     |     0.41%      |       0.90%    |  

## L2Norm Score Fusion
| ASV Model         | DEV SASV EER |  DEV SV EER   |  DEV SPF EER  | EVAL SASV EER |  EVAL SV EER   |   EAVL SPF EER | 
| ---------------   | :---------   | :---------    |  :------------| :---------    |  :---------    |  :------------ | 
| ECAPA-TDNN        |     0.87%    |     1.55%     |     0.08%     |     1.10%     |     1.08%      |       1.10%    |  
| SE-ResNet-34      |     0.54%    |     1.01%     |     0.09%     |     0.77%     |     0.74%      |       0.79%    |  
| SE-Res2Net-50     |     0.13%    |     0.16%     |     0.07%     |     0.55%     |     0.32%      |       0.73%    |  
| Res2NeXt-50       |     0.20%    |     0.50%     |     0.07%     |     0.65%     |     0.41%      |       0.90%    |  



