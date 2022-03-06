# 0x00 ASV
This directory provides code related to extracting speaker embeddings. 
We used some public pre-trained models and some models we trained ourselves on the Voxceleb2 dataset to extract speaker embeddings.
We also provide some models that include the use of the WavLM model as a feature extractor, and then fine-tuning on the Voxceleb2 dataset, because WavLM train with other datasets, so the score file we submit to the sponsor does not include the above Models. 
# 0x01 Extract Speaker Embeddings
```bash
./run.sh
```
# 0x02 Test Speaker Embedings and CM pre-train model
This test script is only for fusion testing with a single CM system, using the simplest score sum method, just to compare the performance of different speaker recognition models.

```python
python test_asv_cm.py --LA_dataset_path LA --model ECAPATDNN 
```
# 0x03 Result
From the results, it can be seen that simple fusion on each ASV model and pre-trained CM system can obtain results that are significantly ahead of the official baseline. Other complex results will be provided in the CM subdirectory, where more complex system fusion methods are provided, and of course better performance than the table below.  


| ASV Model         |  DEV SV EER  |  DEV SPF EER  |  DEV SASV EER | EVAL SV EER  |  EVAL SPF EER  |  EAVL SASV EER | 
| ---------------   | :---------   | :---------    |  :------------| :---------   |  :---------    |  :------------ | 
| ECAPATDNN(L2Norm) |   1.55%      |     0.12%     |      0.88%    |    1.07%     |    1.63%       |      1.35%     |  
| ECAPATDNN(NoNorm) |    1.82%     |     0.19%     |      1.08%    |    1.51%     |    1.77%       |      1.60%     |  
| ResNet34V2(L2Norm) |   1.01%     |     0.13%     |     0.54%     |     0.73%    |     1.17%      |       0.97%    |  
| ResNet34V2(NoNorm) |   1.01%     |     0.13%     |     0.55%     |     0.73%    |     1.15%      |       0.96%    |  
| Res2Net50V2(L2Norm)|     0.16%   |      0.07%    |     0.13%     |     0.30%    |     1.10%      |       0.78%    |  
| Res2Net50V2(NoNorm)|     0.16%   |      0.07%    |     0.13%     |     0.30%    |     1.10%      |       0.78%    |  
| Res2NeXt50(L2Norm) |    0.50%    |      0.09%    |     0.23%     |     0.39%    |     1.32%      |       0.96%    |  
| Res2NeXt50(NoNorm) |    0.54%    |      0.08%    |     0.23%     |     0.41%    |     1.34%      |       0.97%    |  
