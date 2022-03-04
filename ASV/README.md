# ASV
This directory provides code related to extracting speaker embeddings. 
We used some public pre-trained models and some models we trained ourselves on the Voxceleb2 dataset to extract speaker embeddings.
We also provide some models that include the use of the WavLM model as a feature extractor, and then fine-tuning on the Voxceleb2 dataset, because WavLM train with other datasets, so the score file we submit to the sponsor does not include the above Models. 
# Extract Speaker Embeddings
```bash
./run.sh
```
# Test Speaker Embedings and CM pre-train model
`This test script is only for fusion testing with a single CM system, using the simplest score sum method, just to compare the performance of different speaker recognition models.` 

```python
python test_asv_cm.py --LA_dataset_path LA --model ECAPATDNN 
```