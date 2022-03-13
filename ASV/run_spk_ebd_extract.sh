#!/bin/bash
# !!!must set LA dataset path!!!
LA_dataset_path=

weights_dir="weights"
if [ ! -d $weights_dir ];then
    mkdir $weights_dir
fi
# download resnet_34v2 pretrain model from https://github.com/clovaai/voxceleb_trainer
if [ ! -f $weights_dir/resnet34v2.model ];then
    wget  "http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model"
    mv baseline_v2_ap.model $weights_dir/resnet34v2.model
fi
# download ecapatdnn pretrain model from https://github.com/TaoRuijie/ECAPA-TDNN
if [ ! -f $weights_dir/ecapatdnn.model ];then
    wget "https://github.com/TaoRuijie/ECAPA-TDNN/raw/a2290930b910a3cba7e099d2447d02c18919b3a4/exps/pretrain.model"
    mv pretrain.model $weights_dir/ecapatdnn.model
fi
if [ ! -f $weights_dir/res2net50v2.model ];then
    wget "https://www.webprague.com/sasv2022_doubleroc_pretrain/res2net50v2.model"
    mv res2net50v2.model $weights_dir/res2net50v2.model
fi
if [ ! -f $weights_dir/res2next50.model ];then
    wget "https://www.webprague.com/sasv2022_doubleroc_pretrain/res2next50.model"
    mv res2next50.model $weights_dir/res2next50.model
fi
# WavLM Large model is download from https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
# the original model is store to GoogleDriver, can't download use CLI, so we restore to our server.
# if [ ! -f $weights_dir/wavlm_large_finetune.model ];then
#     wget "https://www.webprague.com/sasv2022_doubleroc_pretrain/wavlm_large_finetune.model"
#     mv wavlm_large_finetune.model $weights_dir/wavlm_large_finetune.model
# fi
# download pre-train CM model for test
if [ ! -f $weights_dir/AASIST.model ];then
    wget "https://github.com/clovaai/aasist/raw/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/weights/AASIST.pth"
    mv AASIST.pth $weights_dir/AASIST.model
fi

# extract spk embedding
python save_embeddings.py --model ECAPATDNN   --LA_dataset_path $LA_dataset_path 
python save_embeddings.py --model ResNet34V2  --LA_dataset_path $LA_dataset_path
python save_embeddings.py --model Res2Net50V2 --LA_dataset_path $LA_dataset_path
python save_embeddings.py --model Res2NeXt50  --LA_dataset_path $LA_dataset_path

