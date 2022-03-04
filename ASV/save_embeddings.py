import os
import tqdm
import torch
import argparse
import numpy as np
import pickle as pk
import torch.nn as nn
from typing import Dict, List
import torch.nn.functional as F
from torch.utils.data import DataLoader
from test_dataset import get_dataloader
from models.ResNet34V2 import MainModel as ResNet34V2
from models.ECAPATDNN import MainModel as ECAPATDNN
from models.Res2Net50V2 import MainModel as Res2Net50V2
from models.Res2NeXt50 import MainModel as Res2NeXt50
from models.WavLMLarge import MainModel as WavLMLarge

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--LA_dataset_path', type=str, default='LA', help='The model to Run')
    parser.add_argument(
        "--ASV_embedings_save_path", type=str, default="embeddings"
    )
    parser.add_argument(
        "--model", type=str, default="ECAPATDNN"
    )
    parser.add_argument(
        "--gpu", type=str, default="0"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    LA_dataset_path = args.LA_dataset_path
    if not os.path.exists(LA_dataset_path):
        print('LA Datset is not exist.')
        exit(0)
    if args.model == 'ECAPATDNN':
        model = ECAPATDNN('./weights/ecapatdnn.model')
    elif args.model == 'ResNet34V2':
        model = ResNet34V2('./weights/resnet34v2.model')
    elif args.model == 'Res2Net50V2':
        model = Res2Net50V2('./weights/res2net50v2.model')
    elif args.model == 'Res2NeXt50':
        model = Res2NeXt50('./weights/res2next50.model')
    elif args.model == 'WavLMLarge':
        model = WavLMLarge('./weights/wavlm_large_finetune.model')
    else:
        print(f'model {args.model} is not supoort.')
        exit(0)
    model = model.to(device)
    model = model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.eval()
        elif isinstance(module, nn.BatchNorm1d):
            module.track_running_stats = False
            module.eval()
    for data_dirnmae in ['ASVspoof2019_LA_dev', 'ASVspoof2019_LA_train', 'ASVspoof2019_LA_eval']:
        sasv_dataloader = get_dataloader(os.path.join(LA_dataset_path,data_dirnmae ))
        spk_emb_dic = {}
        spk_emb_norm_dic = {}
        with torch.no_grad():
            for idx, (data_1s, data_2s, files) in  tqdm.tqdm(enumerate(sasv_dataloader), total = len(sasv_dataloader)):
                data_1 = data_1s[0].cuda()
                data_2 = data_2s[0].cuda()
                file = files[0]

                embedding_1 = model(data_1)
                embedding_1_norm = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = model(data_2)
                embedding_2_norm  = F.normalize(embedding_2, p=2, dim=1)
                embedding = torch.cat([embedding_1, embedding_2], dim=0).detach().cpu().numpy()
                embedding_norm = torch.cat([embedding_1_norm, embedding_2_norm], dim=0).detach().cpu().numpy()
                spk_emb_dic[file] = embedding
                spk_emb_norm_dic[file] = embedding_norm
        os.makedirs(os.path.join(args.ASV_embedings_save_path, args.model), exist_ok=True)
        with open( f"{os.path.join(args.ASV_embedings_save_path, args.model)}/{data_dirnmae}_l2norm.pk", "wb") as f:
            pk.dump(spk_emb_norm_dic, f)
        with open( f"{os.path.join(args.ASV_embedings_save_path, args.model)}/{data_dirnmae}_nonorm.pk", "wb") as f:
            pk.dump(spk_emb_dic, f)
