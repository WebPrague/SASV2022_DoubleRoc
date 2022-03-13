import os
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import soundfile as sf
from torch import Tensor
import torch, time, sys, tqdm
from utils import get_all_EERs
from models.AASIST import MainModel as CMModel
from torch.utils.data import Dataset, DataLoader

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
        "--norm", type=str, default="l2norm"
    )
    parser.add_argument(
        "--gpu", type=str, default="0"
    )
    return parser.parse_args()

def init_cm_model(cm_weight_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cm_model = CMModel(cm_weight_path)
    cm_model = cm_model.to(device)
    cm_model = cm_model.eval()
    for module in cm_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.eval()
        elif isinstance(module, nn.BatchNorm1d):
            module.track_running_stats = False
            module.eval()
    return cm_model

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

class Dataset_devNeval(Dataset):
    def __init__(self, base_dir, spk2ids, items, asv_embd):
        self.base_dir = base_dir
        self.spk2ids = spk2ids
        self.items = items
        self.cut = 64600  # take ~4 sec audio (64600 samples), 这个对于CM系统可能需要修改
        self.asv_embd = asv_embd
        self.spk_asv_embd = {}
        for spk in spk2ids.keys():
            ids = spk2ids[spk]
            embds = []
            for id in ids:
                embds.append(asv_embd[id])
            embds = np.concatenate(embds, axis=0).mean(0)
            self.spk_asv_embd[spk] = embds
        for key in asv_embd:
            asv_embd[key] = asv_embd[key].mean(0)
        

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        spk, tst, label = self.items[index]
        X, _ = sf.read('{}/flac/{}.flac'.format(self.base_dir, tst))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return self.spk_asv_embd[spk], self.asv_embd[tst], x_inp, label, index

def gen_dev_and_eval_item_list(enroll_paths, meta_path):
    spk2ids = {}
    if isinstance(enroll_paths, list):
        for enroll_path in enroll_paths:
            with open(enroll_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    spk, ids = line.strip().split()
                    ids = ids.split(',')
                    spk2ids[spk] = ids
    else:
        with open(enroll_paths, 'r', encoding='utf-8') as fin:
            for line in fin:
                spk, ids = line.strip().split()
                ids = ids.split(',')
                spk2ids[spk] = ids
    items = []
    with open(meta_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            spk, tst, _, label = line.strip().split()
            items.append([spk, tst, label])
    return spk2ids, items


def gen_loader(database_path, embedding_path, batch_size, n_cpu, norm):

    dev_database_path  = database_path + "/ASVspoof2019_LA_dev/"
    eval_database_path = database_path + "/ASVspoof2019_LA_eval/"

    dev_trial_path  = database_path + "/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt"
    eval_trial_path = database_path + "/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt"
    
    # dev dataset setting
    dev_spk2ids, dev_items = gen_dev_and_eval_item_list([database_path + '/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.male.trn.txt',
                                                         database_path + '/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.female.trn.txt'],
                                                        dev_trial_path)
    dev_asv_embd = pickle.load(open(os.path.join(embedding_path, f'ASVspoof2019_LA_dev_{norm}.pk'), 'rb'))

    dev_dataset = Dataset_devNeval(base_dir=dev_database_path, spk2ids=dev_spk2ids, items=dev_items, asv_embd=dev_asv_embd)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=n_cpu)
    # eval dataset setting
    eval_spk2ids, eval_items = gen_dev_and_eval_item_list([database_path + '/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.male.trn.txt',
                                                           database_path + '/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.female.trn.txt'],
                                                          eval_trial_path)
    eval_asv_embd = pickle.load(open(os.path.join(embedding_path, f'ASVspoof2019_LA_eval_{norm}.pk'), 'rb'))

    eval_dataset = Dataset_devNeval(base_dir=eval_database_path, spk2ids=eval_spk2ids, items=eval_items, asv_embd=eval_asv_embd)
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             num_workers=n_cpu)
    return dev_loader, eval_loader

if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    print(f'Model: {args.model}')
    print(f'Norm: {args.norm}')
    assert args.norm in ['l2norm', 'nonorm']
    LA_dataset_path = args.LA_dataset_path
    ASV_embedings_save_path = args.ASV_embedings_save_path
    cm_model = init_cm_model('./weights/AASIST.model')
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    dev_loader, eval_loader = gen_loader(LA_dataset_path, os.path.join(ASV_embedings_save_path, args.model), batch_size=64, n_cpu=4, norm=args.norm)
    tag_2_loader = {'dev': dev_loader, 'eval': eval_loader}
    for tag in tag_2_loader.keys():
        loader = tag_2_loader[tag]
        id2score = {}
        id2label = {}
        scores_list, labels_list = [], []
        for idx, (asv_embd_enr, asv_embd_tst, fusion_cm_data, label, index) in enumerate(loader):
            with torch.no_grad():
                embd_cm, output = cm_model.forward(fusion_cm_data.to(device), aug=False)
                score1 = output.softmax(-1).detach().cpu().numpy()[:, 1]
                score2 = cos(asv_embd_enr.to(device), asv_embd_tst.to(device)).detach().cpu().numpy()
                score2 = (score2 + 1) / 2
                score = score1 * score2
                index = index.detach().cpu().numpy()
                    
                for i, l, s in zip(index, label, score):
                    id2label[i] = l
                    id2score[i] = s
        for i in range(len(id2score.keys())):
            scores_list.append(id2score[i])
            labels_list.append(id2label[i])
        sasv_EER, sv_EER, spf_EER = get_all_EERs(preds=scores_list, keys=labels_list)
        print("%s_sasvEER %2.2f%%, eval_svEER %2.2f%%, eval_spfEER %2.2f%%" % (tag, sasv_EER * 100, sv_EER * 100, spf_EER * 100))
    
