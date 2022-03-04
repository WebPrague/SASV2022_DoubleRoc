import os
import torch
import numpy as np
from glob import glob
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class SASVTestDataset(Dataset):
    def __init__(self, eval_path, num_frames=300):
        super().__init__()
        self.eval_path = eval_path
        setfiles = glob(f'{eval_path}/flac/*.flac')
        setfiles = [os.path.split(setfile)[-1].replace('.flac', '') for setfile in setfiles]
        self.setfiles = setfiles
        self.num_frames = num_frames
    
    def __len__(self):
        return len(self.setfiles)
    
    def __getitem__(self, index):
        # return super().__getitem__(index)
        file = self.setfiles[index]
        audio_path = '{}/flac/{}.flac'.format(self.eval_path, file)
        audio, _  = sf.read(audio_path)
        # Full utterance 完整的一句
        data_1 = torch.FloatTensor(np.stack([audio],axis=0))

        # Spliited utterance matrix 将语音切片
        max_audio = self.num_frames * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
        feats = np.stack(feats, axis = 0).astype(np.float)
        data_2 = torch.FloatTensor(feats)
        return data_1, data_2, file

def get_dataloader(eval_path):
    sasv_dataset = SASVTestDataset(eval_path)
    sasv_dataloader = DataLoader(dataset=sasv_dataset, batch_size=1, num_workers=4)
    return sasv_dataloader