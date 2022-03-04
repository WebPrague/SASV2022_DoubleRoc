import os
import torch
import torch.nn as nn
import math
import torchaudio
import librosa
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class Bottle2neckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist blokc of a new stage.
        """
        super(Bottle2neckX, self).__init__()

        D = int(math.floor(planes * (baseWidth/64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D*C*scale, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C*scale)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(D*C, D*C, kernel_size=3, stride = stride, padding=1, groups=C, bias=False))
          bns.append(nn.BatchNorm2d(D*C))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(D*C*scale, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.width  = D*C
        self.stype = stype
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          # sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class Res2NeXt(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, net_scale, baseWidth=4, cardinality=8, n_mels=64, log_input=True, **kwargs):
      # Res2NeXt(Bottle2neckX, layers = [3, 4, 6, 3], baseWidth = 4, cardinality=8, scale = 4, num_classes=1000)
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
            scale: scale in res2net
        """
        super(Res2NeXt, self).__init__()

        self.inplanes  = num_filters[0]
        self.n_mels    = n_mels
        self.log_input = log_input

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.scale = net_scale

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)


        # self.num_classes = num_classes
        # self.inplanes = 64
        # self.output_size = 64

        # self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))
        
        self.instancenorm = nn.InstanceNorm1d(n_mels)
        # self.torchfb        = torch.nn.Sequential(
        #         PreEmphasis(),
        #         torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)
        #         )
        # self.torchfb = LogMelFeatGen2(512, 400, 160, 80, 16000)
        
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=64),
            )

        self.specaug = FbankAug()

        # 调试
        outmap_size = int(self.n_mels/8) # 8
        self.attention = nn.Sequential( # 128 * 8 * 4
            nn.Conv1d(num_filters[3] * outmap_size * self.scale, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size * self.scale, kernel_size=1),
            nn.Softmax(dim=2),
            )
        out_dim = num_filters[3] * outmap_size * 2 * self.scale
        # self.avgpool = nn.AdaptiveAvgPool2d(1)  
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.pooling_bn = nn.BatchNorm1d(out_dim)
        self.fc = nn.Linear(out_dim, nOut)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample, scale=self.scale, stype='stage'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x, aug=False):
        # 提取声学特征
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)
            x = x.unsqueeze(1)

        x = self.conv1(x) # [b, 32, 64, 202]
        x = self.relu(x)
        x = self.bn1(x)
      
        x = self.layer1(x) # b, 128, 64, 202
        x = self.layer2(x) # b, 256, 32, 101
        x = self.layer3(x) # b, 512, 16, 51
        x = self.layer4(x) # b, 1024, 8, 26

        x = x.reshape(x.size()[0],-1,x.size()[-1]) # 5, 4096, 26

        # pooling start
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)  # 均值
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )  # 方差
        x = torch.cat((mu,sg),1)
        x = self.pooling_bn(x)
        # pooling end

        x = self.fc(x)
        return x

# def res2next50(pretrained=False, **kwargs):
#     """    Construct Res2NeXt-50.
#     The default scale is 4.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = Res2NeXt(Bottle2neckX, layers = [3, 4, 6, 3], baseWidth = 4, cardinality=8, scale = 4, num_classes=1000)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['res2next50']))
#     return model


# (self, block, baseWidth, cardinality, layers, num_classes, scale=4)
def MainModel(model_file_path):
    nOut = 256
    net_scale = 4
    num_filters = [32, 64, 128, 128]
    model = Res2NeXt(Bottle2neckX, [3, 4, 6, 3], num_filters, nOut, net_scale,)
    resume_spk_model(model_file_path, model)
    return model

def resume_spk_model(model_file_path, network):
    model_dict = torch.load(model_file_path, map_location='cpu')
    network.load_state_dict(model_dict, strict=True)
    print(f'spk model load {os.path.split(model_file_path)[-1]}')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = MainModel('./weights/res2next50.model').cuda()
    inp = torch.randn(3, 16000).cuda()
    oup = model(inp)
    print(oup.shape)