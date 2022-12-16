# -*- coding: utf-8 -*-
import torch
from torch.backends import cudnn, cuda, mps, openmp, mkldnn

if __name__ == '__main__':
    print(torch.backends.cuda.is_built())
    print(torch.backends.cudnn.is_available())
    print(torch.backends.cudnn.version())
    print(torch.backends.cudnn.enabled)
    print(torch.backends.cudnn.allow_tf32)
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())
    print(torch.backends.openmp.is_available())
    print(torch.backends.mkldnn.is_available())