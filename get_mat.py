import os
import time
import json
import torch
import random
import numpy as np
import scipy.spatial
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from models import UniModel
from loaders import OSMN40_retrive
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

######### must config this #########
data_root = Path('data/OS-MN40')
ckpt_path = 'cache/ckpts/OS-MN40_2022-01-12-20-57-46/ckpt.pth'
####################################

# configure
dist_mat_path = Path(ckpt_path).parent / "cdist.txt"
dist_metric='cosine'
batch_size = 64
n_worker = 16
n_class = 8


def extract(query_loader, target_loader, net):
    net.eval()
    print("Extracting....")

    q_fts_img, q_fts_mesh, q_fts_pt, q_fts_vox = [], [], [], []
    t_fts_img, t_fts_mesh, t_fts_pt, t_fts_vox = [], [], [], []

    st = time.time()
    for img, mesh, pt, vox in query_loader:
        pt = pt.cuda()
        vox = vox.cuda()
        data = (None, None, pt, vox)
        _, ft = net(data, global_ft=True)
        ft_img, ft_mesh, ft_pt, ft_vox = ft
        q_fts_pt.append(ft_pt.detach().cpu().numpy())
        q_fts_vox.append(ft_vox.detach().cpu().numpy())
    q_fts_pt = np.concatenate(q_fts_pt, axis=0)
    q_fts_vox = np.concatenate(q_fts_vox, axis=0)
    q_fts_uni = np.concatenate((q_fts_pt, q_fts_vox), axis=1)

    for img, mesh, pt, vox in target_loader:
        pt = pt.cuda()
        vox = vox.cuda()
        data = (None, None, pt, vox)
        _, ft = net(data, global_ft=True)
        ft_img, ft_mesh, ft_pt, ft_vox = ft
        t_fts_pt.append(ft_pt.detach().cpu().numpy())
        t_fts_vox.append(ft_vox.detach().cpu().numpy())
    t_fts_pt = np.concatenate(t_fts_pt, axis=0)
    t_fts_vox = np.concatenate(t_fts_vox, axis=0)
    t_fts_uni = np.concatenate((t_fts_pt, t_fts_vox), axis=1)

    print(f"Time Cost: {time.time()-st:.4f}")

    dist_mat = scipy.spatial.distance.cdist(q_fts_uni, t_fts_uni, dist_metric)
    np.savetxt(str(dist_mat_path), dist_mat)

def read_object_list(filename, pre_path):
    object_list = []
    with open(filename, 'r') as fp:
        for name in fp.readlines():
            if name.strip():
                object_list.append(str(pre_path/name.strip()))
    return object_list


def main():
    # init train_loader and test loader
    print("Loader Initializing...\n")
    query_list = read_object_list("query.txt", data_root / "query")
    target_list = read_object_list("target.txt", data_root / "target")
    query_data = OSMN40_retrive(query_list)
    target_data = OSMN40_retrive(target_list)
    print(f'query samples: {len(query_data)}')
    print(f'target samples: {len(target_data)}')
    query_loader = DataLoader(query_data, batch_size=batch_size, shuffle=False,
                                               num_workers=n_worker)
    target_loader = DataLoader(target_data, batch_size=batch_size, shuffle=False,
                                             num_workers=n_worker)
    print(f"Loading model from {ckpt_path}")
    net = UniModel(n_class)
    ckpt = torch.load(ckpt_path)
    net.load_state_dict(ckpt['net'])
    net = net.cuda()
    net = nn.DataParallel(net)

    # extracting
    with torch.no_grad():
        extract(query_loader, target_loader, net)

    print(f"cdis matrix can be find in path: {dist_mat_path.absolute()}")


if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
