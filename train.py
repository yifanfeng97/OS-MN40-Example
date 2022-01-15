import os
import time
import json
import torch
import random
import numpy as np
import scipy.spatial
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader

from models import UniModel
from loaders import OSMN40_train
from utils import split_trainval, AverageMeter, res2tab, acc_score, map_score
os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'

######### must config this #########
data_root = 'data/OS-MN40'
####################################

# configure
n_class = 8
n_worker = 16
max_epoch = 150
batch_size = 64
this_task = f"OS-MN40_{time.strftime('%Y-%m-%d-%H-%M-%S')}"

# log and checkpoint
out_dir = Path('cache')
save_dir = out_dir/'ckpts'/this_task
save_dir.mkdir(parents=True, exist_ok=True)

def setup_seed():
    seed = time.time() % 1000_000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f"random seed: {seed}")


def train(data_loader, net, criterion, optimizer, epoch):
    print(f"Epoch {epoch}, Training...")
    net.train()
    loss_meter = AverageMeter()
    all_lbls, all_preds = [], []

    st = time.time()
    for i, (img, mesh, pt, vox, lbl) in enumerate(data_loader):
        img = img.cuda()
        mesh = [d.cuda() for d in mesh]
        pt = pt.cuda()
        vox = vox.cuda()
        lbl = lbl.cuda()
        data = (img, mesh, pt, vox)

        optimizer.zero_grad()
        out = net(data)
        out_img, out_mesh, out_pt, out_vox = out
        out_obj = (out_img + out_mesh + out_pt + out_vox)/4
        loss = criterion(out_obj, lbl)
        # loss = criterion(out_pt, lbl) + criterion(out_vox, lbl)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(out_obj, 1)
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
        loss_meter.update(loss.item(), lbl.shape[0])
        print(f"\t[{i}/{len(data_loader)}], Loss {loss.item():.4f}")

    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")
    print(f"Epoch: {epoch}, Time: {time.time()-st:.4f}s, Loss: {loss_meter.avg:4f}")
    res = {
        "overall acc": acc_mi,
        "meanclass acc": acc_ma
    }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Epoch Done!\n")


def validation(data_loader, net, epoch):
    print(f"Epoch {epoch}, Validation...")
    net.eval()
    all_lbls, all_preds = [], []
    fts_img, fts_mesh, fts_pt, fts_vox = [], [], [], []

    st = time.time()
    for img, mesh, pt, vox, lbl in data_loader:
        img = img.cuda()
        mesh = [d.cuda() for d in mesh]
        pt = pt.cuda()
        vox = vox.cuda()
        lbl = lbl.cuda()
        data = (img, mesh, pt, vox)

        out, ft = net(data, global_ft=True)
        out_img, out_mesh, out_pt, out_vox = out
        ft_img, ft_mesh, ft_pt, ft_vox = ft
        out_obj = (out_img + out_mesh + out_pt + out_vox)/4

        _, preds = torch.max(out_obj, 1)
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
        fts_img.append(ft_img.detach().cpu().numpy())
        fts_mesh.append(ft_mesh.detach().cpu().numpy())
        fts_pt.append(ft_pt.detach().cpu().numpy())
        fts_vox.append(ft_vox.detach().cpu().numpy())

    fts_img = np.concatenate(fts_img, axis=0)
    fts_mesh = np.concatenate(fts_mesh, axis=0)
    fts_pt = np.concatenate(fts_pt, axis=0)
    fts_vox = np.concatenate(fts_vox, axis=0)
    fts_uni = np.concatenate((fts_img, fts_mesh, fts_pt, fts_vox), axis=1)
    dist_mat = scipy.spatial.distance.cdist(fts_uni, fts_uni, "cosine")
    map_s = map_score(dist_mat, all_lbls, all_lbls)
    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")
    print(f"Epoch: {epoch}, Time: {time.time()-st:.4f}s")
    res = {
        "overall acc": acc_mi,
        "meanclass acc": acc_ma,
        "map": map_s
    }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Epoch Done!\n")
    return map_s, res


def save_checkpoint(val_state, res, net: nn.Module):
    state_dict = net.state_dict()
    ckpt = dict(
        val_state=val_state,
        res=res,
        net=state_dict,
    )
    torch.save(ckpt, str(save_dir / 'ckpt.pth'))
    with open(str(save_dir / 'ckpt.meta'), 'w') as fp:
        json.dump(res, fp)


def main():
    setup_seed()
    # init train_loader and val_loader
    print("Loader Initializing...\n")
    train_list, val_list = split_trainval(data_root)
    train_data = OSMN40_train('train', train_list)
    val_data = OSMN40_train('val', val_list)
    print(f'train samples: {len(train_data)}')
    print(f'val samples: {len(val_data)}')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=n_worker, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                             num_workers=n_worker)
    print("Create new model")
    net = UniModel(n_class)
    net = net.cuda()
    net = nn.DataParallel(net)

    optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, 
                                                        eta_min=1e-4)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    best_res, best_state = None, 0
    for epoch in range(max_epoch):
        # train
        train(train_loader, net, criterion, optimizer, epoch)
        lr_scheduler.step()
        # validation
        if epoch != 0 and epoch % 5 == 0:
            with torch.no_grad():
                val_state, res = validation(val_loader, net, epoch)
            # save checkpoint
            if val_state > best_state:
                print("saving model...")
                best_res, best_state = res, val_state
                save_checkpoint(val_state, res, net.module)

    print("\nTrain Finished!")
    tab_head, tab_data = res2tab(best_res)
    print(tab_head)
    print(tab_data)
    print(f'checkpoint can be found in {save_dir}!')
    return best_res


if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
