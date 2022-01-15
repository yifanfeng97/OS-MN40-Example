from pathlib import Path
from random import shuffle


def split_trainval(data_root: str, train_ratio=0.8):
    train_root = Path(data_root) / "train"
    cates = sorted([d.stem for d in train_root.glob('*') if d.is_dir()])
    train_list, val_list = [], []
    for idx, cate in enumerate(cates):
        samples = [d for d in (train_root / cate).glob('*') if d.is_dir()]
        shuffle(samples)
        len_train = int(len(samples) * train_ratio)
        assert len_train > 0
        for _i, sample in enumerate(samples):
            if _i < len_train:
                train_list.append({'path': str(sample.absolute()),'label': idx})
            else:
                val_list.append({'path': str(sample.absolute()),'label': idx})
    return train_list, val_list


def res2tab(res: dict, n_palce=4):
    def dy_str(s, l):
        return  str(s) + ' '*(l-len(str(s)))
    min_size = 8
    k_str, v_str = '', ''
    for k, v in res.items():
        cur_len = max(min_size, len(k)+2)
        k_str += dy_str(f'{k}', cur_len) + '| '
        v_str += dy_str(f'{v:.4}', cur_len) + '| '
    return k_str, v_str


class AverageMeter:
    def __init__(self):
        self.value = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


################################### metric #######################################
import scipy
import scipy.spatial
import numpy as np


def acc_score(y_true, y_pred, average="micro"):
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if average == "micro": 
        # overall
        return np.mean(y_true == y_pred)
    elif average == "macro":
        # average of each class
        cls_acc = []
        for cls_idx in np.unique(y_true):
            cls_acc.append(np.mean(y_pred[y_true==cls_idx]==cls_idx))
        return np.mean(np.array(cls_acc))
    else:
        raise NotImplementedError


def map_score(dist_mat, lbl_a, lbl_b, metric='cosine'):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        p = 0.0
        r = 0.0
        for j in range(n_b):
            if lbl_a[i] == lbl_b[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res.append(p/r)
        else:
            res.append(0)
    return np.mean(res)



if __name__ == "__main__":
    split_trainval('data/OS-MN40')
