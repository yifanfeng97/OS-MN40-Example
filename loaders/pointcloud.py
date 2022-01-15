import torch
import numpy as np
from torch.utils.data import Dataset


def aug_pt(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    aug_pt = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return aug_pt


def load_pt(root, augement=False, resolution=1024):
    filename = root/f'pt_{resolution}.txt'
    if not filename.exists():
        return None
    else:
        pt = np.loadtxt(str(filename))
        pt = pt - np.expand_dims(np.mean(pt, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(pt ** 2, axis=1)), 0)
        pt = pt / dist  # scale
        if augement:
            pt = aug_pt(pt)

        pt = torch.from_numpy(pt.astype(np.float32))
        return pt.transpose(0, 1)

if __name__ == '__main__':
    pass
