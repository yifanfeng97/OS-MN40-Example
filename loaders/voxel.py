import torch
import numpy as np
from .binvox_rw import read_as_3d_array

def aug_vox(vox):
    return vox

def load_vox(root, augement=False, resolution=32):
    filename = root/f'vox_{resolution}.binvox'
    if not filename.exists():
        return None
    else:
        with open(filename, 'rb') as fp:
            vox = read_as_3d_array(fp).data
        if augement:
            vox = aug_vox(vox)

        vox = torch.from_numpy(vox.astype(np.float32)).unsqueeze(0)
        return vox

if __name__ == '__main__':
    pass
