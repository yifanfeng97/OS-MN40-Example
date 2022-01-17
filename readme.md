# Introduction
This is an example code for [track](https://shrec22.moon-lab.tech/) "Open-Set 3D Object Retrieval using Multi-Modal Representation" in [SHREC22](https://www.shrec.net/). The complete dataset OS-MN40 is adopted for input. Dataset can be download as follows:
- [OS-MN40](https://data.shrec22.moon-lab.tech:18443/OS-MN40.tar.gz)
- [OS-MN40-Miss](https://data.shrec22.moon-lab.tech:18443/OS-MN40-Miss.tar.gz)

More details about the dataset and the track can be found in [here](https://shrec22.moon-lab.tech/).

## Models
We implement the baseline via combining multi-modal backbone, as follows:
- image: resnet18 with 2-view input
- mesh: [MeshNet](https://github.com/iMoonLab/MeshNet)
- point cloud: [Pointnet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- voxel: [VoxNet](https://github.com/MonteYang/VoxNet.pytorch)

# Setup
## Install Related Packages
This example code is developed in Python 3.8.12 and pytorch1.8.1+cu102. You can install the required packages as follows.
``` bash 
pip install -r requirements.txt
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
```

## Configure Path
By default, the datasets are placed under the ""data"" folder in the root directory. This code will create a new folder (name depends on the current time) to restore the checkpoint files under ""cache/ckpts"" folder for each run.
``` bash
../
├── cache
│   └── ckpts
│       ├── OS-MN40_2022-01-12-20-57-46
│       │   ├── cdist.txt
│       │   ├── ckpt.meta
│       │   └── ckpt.pth
│       └── OS-MN40_2022-01-15-13-58-50
│           ├── cdist.txt
│           ├── ckpt.meta
│           └── ckpt.pth
└── data
    ├── OS-MN40/
    └── OS-MN40-Miss/
```
You can also place the datasets anywhere you want. Don't forget change the related path in "line 19 in train.py" and "line 19 in get_mat.py".

# Train and Validation
Run "train.py". By default, 80% data in train folder is used for training and the rest is used for validation.
``` bash
python train.py
```

# Submission
## Generate Distance Matrix
Modify the data_root and ckpt_path in "line 17-18 in get_mat.py". Then run:
``` bash
python get_mat.py
```
The generated cdist.txt can be found in the same folder of the specified checkpoints. 

## Online Evaluation
You can submit the cdist.txt file with your personal key on the track[website](https://shrec22.moon-lab.tech/). The submission with invalid personal will not appear in the leadboard. The online evaluation will evaluate mAP, NN, NDCG@100, and ANMRR. The details of those scores can be found in "utils.py". The defination of those scores refer to the book [View-Based 3-d Object Retrieval](https://www.sciencedirect.com/topics/computer-science/criterion-measure).

