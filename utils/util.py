import os
import numpy as np
from glob import glob
import math
import torch
import torch.nn as nn
from scipy.fftpack import fft


def load_folds_data(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    if "78" in np_data_path:
        r_p_path = r"utils/r_permute_78.npy"
    else:
        r_p_path = r"utils/r_permute_20.npy"

    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
    else:
        print("============== ERROR =================")

    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1]
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs)
    files_pairs = files_pairs[r_permute]

    train_files = np.array_split(files_pairs, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        subject_files = [item for sublist in subject_files for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        training_files = list(set(files_pairs2) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data


def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5]

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
