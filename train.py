import time
from apex import amp
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import fft

from config import Config
from model.metric import *
from model.model import MMNet
from model.loss import weighted_CrossEntropyLoss
from data_loaders import *
from utils.util import *
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

training_files = ['']

cfg = Config()
if not os.path.exists(cfg.model_pth):
    os.makedirs(cfg.model_pth)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build model+optimizer
model = MMNet().to(device)
model.apply(weights_init_normal)
epoch = 1

optimizer = torch.optim.Adam(
    model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, amsgrad=True)

model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

# data
train_loader, test_loader, data_count = train_loader_generator(
    training_files, cfg.batch_size)

weights_for_each_class = calc_class_weight(data_count)


train_loss = MetricTracker()
train_acc = MetricTracker()
train_fscore = MetricTracker()
train_kappa = MetricTracker()

test_loss = MetricTracker()
test_acc = MetricTracker()
test_fscore = MetricTracker()
test_kappa = MetricTracker()

st_time = time.time()
for epoch in range(1, cfg.epochs+1):
    model.train()
    train_loss.reset()
    train_acc.reset()
    train_fscore.reset()
    train_kappa.reset()

    for data, target in tqdm(train_loader):
        t_domain, f_domain, target = data[0].to(device), data[1].to(device), target.to(device)
        optimizer.zero_grad()
        fin = model(t_domain, f_domain)
        loss = weighted_CrossEntropyLoss(
            fin, target, weights_for_each_class, device)

        with amp.scale_loss(loss, optimizer) as scale_loss:
            scale_loss.backward()
        optimizer.step()

        train_loss.update(loss.item())
        train_acc.update(accuracy(fin, target))
        train_fscore.update(f1(fin, target))
        train_kappa.update(kappa(fin, target))

    print(f'Epoch: {epoch:03d} / {cfg.epochs:03d}')
    print(f'Loss: {train_loss.avg:.4f} | Acc: {train_acc.avg:.4f} | F1: {train_fscore.avg:.4f} | Kappa: {train_kappa.avg:.4f} ')

    torch.save(model.state_dict(), os.path.join(
        cfg.model_pth, '{0}.pth'.format(epoch)))



    model.eval()
    test_loss.reset()
    test_acc.reset()
    test_fscore.reset()
    test_kappa.reset()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            t_domain, f_domain, target = data[0].to(device), data[1].to(device), target.to(device)
            fin = model(t_domain, f_domain)
            loss = weighted_CrossEntropyLoss(
                fin, target, weights_for_each_class, device)
            
            test_loss.update(loss.item())
            test_acc.update(accuracy(fin, target))
            test_fscore.update(f1(fin, target))
            test_kappa.update(kappa(fin, target))
