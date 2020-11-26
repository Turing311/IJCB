import os
import numpy as np
import cv2
import sys
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributions
from utils import get_dataset_list, CustomFaceDataset, cycle, get_val_hter, TPC_loss
from vgg_face_dag import vgg_face_dag, spoof_model
import pdb
import torchvision.utils as tutil
import argparse
import torchvision
from parameters import *
import time
import scipy.io
import mfn

before_model_name = 'ckpt'
parser = argparse.ArgumentParser()
args = parser.parse_args()

np.set_printoptions(formatter='10.3f')
torch.set_printoptions(sci_mode=False, threshold=5000)

# for Fixing the seed
# torch.manual_seed(0)
#
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#
# np.random.seed(0)

# Load train data
train_real_list, num_classes = get_dataset_list('easen', 'real', 'train')
train_attack_list, num_classes1 = get_dataset_list('easen', 'attack', 'train')

if len(train_real_list) %2 != 0:
    del train_real_list[-1]
if len(train_attack_list) % 2 != 0:
    del train_attack_list[-1]
print("No of training real samples: {0}, attack samples: {1}, num_classes: {2}".format(len(train_real_list), len(train_attack_list), num_classes) )

# Load test data
devel_real_list, test_num_classes = get_dataset_list('easen', 'real', 'devel')
devel_attack_list, test_num_classes1 = get_dataset_list('easen', 'fake', 'devel')

if len(devel_real_list) % 2 != 0:
    del devel_real_list[-1]
if len(devel_attack_list) % 2 != 0:
    del devel_attack_list[-1]
print("No of real samples(target): {0}, attack samples(target): {1}, number of dev classes: {2}".format(len(devel_real_list), len(devel_attack_list), test_num_classes))



# load model
face_model = mfn.MfnModel().cuda()
face_model.load_state_dict( torch.load('mfn_2510.pth'))

# set eval mode
face_model.eval()

spoof_classifier = spoof_model(run_parameters['dimension'])
spoof_classifier.train()

if run_parameters['multi_gpu']:
    face_model = nn.DataParallel(face_model)
    spoof_classifier = nn.DataParallel(spoof_classifier)


if torch.cuda.is_available():
    face_model.cuda()
    spoof_classifier.cuda()

spoof_optim = optim.Adam(spoof_classifier.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)

inm = nn.InstanceNorm1d(1, affine=False)

live_faces_dataset = CustomFaceDataset(train_real_list)
live_face_loader = DataLoader(live_faces_dataset, batch_size=run_parameters['train_batch_size'], shuffle=True)

criterion = nn.CrossEntropyLoss()
criterion.cuda()
best_hter = 100.0

f = 0
for ep in range(run_parameters['epoch']):
    if ep % run_parameters['print_epoch'] == 0:
        hter = get_val_hter(face_model, spoof_classifier, devel_real_list, devel_attack_list, run_parameters['apply_inm'], test_num_classes, run_parameters['test_batch_size'])
        if hter < best_hter:
            best_hter = hter
            f = 1
            save_name = os.path.join('models', before_model_name + '_spoof_classifier.pth')
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            torch.save(spoof_classifier.state_dict(), save_name)

        print("Epoch {0}>>> HTER: {1}, best ACER: {2}".format(ep, hter, best_hter))


    spoof_classifier.train()

    for it, (data, _, _) in enumerate(live_face_loader):
        data = data.cuda().float()

        spoof_optim.zero_grad()

        features = face_model(data)

        if run_parameters['apply_inm']:
            features = features.view(features.shape[0], 1, features.shape[-1])
            features = inm(features)
            features = features.view(features.shape[0], features.shape[-1])

        # spoof work
        if run_parameters['white_noise']:
            # push from origin
            sampler = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(run_parameters['dimension']).cuda(),  \
                                                                                 run_parameters['std_dev'] * torch.eye(run_parameters['dimension']).cuda())
        else:
            # push from shifted mean cluster
            if it == 0:
                old_mean = torch.zeros(run_parameters['dimension']).cuda()
            else:
                old_mean = mean_vector
            mean_vector = torch.mean(features, axis=0)
            # like running avg
            new_mean = run_parameters['alpha'] * old_mean + (1 - run_parameters['alpha']) * mean_vector

            if f == 1:
                save_name = os.path.join('models', before_model_name + '_mean_vector.npy')
                scipy.io.savemat(save_name, {'mean':new_mean})
                f = 0
            sampler = torch.distributions.multivariate_normal.MultivariateNormal(new_mean,  \
                                                                                 run_parameters['std_dev'] * torch.eye(run_parameters['dimension']).cuda())
        # sample from pseudo-negative gaussian
        noise = sampler.sample((features.shape[0],))
        noise = noise.cuda()
        spoof_input = torch.cat([features, noise], dim=0)

        spoof_output = spoof_classifier(spoof_input)

        spoof_label = torch.cat([torch.zeros(features.shape[0]), torch.ones(noise.shape[0])], dim=0)
        spoof_label = spoof_label.cuda()
        spoof_label = spoof_label.long()

        if run_parameters['use_pc']:
            # Calculate TPC loss
            tpc_loss = TPC_loss(features)

        # Spoof Loss for classifier
        spoof_loss = criterion(spoof_output, spoof_label)

        if run_parameters['use_pc']:
            loss = run_parameters['lambda1'] * tpc_loss + run_parameters['lambda2'] * spoof_loss
        else:
            loss = run_parameters['lambda2'] * spoof_loss

        loss.backward()

        spoof_optim.step()
        if it % 10 == 0:
            if run_parameters['use_pc']:
                print("TPC Loss: {0}, Spoof Loss: {1}, Total Loss: {2}".format(tpc_loss.item(), spoof_loss.item(), loss.item()))
            else:
                print("Spoof Loss: {0}, Total Loss: {1}".format(spoof_loss.item(), loss.item()))
