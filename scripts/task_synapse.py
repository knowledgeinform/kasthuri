#!/usr/bin/env python

import torch
from typing import Tuple, List
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils import data
import pathlib 
from kasthuri.models.unet import UNet
from kasthuri.trainer import Trainer
from torchvision import transforms
import json as json
from kasthuri.bossdbdataset import BossDBDataset
from datetime import datetime
import argparse
import os
import sys
import ssl
from tqdm import tqdm 
import segmentation_models_pytorch as smp
os.system('git clone https://github.com/black0017/MedicalZooPytorch/ && cd MedicalZooPytorch && mv lib ../. && cd .. && rm -rf MedicalZooPytorch')
sys.path.append(os.getcwd()) 

from torchsummary import summary
from loading_model import load_model

#This was necessary to overcome an SSL cert error when downloading pretrained weights for SMP baselines- your milelage may vary here
ssl._create_default_https_context = ssl._create_unverified_context

def train_model(task_config,network_config,boss_config=None,gpu='cuda'):
    torch.manual_seed(network_config['seed'])
    np.random.seed(network_config['seed'])
    if network_config['augmentations'] == 0:
        transform = transforms.Compose([transforms.ToTensor(),
                                ])
    if network_config['augmentations'] == 1:
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation((0,180)),
                                    ])

    train_data = BossDBDataset(
        task_config, boss_config, "train", image_transform = transform, mask_transform = transform)

    val_data =  BossDBDataset(
        task_config, boss_config, "val", image_transform = transform, mask_transform = transform)

    test_data =  BossDBDataset(
        task_config, boss_config, "test", image_transform = transform, mask_transform = transform)

    training_dataloader = data.DataLoader(dataset=train_data,
                                        batch_size=network_config['batch_size'],
                                        shuffle=True)
    validation_dataloader = data.DataLoader(dataset=val_data,
                                        batch_size=network_config['batch_size'],
                                        shuffle=True)
    test_dataloader = data.DataLoader(dataset=test_data,
                                        batch_size=network_config['batch_size'],
                                        shuffle=False)


    x, y = next(iter(training_dataloader))

    # Specify device
    device = torch.device('cuda') if torch.cuda.is_available() else  torch.device('cpu')

    model = load_model(network_config, device)

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    if network_config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=network_config["learning_rate"])
    if network_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=network_config["learning_rate"], betas=(network_config["beta1"],network_config["beta2"]))
    
    # trainer (I changed the epochs to 5 just to make it run faster)
    trainer = Trainer(model=model,
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    training_DataLoader=training_dataloader,
                    validation_DataLoader=validation_dataloader,
                    lr_scheduler=None,
                    epochs=network_config["epochs"],
                    epoch=0,
                    notebook=False)

    # start training
    training_losses, validation_losses, lr_rates = trainer.run_trainer()

    # save the model
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    model_name =  network_config['outweightfilename'] + '_' + task_config['task_type'] + '_' + date + '.pt'
    os.makedirs(pathlib.Path.cwd() / network_config['outputdir'], exist_ok = True) 
    torch.save(model.state_dict(), pathlib.Path.cwd() / network_config['outputdir'] / model_name)

    #take loss curves
    plt.figure()
    plt.plot(training_losses,label='training')
    plt.plot(validation_losses,label='validation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Learning Curve')
    plt.legend()
    model_name = model_name[:len(model_name)-3] + '_learning_curve.png'
    plt.savefig(pathlib.Path.cwd() / network_config['outputdir'] / model_name)

    def predict(img,
                model,
                device,
                ):
        model.eval()
        x = img.to(device)  # to torch, send to device
        with torch.no_grad():
            out = model(x)  # send through model/network

        out_argmax = torch.argmax(out, dim=1)  # perform softmax on outputs
        return out_argmax

    batch_iter = tqdm(enumerate(test_dataloader), 'test', total=len(test_dataloader), leave=False)
    # predict the segmentations of test set
    tp_tot = torch.empty(0,network_config['classes'])
    fp_tot = torch.empty(0,network_config['classes'])
    fn_tot = torch.empty(0,network_config['classes'])
    tn_tot = torch.empty(0,network_config['classes'])

    # first compute statistics for true positives, false positives, false negative and
    # true negative "pixels"
    for i, (x, y) in batch_iter:
        #input, target = x.to(device), y.to(device)  # send to device (GPU or CPU)
        target = y.to(device) #can do this on CPU

        with torch.no_grad():
            # get the output image
            output = predict(x, model, device)
            plt.figure()
            plt.imshow  (output[0,:,:]  )
            plt.show()
            tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multiclass', num_classes = network_config['classes'])
            tp_tot = torch.vstack((tp_tot,tp))
            fp_tot = torch.vstack((fp_tot,fp))
            fn_tot = torch.vstack((fn_tot,fn))
            tn_tot = torch.vstack((tn_tot,tn))


    # then compute metrics with required reduction (see metric docs)
    model_name = model_name[:len(model_name)-19] + '_report.rpt'
    rh = open(pathlib.Path.cwd() / network_config['outputdir'] / model_name, 'w')
 
    #Accuracy Per Class 
    acc = (tp_tot.mean(dim=0)+tn_tot.mean(dim=0))/(fp_tot.mean(dim=0)+tn_tot.mean(dim=0)+fn_tot.mean(dim=0)+tp_tot.mean(dim=0))
    print('Accuracy per Class:')
    print(np.array(acc.cpu()))
    rh.write('Accuracy per Class:\n')
    rh.write(str(np.array(acc.cpu())))
    
    spec =  (tn_tot[:,1:].mean())/(fp_tot[:,1:].mean()+tn_tot[:,1:].mean())
    sens =  (tp_tot[:,1:].mean())/(fn_tot[:,1:].mean()+tp_tot[:,1:].mean())
    balacc = (spec + sens)/2
    print(f'Balanced accuracy (No background): {balacc}')
    rh.write(f'Balanced accuracy (No background): {balacc}\n')
    
    prec = tp_tot.mean(dim=0)/(fp_tot.mean(dim=0)+tp_tot.mean(dim=0))
    reca = tp_tot.mean(dim=0)/(fn_tot.mean(dim=0)+tp_tot.mean(dim=0))
    f1 = (2*reca*prec)/(reca+prec)
    print(f'F1-score: {np.array(f1.cpu())} Avg. F1-score: {f1.mean()}')
    rh.write(f'F1-score: {np.array(f1.cpu())} Avg. F1-score: {f1.mean()}\n')

    iou = (tp_tot.mean(0))/(fp_tot.mean(0)+fn_tot.mean(0)+tp_tot.mean(0))
    print(f'IoU: {np.array(iou.cpu())} Avg. IoU-score: {iou.mean()}')
    rh.write(f'IoU: {np.array(iou.cpu())} Avg. IoU-score: {iou.mean()}\n')

    rh.close()


if __name__ == '__main__':
    # usage python3 task2_2D_smp_main.py --task task2.json --network network_config_smp.json --boss boss_config.json
    parser = argparse.ArgumentParser(description='flags for training')
    parser.add_argument('--task', default="kasthuri/taskconfig/synapse_task.json",
                        help='task config json file')
    parser.add_argument('--network', default="kasthuri/networkconfig/UNet_2D.json",
                        help='network config json file')
    parser.add_argument('--boss', 
                        help='boss config json file')
    parser.add_argument('--gpu', 
                        help='index of the gpu to use')
    args = parser.parse_args()
    
    if args.gpu:
        gpu = 'cuda:'+args.gpu
    else:
        gpu = 'cuda'

    task_config = json.load(open(args.task))
    network_config = json.load(open(args.network))
    if args.boss:
        boss_config = json.load(open(args.boss))
    else:
        boss_config = None
    print('begining training')
    train_model(task_config,network_config,boss_config,gpu)
