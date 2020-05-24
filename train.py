import glob
import torch
import dataset
import numpy as np
from unet import UNet
import torch.nn as nn
from metrics import dice_loss
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import cv2
import random
from early_stopping import EarlyStopping
import matplotlib.pyplot as plt

es = EarlyStopping(patience = 5)

num_of_epochs = 30
criterion = torch.nn.BCEWithLogitsLoss()

def evaluate(teacher, val_loader):
    teacher.eval().cuda()

    #change to dice
    ll = []
    with torch.no_grad():
        for i,(img,gt) in enumerate(val_loader):

            # cv2.imshow("img", np.array(img))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    
            if torch.cuda.is_available():
                img, gt = img.cuda(), gt.cuda()
            img, gt = Variable(img), Variable(gt)
            output = teacher(img)
            cv2.imshow('img -> val', img.squeeze(0).cpu().numpy().transpose(2, 1, 0))
            cv2.imshow('gt -> val', gt.squeeze(0).cpu().numpy().transpose(2, 1, 0))
            cv2.imshow('op -> val', output.squeeze(0).cpu().detach().numpy().transpose(2, 1, 0))
            cv2.waitKey(10)
            output = output.clamp(min = 0, max = 1)
            gt = gt.clamp(min = 0, max = 1)
            loss = criterion(output, gt[:, 0, :, :].unsqueeze(0))
            #ll.append(loss.item())

    
    #mean_dice = np.mean(ll)
    print('Eval metrics:\n\tAverabe Dice loss:{}'.format(loss.item()))
    return loss


def train(teacher, optimizer, train_loader):
    print(' --- teacher training')
    teacher.train().cuda()
    ll = []
    for i, (img, gt) in enumerate(train_loader):
        if torch.cuda.is_available():
            img, gt = img.cuda(), gt.cuda()
        
        img, gt = Variable(img), Variable(gt)

        output = teacher(img)
        cv2.imshow('img -> train', img.squeeze(0).cpu().numpy().transpose(2, 1, 0)*255)
        cv2.imshow('gt -> train', gt.squeeze(0).cpu().numpy().transpose(2, 1, 0))
        cv2.imshow('op -> train', output.squeeze(0).cpu().detach().numpy().transpose(2, 1, 0))
        cv2.waitKey(10)
        output = output.clamp(min = 0, max = 1)
        gt = gt.clamp(min = 0, max = 1)
        loss = criterion(output, gt[:, 0, :, :].unsqueeze(0))
        ll.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    
    mean_dice = np.mean(ll)

    print("Average loss over this epoch:\n\tDice:{}".format(mean_dice))
    return mean_dice

def set_seed(manual_seed):
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

if __name__ == "__main__":
    best = 1000
    set_seed(1)
    model = UNet(channel_depth = 32, n_channels = 3, n_classes=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    scheduler = StepLR(optimizer, step_size = 4, gamma = 0.2)

    #load teacher and student model

    #NV: add val folder
    train_list = glob.glob('/home/nirvi/NIO/diabetic-20200318T171807Z-001/diabetic/rgb/*jpg')
    val_list = glob.glob('/home/nirvi/NIO/diabetic-20200318T171807Z-001/diabetic/test_rgb/*jpg')
    print(len(val_list), 'val_list')

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #2 tensors -> img_list and gt_list. for batch_size = 1 --> img: (1, 3, 320, 320); gt: (1, 1, 320, 320)
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
        shuffle = False,
        transform = tf,
        ),
        batch_size = 1
    )


    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
        shuffle = False,
        transform = tf,
        ),
        batch_size = 1
    )    
    train_loss_values = []
    for epoch in range(num_of_epochs):
        print(' --- teacher training: epoch {}'.format(epoch+1))
        train_loss = train(model, optimizer, train_loader)
        #evaluate for one epoch on validation set
        val = evaluate(model, val_loader)
        train_loss_values.append(val)

        #if val_metric is best, add checkpoint
        if(True):
            #print("New Best!")
            #best = val.item()
            torch.save(model.state_dict(), 'checkpoints/CP_{}.pth'.format(epoch))
            print("Checkpoint {} saved!".format(epoch+1))

        if es.step(val):
            plt.plot(train_loss_values)
            print("Early Stopping . . .")
            break

        scheduler.step()