import torch
import cv2
from gen_map import main as map_generator
import argparse
from unet import UNet
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='Get input')
parser.add_argument("--i", help="path to input image ", required = True)

def evaluate(teacher, img):
    teacher.eval().cuda()

    #change to dice
    with torch.no_grad():
        if torch.cuda.is_available():
            img = img.cuda()
        img = Variable(img).unsqueeze(0).float()
        output = teacher(img)
        o = output.squeeze(0).cpu().detach().numpy().transpose(2, 1, 0)
        cv2.imshow('op -> val', o)
        cv2.waitKey(0)
        cv2.imwrite("output.jpg", o)
        #output = output.clamp(min = 0, max = 1)
        #gt = gt.clamp(min = 0, max = 1)
        #loss = criterion(output, gt[:, 0, :, :].unsqueeze(0))
        #ll.append(loss.item())

    
    #mean_dice = np.mean(ll)
    #print('Eval metrics:\n\tAverabe Dice loss:{}'.format(mean_dice))


if __name__ == "__main__":
    teacher = UNet(channel_depth = 32, n_channels = 3, n_classes=1)
    teacher.load_state_dict(torch.load('checkpoints/CP_5.pth'))

    args = parser.parse_args()
    img_path = args.i
    #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #img = map_generator(img)
    #img_path = img_path.split("/")[-1]
    #img = cv2.imwrite(img_path, img)

    img = Image.open(img_path)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    cv2.imshow("img", np.array(img))

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = tf(img)
    evaluate(teacher, img)
    #apply box counting


