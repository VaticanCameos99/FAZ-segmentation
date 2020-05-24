import torch
import numpy as np
from PIL import Image
import random
import cv2

def crop(image, heatmap):
    image = np.array(image)
    heatmap = np.array(heatmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    gt = heatmap
    height, width, _ = image.shape

    SCALE =  [1.,0.9, 0.85, 0.8,0.75,0.7, 0.6, 0.65, 0.5, 0.55, 0.4, 0.45, 0.3, 0.35, 0.2, 0.1]
    # print(heatmap.max(), '1st max')
    # print(heatmap.min(), '1st min')

    for _ in range(1000):
        scale = random.randrange(len(SCALE))
        scale = SCALE[scale]
        short_side = min(height, width)
        w = min(int(scale*short_side) - int(scale*short_side)%4,width)
        h = w
        l = random.randrange(0,width-w + 1)
        left = l - (l%4)
        t = random.randrange(0,height-h + 1) # changed height-t to height-h
        top = t - (t%4)
        crop_rgn = [left, top, left+w, top+h]
        crop_im = image[crop_rgn[1]:crop_rgn[3], crop_rgn[0]:crop_rgn[2]]
        heatmap = heatmap[int(crop_rgn[1]):int(crop_rgn[3]), int(crop_rgn[0]):int(crop_rgn[2])]
        c0, c1 = np.where(heatmap>=250)
        # print(heatmap.max(), '2nd max')
        # print(heatmap.min(), '2nd min')
        # print(c0, 'c0')
        # print(c1, 'c1')
        # exit(1)
        if len(c0) >= 1:
            #crop_im = cv2.resize(crop_im,(800,800),cv2.INTER_LINEAR)
            #hmap_img = cv2.resize(heatmap.astype(np.float32),(800,800),cv2.INTER_NEAREST)
            hmap_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
            return crop_im, hmap_img
        else:
            heatmap = gt
            #heatmap[np.where(heatmap>0)]=1
            continue

    #image = cv2.resize(image, (800, 800), cv2.INTER_LINEAR)
    #heatmap = cv2.resize(heatmap.astype(np.float32), (800, 800), cv2.INTER_NEAREST)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    return image, heatmap

def flip(image, heatmap):
    image = np.fliplr(image)
    heatmap = np.fliplr(heatmap)
    return image, heatmap

def distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()
    if random.randrange(2):
        #brightness distortion
        # if random.randrange(2):
        #     _convert(image, beta=random.uniform(-32, 32))
        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
        #hue distortion
        # if random.randrange(2):
        #     tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        #     tmp %= 180
        #     image[:, :, 0] = tmp
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    else:
        #brightness distortion
        # if random.randrange(2):
        #     _convert(image, beta=random.uniform(-32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
        #hue distortion
        # if random.randrange(2):
        #     tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        #     tmp %= 180
        #     image[:, :, 0] = tmp
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))
    return image

def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    
    for X_img in X_imgs:
        X_img = np.array(X_img, np.float32)
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        gaussian_img = cv2.addWeighted(X_img, 0.005, gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    return gaussian_noise_imgs

def load_data(img_path):
    if img_path.find("/rgb/") != -1:
        gt_path = img_path.replace("/rgb/", "/masks/").replace("/image", "/label")
    else:
        gt_path = img_path.replace("/test_rgb/", "/test_masks/").replace("/image", "/label")
    
    img = Image.open(img_path)
    gt = Image.open(gt_path)
    if img_path.find("/rgb/") != -1:
        img, gt = crop(img, gt)
        i = img
        #chance = np.random.random()
        # if chance < 0.0:     
        #     img, gt = flip(img, gt)
        # if chance < 0.2:
        #     img = distort(img)
        if random.randrange(2):
            value = random.randrange(5)
            img[:,:,2] += value
            img = np.clip(img, 0, 255)
            # cv2.imshow( "brightness img",img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        if random.randrange(2):
            img = add_gaussian_noise([img])[0]
            # cv2.imshow("img", np.array(i))
            # cv2.imshow( "noise img",img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # exit(1)
    else:
        img = np.array(img)
        gt = np.array(gt)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, gt
    #add data aug functions
    #return img
        
