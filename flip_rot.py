import cv2
import glob
import argparse

images_list = glob.glob('/home/nirvi/NIO/diabetic-20200318T171807Z-001/diabetic/rgb/*.jpg')


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Get method')
parser.add_argument("--flip", help="specify flip ",type=str2bool)
parser.add_argument("--rot", help="specify rot ",type=str2bool)

def rotate(img, name):
    img = cv2.flip(img, cv2.ROTATE_90_CLOCKWISE)
    name = name.replace(".jpg", "_rot2.jpg")
    return img, name

def flip(img, name):
    img = cv2.flip(img, 1)
    name = name.replace(".jpg", "_flip2.jpg")
    return img, name

if __name__ == "__main__":
    args = parser.parse_args()
    for i in images_list:
        img = cv2.imread(i)
        m = i.replace("/rgb/", "/masks/").replace("/image", "/label")
        mask = cv2.imread(m)
        if(args.flip):
            img, i = flip(img, i)
            mask, m = flip(mask, m)
        if(args.rot):
            img, i = rotate(img, i)
            mask, m = rotate(mask, m)
        
        cv2.imwrite(i, img)
        cv2.imwrite(m, mask)

    print("Done")
