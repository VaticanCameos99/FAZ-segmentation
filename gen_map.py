from BoxCouning import rgb2gray, fractal_dimension
import numpy as np
import scipy.misc
#from scipy.misc import toimage
import cv2
import glob
from PIL import Image

src = glob.glob('/home/nirvi/NIO/diabetic-20200318T171807Z-001/diabetic/test_images/*.jpg')


def main(OrigImg):
    #	I = rgb2gray(OrigImg)
    I = OrigImg
    w = 5
    height = I.shape[0]
    width = I.shape[1]
    I = np.array(I)
    if(height < width):
        temp = height
        height = width
        width = temp

    hpad =(height %w)
    wpad =(width % w)
    h = 0
    wid = 0
    I = np.lib.pad(I, ((hpad, hpad), (wpad, wpad)), 'constant', constant_values = 0)
    ICopy = I
    maxFrac = 0

    while h+w < height:
        while wid+w < width:
            window = I[wid:wid+w, h:h+w]
            val = fractal_dimension(window)
            #print(window)
            if(np.isnan(val)):
                val = 0
            if(maxFrac < val):
                maxFrac = val;
            ICopy[int(wid+w/2), int(h+w/2)] = val #fractal number is added to the center of each window. Is this the center?
        #	print("ICopy: ", ICopy[wid:wid+w, h:h+w])
            wid = wid+1
        h = h+1
    ICopy = np.divide(ICopy, maxFrac)
    r = np.amax(ICopy);

    #dividing into 3 colours
    dif = r/6;
    a = np.arange(0, r, dif)
    #scipy.misc.imshow(ICopy)
    IImg = ICopy

    #converting 2d to 3d
    IImg = np.expand_dims(IImg, axis=-1)
    """IImg = np.append(IImg, IImg, axis = 2)
    IImg = np.append(IImg, IImg[:,:,0], axis = 2)"""
    IImg = np.atleast_3d(IImg)
    b = np.zeros([IImg.shape[0], IImg.shape[1]], dtype = float)
    b = np.expand_dims(b, axis =-1)
    IImg = np.append(IImg, b, axis = 2)
    IImg = np.append(IImg, b, axis = 2)
    IImg[:,:,0] = 0
    #IImg[:,:,1] = 0
    #IImg[:,:,2] = 0
    #cv2.imshow("IImg", IImg)
    #cv2.imshow("ICopy", ICopy)
    max = np.amax(ICopy)
    #print("max: ", max)
    contour = gray2rgb(ICopy, IImg, a);
    cv2.imshow("contour", contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # name = name.split("/")[-1]
    # name = name.replace(".jpg", "_map.jpg")
    # cv2.imwrite(name, contour)
    return contour
    #try multithreading on sudb code


def gray2rgb(copy, gray, a):
	i = 0
	j = 0
	max1 = np.amax(copy)
	#cv2.imshow("Copy", copy)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	while i < copy.shape[0]:
		j = 0
		while j < copy.shape[1]:
			if copy[i][j] > a[0] and copy[i][j] < a[1]:
				gray[i][j][2] = 0;	#b -> 0, g -> 1, r -> 2
				gray[i][j][0] = 255;
				gray[i][j][1] = 0;
			elif copy[i][j] > a[1] and copy[i][j] < a[2]:#light blue
				gray[i][j][1]= 204;
				gray[i][j][0] = 0;
				gray[i][j][2] = 0;
			elif copy[i][j] > a[2] and copy[i][j] < a[3]:#light green
				gray[i][j][1]= 153;
				gray[i][j][0] = 0;
				gray[i][j][2] = 0;
			elif copy[i][j] > a[3] and copy[i][j] < a[4]:#yellow
				gray[i][j][1]= 255;
				gray[i][j][0] = 0;
				gray[i][j][2] = 255;
			elif copy[i][j] > a[4] and copy[i][j] < a[5]:#orange
				gray[i][j][1]= 128;
				gray[i][j][0] = 0;
				gray[i][j][2] = 255;
			elif copy[i][j] > a[5]:
				gray[i][j][0]= 0;
				gray[i][j][2] = 255;
				gray[i][j][1] = 0;
			j= j+1

		i = i+1

	return gray

# if __name__ == "__main__":
#     for i in src:
#         print(i)
#         OrigImg = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
#         main(OrigImg, i)