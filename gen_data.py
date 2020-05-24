from gen_map import main as map_generator
import glob
import cv2

src = glob.glob('/home/nirvi/NIO/diabetic-20200318T171807Z-001/diabetic/images/*.jpg')

if __name__ == "__main__":
    for i in src:
        print(i)
        OrigImg = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        rgb = map_generator(OrigImg)
        i = i.replace("images", "rgb")
        cv2.imwrite(i, rgb)