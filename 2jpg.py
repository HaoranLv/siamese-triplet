import os
import cv2
import sys
import numpy as np
 
path = "/Users/lvhaoran/Downloads/bl-data-for-siamese-network/true_bl/"
path_out = "/Users/lvhaoran/Downloads/bl-data-for-siamese-network/true_bl/"
print(path)
 
for filename in os.listdir(path):
    if os.path.splitext(filename)[-1] == '.png' or os.path.splitext(filename)[-1] == '.jpeg':
        print(filename)
        img = cv2.imread(path + filename)
        print(filename.replace(os.path.splitext(filename)[-1],".jpg"))
        newfilename = filename.replace(os.path.splitext(filename)[-1],".jpg")
        # cv2.imshow("Image",img)
        # cv2.waitKey(0)
        cv2.imwrite(path_out + newfilename,img)
        os.remove(os.path.join(path,filename))
