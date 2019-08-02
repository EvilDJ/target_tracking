from PIL import Image
import numpy as np
import os

dir_n=input('please input the operate label dir:')
file_list=os.listdir(dir_n)
for filename in file_list:
    path_n=dir_n + filename
    im=Image.open(path_n)
    print(np.unique(im))