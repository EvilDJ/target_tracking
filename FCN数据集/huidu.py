# coding = utf-8
from PIL import Image
import os

dir =input('please input the operate png dir:')
dir_n=input('please input the operate label dir:')
file_list=os.listdir(dir)
for filename in file_list:
    path = dir + filename
    path_n=dir_n + filename
    im=Image.open(path).convert('L')
    im.save(path_n)


