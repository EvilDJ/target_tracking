import sys
import glob
import os
import cv2
import numpy as np

# vpath = './images/demo-sequences/vot15_bag/vot_bag/imgs/'
# r = 0
# print(glob.glob(os.path.join(vpath, "*.jpg")))
# for img in os.path.join(vpath, "*.jpg"):
#     print(r,img)
#     r += 1
#     cv2.imread(img)

print(round(212.132206692281))

def fib(max):
    n,a,b=0,0,1
    while(n<max):
        yield a
        a,b=b,a+b
        n+=1

g=fib(5)
print(g.__next__(),g.__next__(),g.__next__(),g.__next__(),g.__next__())
