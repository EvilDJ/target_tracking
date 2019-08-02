import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def txt_(txt):
    dd = []
    with open(txt, 'r') as f:
        for i in f:
            i = i.strip().split(',')
            jj = []
            for j in i:
                jj.append(int(float(j)))
            dd.append(jj)
    return np.array(dd)

ground = './data/groundtruth_rect.txt'
lstm = './data/lstm.txt'
stc = './data/position.txt'

g = txt_(ground)
l = txt_(lstm)
s = txt_(stc)

for dir, file, file_ in os.walk('G:\DATA\OTB100\OTB100\OTB100\Basketball\img'):
    for num in range(len(file_)):
        path = dir + '\\' + file_[num]
        img = cv2.imread(path)
        g_ = g[num]
        l_ = l[num]
        s_ = s[num]
        cv2.rectangle(img, (g_[0], g_[1]), (g_[2] + g_[0], g_[1] + g_[3]), (255, 0, 0), 2)#蓝色标准
        cv2.rectangle(img, (l_[0], l_[1]), (l_[2] + l_[0], l_[1] + l_[3]), (0, 0, 255), 2)#红色本人
        cv2.rectangle(img, (s_[0], s_[1]), (s_[2] + s_[0], s_[1] + s_[3]), (0, 255, 0), 2)#绿色对比
        tx = '#' + str(num)
        cv2.putText(img, tx, (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0 ,(0, 0, 255), 2)
        cv2.imshow('img', img)#
        cv2.waitKey(250)
        cv2.destroyAllWindows()
