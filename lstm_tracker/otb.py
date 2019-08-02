from matplotlib import pyplot as plt
import cv2

img = cv2.imread(r'G:\DATA\OTB100\OTB100\OTB100\Basketball\img\0725.jpg')
ground = [323, 198, 34, 81]
grount = [330, 201, 34, 81]

# 两个检测框框是否有交叉，如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
def bb_overlab(x, y):
    # if(x1>x2+w2):
    #     return 0
    # if(y1>y2+h2):
    #     return 0
    # if(x1+w1<x2):
    #     return 0
    # if(y1+h1<y2):
    #     return 0
    x1, y1, w1, h1 = x
    x2, y2, w2, h2 = y
    colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    return overlap_area / (area1 + area2 - overlap_area)

print(bb_overlab(ground, grount))

cv2.rectangle(img, (ground[0], ground[1]), (ground[2] + ground[0], ground[1] + ground[3]),(255, 0, 0), 1)
cv2.rectangle(img, (grount[0], grount[1]), (grount[2] + grount[0], grount[1] + grount[3]),(0, 0, 255), 1)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()