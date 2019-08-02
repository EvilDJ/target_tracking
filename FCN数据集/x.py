import colorlabel
import PIL.Image
import numpy as np
from skimage import io,data,color

#将生成的json文件转换成png的格式文件


img = PIL.Image.open('F:网络数据集/测试/2007_000039_json/label.png')
label = np.array(img)
dst = colorlabel.label2rgb(label,bg_label = 0,bg_color =(0,0,0))

io.imsave('F:网络数据集/测试/000004.png',dst)