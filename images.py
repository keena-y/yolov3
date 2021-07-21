import os
from PIL import Image
dir_img="data/JPEGImages/"
dir_save="data/images/"
size=(416,416)

list_img = os.listdir(dir_img)#获取目录下所有图片名

#遍历
for img_name in list_img:
    pri_image = Image.open(dir_img+img_name)
    tmppath=dir_save+img_name

    #保存缩小的图片
    pri_image.resize(size, Image.ANTIALIAS).save(tmppath)