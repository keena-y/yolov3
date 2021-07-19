# **YOLOV3**
<https://blog.csdn.net/public669/article/details/98020800>
## step1:创建文件项目
![alt 属性文本](https://github.com/keena-y/yolov3-imgs/blob/main/1.jpg)    
> cfg:yolo算法卷积层信息![alt 属性文本](https://github.com/keena-y/yolov3-imgs/blob/main/2.jpg)    
> weights:权重信息![alt 属性文本](https://github.com/keena-y/yolov3-imgs/blob/main/3.jpg) ![alt 属性文本](https://github.com/keena-y/yolov3-imgs/blob/main/4.jpg)   
> data:数据集![alt 属性文本](https://github.com/keena-y/yolov3-imgs/blob/main/5.jpg) ![alt 属性文本](https://github.com/keena-y/yolov3-imgs/blob/main/6.jpg)
## step2:准备数据集
#### 1.将图片放入JPEGImages文件夹中
#### 2.修改照片尺寸（416*416），提高模型训练的准确度
```images.py```
```python
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
```
#### 3.使用LabelImg软件标注images文件夹中图片，并将标注得到的xml文件放至Annotations文件夹中 ![alt 属性文本](https://github.com/keena-y/yolov3-imgs/blob/main/7.jpg)
#### 4.划分数据集，有train、trainval、test、val，并将个数据集信息保存至ImageSets文件夹中 ![alt 属性文本](https://github.com/keena-y/yolov3-imgs/blob/main/8.jpg)
```maketxt.py```
```python
import os
import random

trainval_percent = 0.3
train_percent = 0.7
xmlfilepath = 'data/Annotations'
txtsavepath = 'data/ImageSets'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('data/ImageSets/trainval.txt', 'w')
ftest = open('data/ImageSets/test.txt', 'w')
ftrain = open('data/ImageSets/train.txt', 'w')
fval = open('data/ImageSets/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
```
#### 5.将xml文件转换成txt文件，内容包括标注类别及其标注框位置
```voc_annotation.py```
```python
import os
import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2021', 'train'), ('2021', 'val'), ('2021', 'test')]

classes = ["eye"]#此处需要根据自己的分类修改

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation( year,image_id):
    in_file = open('data/Annotations/%s.xml'%(image_id))
    out_file = open('data\\labels\\%s.txt' % (image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        #list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()
print(wd)

for year, image_set in sets:
    if not os.path.exists('data/labels/'):
        os.makedirs('data/labels/')
    image_ids = open('data/ImageSets/%s.txt'%(image_set)).read().strip().split()
    list_file = open('data/%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('data/images/%s.jpg\n' %(image_id))
        convert_annotation(year, image_id)
    list_file.close()
```
#### 6.将待测图片放至samples文件夹中

## step3:模型训练
#### 1.在data文件夹下新建两个文件`yolo.names` `yolo.data`
> yolo.nanes 填入你所要识别的类名    
> yolo.data  ![alt 属性文本](https://github.com/keena-y/yolov3-imgs/blob/main/9.jpg) 根据自己项目来修改相应的路径
#### 2.修改`train.py`文件中内容，并执行   
只要修改主函数内容
```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/yolo.data', help='coco.data file path')  # TODO 改
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=416, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    opt = parser.parse_args()
    print(opt, end='\n\n')
```
> --epochs 迭代次数    
> --batch-size 一次训练时抓取的数据量样本个数     
> --cfg 所用卷积层信息所处位置，`.cfg`文件路径       
> --data-cfg 训练时所需信息的介绍，`.data`文件路径    
> **注：根据自己需要和项目路径修改**    

执行`train.py`文件后会生成2个文件`best.pt` `latest.pt`，文件分别为最好模型权重数据和最后模型权重数据。后面会基于两个文件对模型进行预测和评估。
## step4:预测
在`detect.py`文件中只需要修改主函数内容即可
![alt 属性文本](https://github.com/keena-y/yolov3-imgs/blob/main/11.jpg)
## step5:评估
修改`test.py`内容即可，原理同上。