# -*- coding=utf-8 -*-
#!/usr/bin/python

import sys
import os
import shutil
import numpy as np
import json
import xml.etree.ElementTree as ET
# import mmcv
# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                         #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                         #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                         #  "motorbike": 14, "person": 15, "pottedplant": 16,
                         #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars



def convert( json_file):
    '''
    :param xml_list: 需要转换的XML文件列表
    :param xml_dir: XML的存储文件夹
    :param json_file: 导出json文件的路径
    :return: None
    '''
    # list_fp = xml_list
    image_id=1
    # 标注基本结构
    json_dict = {"images":[],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}

    path = "D:/cancer/gan/coco/train2017/"  # 文件夹目

    g = os.walk(path)
    i = 0
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for path, dir_list, file_list in g:
        for file_name in file_list:

            # print(os.path.join(path, file_name))
            print(file_name)
            labels = file_name.replace('.nrrd', '')
            labels = labels.split("_")
            labels_roi = list(map(int, labels[3:7]))
            labels_ggo = labels[-1]
            height, width = 512, 512
            filename = file_name

            # 取出图片名字
            image_id+=1

            image = {'file_name': filename,
                     'height': height,
                     'width': width,
                     'id':image_id}
            json_dict['images'].append(image)

            annotation = dict()
            annotation['area'] = width*height
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = labels_roi
            annotation['category_id'] = labels_ggo
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            # 设置分割数据，点的顺序为逆时针方向
            # annotation['segmentation'] = []

            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1

    categories=[{'id':1, 'name': '1'},{'id':2, 'name': '2'},{'id':3, 'name': '3'},{'id':4, 'name': '4'},{'id':5, 'name': '5'}]
    json_dict['categories'].append(categories)
    # 导出到json
    #mmcv.dump(json_dict, json_file)
    print(type(json_dict))
    json_data = json.dumps(json_dict)
    with  open(json_file, 'w') as w:
        w.write(json_data)


if __name__ == '__main__':
    root_path = './demo'
    json_file = os.path.join(root_path, 'coco/annotations/instances_train2014.json')
    convert(json_file)
    # path = "D:/cancer/gan/coco/train2017/"  # 文件夹目录
    # save_path = 'D:/cancer/gan/lungnrrd/COCO_LIDC/annotations/'
    # if not os.path.exists(os.path.join(root_path,'coco/annotations')):
    #     os.makedirs(os.path.join(root_path,'coco/annotations'))
    # if not os.path.exists(os.path.join(root_path, 'coco/train2014')):
    #     os.makedirs(os.path.join(root_path, 'coco/train2014'))
    # if not os.path.exists(os.path.join(root_path, 'coco/val2014')):
    #     os.makedirs(os.path.join(root_path, 'coco/val2014'))
    #
    # xml_dir = os.path.join(root_path,'voc/Annotations') #已知的voc的标注
    #
    #
    # xml_labels = os.listdir(xml_dir)
    # np.random.shuffle(xml_labels)
    # split_point = int(len(xml_labels)/10)
    #
    # # validation data
    # xml_list = xml_labels[0:split_point]
    # json_file = os.path.join(root_path,'coco/annotations/instances_val2014.json')
    # convert(xml_list, xml_dir, json_file)
    # for xml_file in xml_list:
    #     img_name = xml_file[:-4] + '.jpg'
    #     shutil.copy(os.path.join(root_path, 'voc/JPEGImages', img_name),
    #                 os.path.join(root_path, 'coco/val2014', img_name))
    # # train data
    # xml_list = xml_labels[split_point:]
    # json_file = os.path.join(root_path,'coco/annotations/instances_train2014.json')
    # convert(xml_list, xml_dir, json_file)
    # for xml_file in xml_list:
    #     img_name = xml_file[:-4] + '.jpg'
    #     shutil.copy(os.path.join(root_path, 'voc/JPEGImages', img_name),
    #                 os.path.join(root_path, 'coco/train2014', img_name))