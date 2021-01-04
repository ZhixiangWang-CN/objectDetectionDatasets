# -*- coding: utf-8 -*-
'''
@time: 2020/10/28 12:06
spytensor,zhoujianwen
'''

import os
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import shutil
from IPython import embed
from sklearn.model_selection import train_test_split

np.random.seed(41)

# 0为背景
classname_to_id = {"__background__": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}


class Txt2CoCo:

    def __init__(self, image_dir, total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            shapes = self.total_annos[key]
            for shape in shapes:
                bboxi = []
                for cor in shape[:-1]:
                    bboxi.append(float(cor))
                label = shape[-1]
                annotation = self._annotation(bboxi, label)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        print(path)
        img = cv2.imread(os.path.join(self.image_dir, path))
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape, label):
        # label = shape[-1]
        points = shape[:8]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [self._get_seg(points)]
        annotation['bbox'] = one_box
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    # def _get_box(self, points):  # 需要额外处理by zhoujianwen
    #     min_x = min_y = np.inf
    #     max_x = max_y = 0
    #     arraypoints = np.array(points).reshape(4,2)
    #     for x, y in arraypoints:
    #         min_x = min(min_x, x)
    #         min_y = min(min_y, y)
    #         max_x = max(max_x, x)
    #         max_y = max(max_y, y)
    #     return [min_x, min_y, max_x - min_x, max_y - min_y]

    # Segmentation的格式：[x1,y1,x1,y2,x2,y2,x1,y2]
    def _get_seg(self, points):
        x1, y1, x2, y2 = points
        return [x1, y1, x2, y2]


if __name__ == '__main__':
    path = "D:/cancer/gan/coco/train2017/"  # 文件夹目
    image_dir = "D:/cancer/gan/coco/train2017/"
    saved_coco_path = "outputs"
    # 整合txt格式标注文件
    total_txt_annotations = {}
    annotations = []


    g = os.walk(path)
    i = 0

    for path, dir_list, file_list in g:
        for file_name in file_list:
            # print(os.path.join(path, file_name))
            print(file_name)
            labels = file_name.replace('.jpg', '')
            labels = labels.split("_")
            labels_roi = list(map(int, labels[3:7]))
            labels_ggo = labels[-1]
            height, width = 512, 512
            nameOfImage = file_name



            cls = (labels_ggo)  # class
            x1 = float(labels_roi[0])
            y1 = float(labels_roi[1])
            w = float(labels_roi[2])
            h = float(labels_roi[3])

            one_box = [nameOfImage, x1, y1, w, h, cls]
            annotations.append(one_box)


    # annotations = pd.read_csv(csv_file, header=None).values
    for annotation in annotations:
        key = annotation[0].split(os.sep)[-1]
        value = np.array([annotation[1:]])
        if key in total_txt_annotations.keys():
            total_txt_annotations[key] = np.concatenate((total_txt_annotations[key], value), axis=0)
        else:
            total_txt_annotations[key] = value
    # 按照键值划分数据
    total_keys = list(total_txt_annotations.keys())

    train_keys, test_keys = train_test_split(total_keys, test_size=0.2)
    #train_keys = total_keys
    #test_keys = []
    print("train_n:", len(train_keys), 'test_n:', len(test_keys))
    # 创建必须的文件夹
    if not os.path.exists('%scoco/annotations/' % saved_coco_path):
        os.makedirs('%scoco/annotations/' % saved_coco_path)
    if not os.path.exists('%scoco/images/train2020/' % saved_coco_path):
        os.makedirs('%scoco/images/train2020/' % saved_coco_path)
    if not os.path.exists('%scoco/images/test2020/' % saved_coco_path):
        os.makedirs('%scoco/images/test2020/' % saved_coco_path)
    # 把训练集转化为COCO的json格式
    l2c_train = Txt2CoCo(image_dir=image_dir, total_annos=total_txt_annotations)
    train_instance = l2c_train.to_coco(train_keys)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2020.json' % saved_coco_path)
    for file in train_keys:
        shutil.copy(os.path.join(image_dir,file), "%scoco/images/train2020/" % saved_coco_path)
    for file in test_keys:
        shutil.copy(os.path.join(image_dir,file), "%scoco/images/test2020/" % saved_coco_path)
    # 把验证集转化为COCO的json格式
    l2c_val = Txt2CoCo(image_dir=image_dir, total_annos=total_txt_annotations)
    val_instance = l2c_val.to_coco(test_keys)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_test2020.json' % saved_coco_path)
