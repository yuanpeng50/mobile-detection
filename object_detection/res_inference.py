# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from utils import visualization_utils as vis_util
from utils import label_map_util
#from pylab import mpl

#mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
#mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

NUM_CLASSES = 764


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=True)


    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    
    data = tf.compat.as_str(data).split('\n')
    return data


#def create_category_index(filename):
def convert_label_map_to_categories(filename):

    data = read_data(filename)
    categories = []
    category_index = {}
    for word in data:
        if word is not None:
            split_data = tf.compat.as_str(word).split(':')
            if len(split_data) > 1 :
                categories.append({'id': int(split_data[0]), 'vehicle_name': split_data[1]})
                #category_index[int(split_data[0])]={'id': int(split_data[0]), 'vehicle_name': split_data[1]}
    categories.append({'id': 764, 'vehicle_name': 'N/A'})

    #print(category_index)
    #return category_index   
    return categories

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()

    PATH_TO_CKPT = os.path.join(FLAGS.output_dir, 'output/frozen_graph.pb')
    PATH_TO_CKPT1 = os.path.join(FLAGS.output_dir, 'output/rfcn_resnet101_coco/frozen_inference_graph.pb')

    PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir,'labels.txt')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT1, 'rb') as fid1:
            serialized_graph1 = fid1.read()
            od_graph_def.ParseFromString(serialized_graph1)
            tf.import_graph_def(od_graph_def, name='')


    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    test_img_path = os.path.join(FLAGS.dataset_dir, FLAGS.file_name)
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    #categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    categories = convert_label_map_to_categories(PATH_TO_LABELS)
    #print(categories)
    category_index = label_map_util.create_category_index(categories)
    #category_index = create_category_index(PATH_TO_LABELS)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('input:0')
            logits = detection_graph.get_tensor_by_name('output:0')
            #image = Image.open(test_img_path)
            image = open(test_img_path,'rb').read() 
            #image_np = load_image_into_numpy_array(image)
            #image_np_expanded = np.expand_dims(image_np, axis=0)
            logit_value = sess.run(
                [logits],
                feed_dict={image_tensor: image})
            #print(logit_value)
            pre_classes=np.argmax(logit_value)
            pre_cores=np.max(np.array(logit_value))
            print(np.max(np.array(logit_value)))
            #print(np.sum(np.array(logit_value)))
            print(pre_cores)
            print(pre_classes)
            if pre_classes >=0 and pre_classes < 764 and pre_cores > 0.9999:
               image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
               detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
               detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
               detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
               num_detections = detection_graph.get_tensor_by_name('num_detections:0')
               image = Image.open(test_img_path)
               image_np = load_image_into_numpy_array(image)
               image_np_expanded = np.expand_dims(image_np, axis=0)
               (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
               #print(classes)
               #print(category_index)
               vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=1)
               plt.imsave(os.path.join(FLAGS.output_dir, str(pre_classes)+'_output.png'), image_np)
            else:
               image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
               detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
               detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
               detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
               num_detections = detection_graph.get_tensor_by_name('num_detections:0')
               image = Image.open(test_img_path)
               image_np = load_image_into_numpy_array(image)
               image_np_expanded = np.expand_dims(image_np, axis=0)
               (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
               #print(classes)
               #print(category_index)
               #classes=[]
               category_index = {}
               category_index[764]={'id': 764, 'vehicle_name': 'N/A'}
 
               #print(scores)
               vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=8)
               plt.imsave(os.path.join(FLAGS.output_dir, str(pre_classes)+'_output.png'), image_np)

               #print(np.argmax(logit_value))
