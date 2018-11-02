import argparse
import os
import time
import shutil

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from utils import visualization_utils as vis_util
from utils import label_map_util
from pylab import mpl


if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

NUM_CLASSES = 764


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    #parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', default='./project/dataset',type=str)
    parser.add_argument('--labels', default='labels.txt',type=str)
    parser.add_argument('--in_file_path', type=str, required=True)
    parser.add_argument('--out_file_path',default='./project/output' , type=str)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    
    data = tf.compat.as_str(data).split('\n')
    return data

def convert_label_map_to_categories(filename):

    data = read_data(filename)
    categories = []
    for word in data:
        if word is not None:
            split_data = tf.compat.as_str(word).split(':')
            if len(split_data) > 1 :
                categories.append({'id': int(split_data[0]), 'vehicle_name': split_data[1]})

    return categories


def trans_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width)).astype(np.uint8)


def output_crop_image(image,boxes,classes,image_tensor,logits,scores):
    (im_width, im_height) = image.size
    bbox = boxes[0]
    for i,box in enumerate(bbox):
        offset_height = int(im_height * box[0])
        offset_width = int(im_width * box[1])
        target_height = int(im_height * box[2])
        target_width = int(im_width * box[3])
        if target_height > 0 and target_width > 0 and classes[0][i] == 3:
            crop_image = image.crop((offset_height, offset_width, target_height, target_width))
            crop_image.save("./crop_test.jpg")
            trans_image = open("./crop_test.jpg", 'rb').read()
            logit_value = sess.run(
                [logits],
                feed_dict={image_tensor: trans_image})
            pre_classes = np.argmax(logit_value)
            pre_cores = np.max(np.array(logit_value))
            if pre_classes >= 0 and pre_classes < 764:
                classes[0][i] = pre_classes
            else: classes[0][i] = 0
        else:
            boxes[0][i]=1
    return classes,boxes,scores


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    PATH_TO_CKPT = os.path.join(FLAGS.dataset_dir, 'frozen_graph.pb')
    PATH_TO_CKPT1 = os.path.join(FLAGS.dataset_dir, 'frozen_inference_graph.pb')

    PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir,FLAGS.labels)

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


    categories = convert_label_map_to_categories(PATH_TO_LABELS)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            fl_image_tensor = detection_graph.get_tensor_by_name('input:0')
            logits = detection_graph.get_tensor_by_name('output:0')
            
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            img_file_path = os.path.join(FLAGS.in_file_path)
            start_time = time.time()

            if os.path.isfile(img_file_path):
                image = Image.open(img_file_path)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                fl_classes,fl_boxes,fl_scores  = output_crop_image(image,boxes,classes,fl_image_tensor,logits,scores)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(fl_boxes),
                    np.squeeze(fl_classes).astype(np.int32),
                    np.squeeze(fl_scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4)
                plt.imsave(os.path.join(FLAGS.out_file_path , 'output1.png'), image_np)
                plt.show()
            else:
                print(img_file_path)
            print('total use time =',time.time()-start_time)




