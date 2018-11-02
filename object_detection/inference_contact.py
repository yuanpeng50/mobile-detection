from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

from utils import visualization_utils as vis_util
from utils import label_map_util
import classify_image_contact


if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

NUM_CLASSES = 90


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()

    PATH_TO_CKPT = os.path.join(FLAGS.output_dir, 'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'mscoco_label_map.pbtxt')
    #PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'kitti_label_map.pbtxt')
    #PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'oid_bbox_trainable_label_map.pbtxt')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    test_img_path = os.path.join(FLAGS.dataset_dir, 'test1.jpg')

    def cropmake(im,x1,y1,x2,y2,img_name):
        bbox = (im.size[0]*x1, im.size[1]*y1, im.size[0]*x2, im.size[1]*y2)
        bbox=tuple(bbox)
        try:
            newim=im.crop(bbox)
            print(newim)
            newim.save(os.path.join(FLAGS.output_dir, img_name))
        except SystemError:
            print("Error")

    #cropmake()



    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
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
            print(boxes)
            print(scores)
            print(classes)
            print(num)
            boxes_squeeze = np.squeeze(boxes)
            classes_squeeze = np.squeeze(classes)
            for i in range(len(boxes_squeeze)):
                if classes_squeeze[i] == 3 or classes_squeeze[i] == 6 or classes_squeeze[i] == 8:
                    #cropmake(image,boxes_squeeze[i][1],boxes_squeeze[i][0],boxes_squeeze[i][3],boxes_squeeze[i][2],str(i)+"slice.jpg")
                    bbox = (image.size[0]*boxes_squeeze[i][1], image.size[1]*boxes_squeeze[i][0], image.size[0]*boxes_squeeze[i][3], image.size[1]*boxes_squeeze[i][2])
                    bbox=tuple(bbox)
                    try:
                        newim=image.crop(bbox)
                    except SystemError:
                        print("crop Error")
                    classify_image_contact.run_inference_on_image(newim)
                    newim.save(os.path.join(FLAGS.output_dir, str(i)+"slice.jpg"))




            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=1)
            #plt.imsave(os.path.join(FLAGS.output_dir, 'output.png'), image_np)

    #classify_image_contact.run_inference_on_image('./faster_rcnn_nas_lowproposals_coco_2018_01_28/0slice.jpg')
