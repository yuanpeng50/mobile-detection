
# coding: utf-8

# In[ ]:


import tensorflow as tf
import PIL as pil
import matplotlib.pyplot as plt


with tf.Session() as sess:
    #provided that the tfrecord format is as same as the one used in assignment7
    #refer to quiz-w7-code/datasets/dataset_utils.py
    feature={
      'image/encoded': tf.FixedLenFeature([], tf.string),
      'image/format': tf.FixedLenFeature([], tf.string),
      'image/class/label': tf.FixedLenFeature([], tf.int64),
      'image/height': tf.FixedLenFeature([], tf.int64),
      'image/width': tf.FixedLenFeature([], tf.int64),
    }

    # define a queue base on input filenames
    filename_queue = tf.train.string_input_producer(["pj_vehicle_validation_00000-of-00004.tfrecord"])
    # define a tfrecords file reader
    reader = tf.TFRecordReader()
    # read in serialized example data
    image_name, serialized_example = reader.read(filename_queue)
    # decode example by feature
    rcfeatures = tf.parse_single_example(serialized_example, features=feature)

    image = tf.image.decode_jpeg(rcfeatures['image/encoded'])

    height = tf.cast(rcfeatures['image/height'], tf.int32)
    width = tf.cast(rcfeatures['image/width'], tf.int32)
    image_format = rcfeatures['image/format']
    # restore image to [height, width, 3]
    image = tf.reshape(image, [height, width, 3])
    label = tf.cast(rcfeatures['image/class/label'], tf.int32)  
    
    with tf.Session() as sess: #开始一个会话
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        for i in range(150):
            example, l , f, h, w = sess.run([image,label,image_format, height, width])#在会话中取出image和label
            img=pil.Image.fromarray(example, 'RGB')
            img_name = str(i+1)
            if i < 99:
                img_name = "0"+img_name
            if i < 9:
                img_name = "0"+img_name
            img.save(img_name+'.jpg')#存下图片

            coord.request_stop()
            coord.join(threads)
    
    sess.close()

