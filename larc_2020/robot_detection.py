#coding=utf-8

#* IMPORTS
import tensorflow as tf

import numpy as np
import os
# import six.moves.urllib as urllib
import sys

# from collections import defaultdict
from io import StringIO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg

from datetime import datetime

#*ENV SETUP
sys.path.append("..")

#*OBJECT DETECTION IMPORTS
os.chdir('TFmodels/research/')
sys.path.append(os.getcwd()+"object_detection/")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as util_ops


#*VARIABLES
os.chdir('../../')
MODEL_NAME = 'larc_detection_inference_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('./object_detection/data', 'detection_label_map.pbtxt')
NUM_CLASSES = 1

#*LOAD A FROZEN TENSORFLOW MODEL INTO MEMORY
print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

#*LOADING LABEL MAP
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#*HELPER CODE
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

#*DETECTION IN IMAGES
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'robot_test_{}.jpg'.format(i)) for i in range(1, len(os.listdir("test_images")) + 1)]
IMAGE_SIZE = (12, 8)

images = []
count_to_save = 0
list_of_arq_names = []
list_of_boundings = []
times = []
list_of_images_arrays = []
list_of_images_names = []

os.system("clear")
print("Loading images...\n")

for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    list_of_images_arrays.append(image_np)

    image_name = image_path[12:len(image_path)]
    list_of_images_names.append(image_name)

    arq_name = image_path[12:len(image_path)-4] + '.txt'
    list_of_arq_names.append(arq_name)


count_to_save = 0


with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        os.system('clear')
        print('Processing detection...\n\n')
        for image_np in list_of_images_arrays:
            image_name = list_of_images_names[count_to_save]

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            
            image_to_save = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                min_score_thresh=.50,
                use_normalized_coordinates=True,
                line_thickness=8,
            )
            if(count_to_save%5 == 0):
                print('Number of images processed: ', count_to_save)
                
            if(count_to_save < 20):
                images.append(image_to_save)
            count_to_save = count_to_save + 1

            #Escrevendo .txt
            bounding = []
            bounding_box_param = []
            lista_scores = []
            #Obtendo scores de cada reconhecimento
            scores = np.array(scores[0])
            lista_scores = scores[scores>=0.5]
            #Obtendo parâmetros de cada bounding box
            image_size_param = np.array([image_np.shape[0], image_np.shape[1],image_np.shape[0], image_np.shape[1]])
            
            for i in range(len(lista_scores)):
                bounding_box_param = np.multiply(boxes[0][i], image_size_param)
                bounding_box_param = [
                    int(bounding_box_param[0]), #TOP
                    int(bounding_box_param[1]), #LEFT
                    int(bounding_box_param[2]), #BOTTOM
                    int(bounding_box_param[3]), #RIGHT
                ] 
                line_of_arq = 'robot ' + str(lista_scores[i]) + ' ' + str(bounding_box_param[1]) + ' ' + str(bounding_box_param[0]) + ' ' + str(bounding_box_param[3]) + ' ' + str(bounding_box_param[2]) + '  \n'
                bounding.append(line_of_arq)
            list_of_boundings.append(bounding)

            if(count_to_save == 1) :
                times.append(datetime.now())
times.append(datetime.now())
print('Number of images processed: ', count_to_save)

#*PROCESSING TIME
print('\nRecording times...')
print('Start time: ', str(times[0]))
print('End time: ', str(times[1]))
tempo = times[1] - times[0]
print('\nProcess time: ', tempo)

#*RECORDING .txt WITH DATA
os.chdir('results')
i = 0
for arq in list_of_arq_names:
    with open(arq, 'a') as File:
        bounding = list_of_boundings[i]
        for line in bounding:
            File.write(line)
    i = i+1

os.chdir('..')

#*Recording images with bounding box
i = 1
for image in images:
    os.chdir('result_images')
    img = Image.fromarray(image)
    img.save('robot_test_' + str(i) + '.jpg')
    os.chdir('..')
    i = i +1
