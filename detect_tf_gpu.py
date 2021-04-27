import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import time
import numpy as np
import tensorflow as tf
from yolov3.utils import Load_Yolo_model, detect_images_multi_process
from yolov3.configs import *


def arg_parser():
    global args

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", required=True, help="relative path to the dataset", dest="dir")
    ap.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.87,
        dest="cnf",
        help="minimum probability to filter weak detections",
    )
    ap.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.45,
        dest="trs",
        help="threshold when applyong non-maxima suppression",
    )

    result = ap.parse_args()
    args = vars(ap.parse_args())

    return result


def detect(directory, confidence, threshold):
    """Function to detect objects and generate adnotations from multiple images using a yolo pretrained model.
    Includes options for: maximum number of labeling (ex: you want to label maxim 3500 objects from each class in order to have a ballance ds),
    class offset (ex: supose you have a dataset composed of 2 datasets), selective labeling from more complex model.
    """

    cwd = os.getcwd()
    image_list = []

    images_path = os.path.join(cwd, directory)

    # loading images from the directory
    for image in os.listdir(images_path):
        if image.endswith(".jpg"):
            elem_path = os.path.join(images_path, image)
            image_list.append(elem_path)

    print("Image list len: ", len(image_list))

    # map one detector output to your class configuration
    # if you don't want to use a bigger detector set bigger_detector to None
    
    interest_classes = [0, 1, 2, 3, 5, 7, 11]  # detected classes
    mapped_class = [7, 9, 11, 13, 10, 17, 2]  # replace with the following classes
    bigger_detector = [True, interest_classes, mapped_class]

    # specify the number of classes already labeled and the maximum number you wish to be labeled
    already_labeled_number = [2187, 1976, 2051, 2255, 2030, 0, 1386]
    total_number = [2300, 2300, 2300, 2300, 2300, 2300, 2300]

    # 3 14
    detect_images_multi_process(
        image_list, 
        confidence,
        threshold,
        False, #check labeled; if used with bigger_detector set to False 
        0, # number of classes; if used with bigger_detector set to 0
        0, # offset 
        total_number, 
        already_labeled_number,
        bigger_detector, 
    )


if __name__ == "__main__":

    result = arg_parser()

    directory = result.dir
    confidence = result.cnf
    threshold = result.trs

    detect(directory, confidence, threshold)
