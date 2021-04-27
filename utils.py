# ================================================================
#
#   File name   : utils.py
#   Author      : PyLessons
#   Created date: 2020-09-27
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : additional yolov3 and yolov4 functions
#
# ================================================================
from multiprocessing import Process, Queue, Pipe
import cv2
import time
import random
import colorsys
import numpy as np
import tensorflow as tf
from yolov3.configs import *
from yolov3.yolov4 import *
from tensorflow.python.saved_model import tag_constants
from loguru import logger
from datetime import datetime, timedelta
import collections


def load_yolo_weights(model, weights_file):
    tf.keras.backend.clear_session()  # used to reset layer names
    # load Darknet original weights to TensorFlow model
    if YOLO_TYPE == "yolov3":
        range1 = 75 if not TRAIN_YOLO_TINY else 13
        range2 = [58, 66, 74] if not TRAIN_YOLO_TINY else [9, 12]
    if YOLO_TYPE == "yolov4":
        range1 = 110 if not TRAIN_YOLO_TINY else 21
        range2 = [93, 101, 109] if not TRAIN_YOLO_TINY else [17, 20]

    with open(weights_file, "rb") as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(range1):
            if i > 0:
                conv_layer_name = "conv2d_%d" % i
            else:
                conv_layer_name = "conv2d"

            if j > 0:
                bn_layer_name = "batch_normalization_%d" % j
            else:
                bn_layer_name = "batch_normalization"

            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in range2:
                # darknet weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape)
            )
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in range2:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, "failed to read all data"


def Load_Yolo_model():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if len(gpus) > 0:
        print(f"GPUs {gpus}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    if YOLO_FRAMEWORK == "tf":  # TensorFlow detection
        if YOLO_TYPE == "yolov4":
            Darknet_weights = (
                YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
            )
        if YOLO_TYPE == "yolov3":
            Darknet_weights = (
                YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS
            )

        if YOLO_CUSTOM_WEIGHTS == False:
            print("Loading Darknet_weights from:", Darknet_weights)
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
            load_yolo_weights(yolo, Darknet_weights)  # use Darknet weights
        else:
            checkpoint = f"./checkpoints/{TRAIN_MODEL_NAME}"
            if TRAIN_YOLO_TINY:
                checkpoint += "_Tiny"
            print("Loading custom weights from:", checkpoint)
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
            yolo.load_weights(checkpoint)  # use custom weights

    elif YOLO_FRAMEWORK == "trt":  # TensorRT detection
        saved_model_loaded = tf.saved_model.load(
            YOLO_CUSTOM_WEIGHTS, tags=[tag_constants.SERVING]
        )
        signature_keys = list(saved_model_loaded.signatures.keys())
        yolo = saved_model_loaded.signatures["serving_default"]

    return yolo


def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh : nh + dh, dw : nw + dw, :] = image_resized
    image_paded = image_paded / 255.0

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method="nms"):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = bboxes[:, 5] == cls
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[:max_ind], cls_bboxes[max_ind + 1 :]]
            )
            # Process 3: Calculate this bounding box A and
            # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ["nms", "soft-nms"]

            if method == "nms":
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == "soft-nms":
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.0
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, h, w, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate(
        [
            pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
            pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5,
        ],
        axis=-1,
    )
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = h, w
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate(
        [
            np.maximum(pred_coor[:, :2], [0, 0]),
            np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1]),
        ],
        axis=-1,
    )
    invalid_mask = np.logical_or(
        (pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3])
    )
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(
        np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1)
    )
    scale_mask = np.logical_and(
        (valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1])
    )

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate(
        [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1
    )


@logger.catch
def predict_bbox_mp(
    Frames_data, Predicted_data, label_interval, check_labeled, maping_dictionary):
    ''' Checks if the image txt file contains a target class already labeled
    or in the labeling dictionary and if not runs detection on the image'''

    gpus = tf.config.experimental.list_physical_devices("GPU")

    class_range = False
    check_class = False

    if bool(maping_dictionary):
        check_class = True

    elif check_labeled is True:
        if label_interval is None:
            RuntimeError: logger.error("Specify classes interval")
        else:
            min_class = label_interval[0]
            max_class = label_interval[0] + label_interval[1] - 1
            class_range = True
            logger.debug("Min class: " + min_class + " Max class: " + str(max_class))

    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            logger.error(
                "RuntimeError in tf.config.experimental.list_physical_devices('GPU')"
            )

    Yolo = Load_Yolo_model()

    while True:
        if Frames_data.qsize() > 0:
            skip_image = False

            # read image from pipeline
            (image_data, txt_path, (h, w), counter) = Frames_data.get()

            # if contains objects with the id in the interval
            # it means that the image was already labeled or if the image containts
            # a class that is in our dictionary and we skip it

            if class_range or check_class:
                with open(txt_path) as f:
                    lines = f.readlines()
                for line in lines:
                    splitted = line.split(" ")
                    try:
                        value = int(splitted[0])
                        if check_class:
                            if value in maping_dictionary.values():
                                skip_image = True
                                break
                        elif value <= max_class and value >= min_class:
                            skip_image = True
                            break
                    except:
                        skip_image = True

            if skip_image is False:
                batched_input = tf.constant(image_data)
                result = Yolo(batched_input)
                pred_bbox = []

                for key, value in result.items():
                    value = value.numpy()
                    pred_bbox.append(value)

                pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
                pred_bbox = tf.concat(pred_bbox, axis=0)

                Predicted_data.put((pred_bbox, txt_path, (h, w), counter))


@logger.catch
def postprocess_mp(
    Predicted_data, Processed_frames, input_size, score_threshold, iou_threshold):

    while True:
        if Predicted_data.qsize() > 0:
            (pred_bbox, txt_name, (h, w), counter) = Predicted_data.get()

            bboxes = postprocess_boxes(pred_bbox, h, w, input_size, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method="nms")

            Processed_frames.put((bboxes, txt_name, (h, w), counter))


def convert(size, box):
    # xmin, xmax, ymin, ymax
    # converts to yolo format

    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


@logger.catch
def save_image_mp(
    Processed_frames,
    offset,
    start_time,
    total_counter,
    labeled_objects_number,
    maping_dictionary,
    labeling_threshold = 0.8
):

    internal_counter = 0
    internal_time = 0
    max_number = False
    map_classes = False

    # load dictionary
    if bool(maping_dictionary):
        map_classes = True

    # load classes numbers infos
    if labeled_objects_number is not None:

        max_number = True
        current_dict = {}
        max_number_dict = {}

        for label in labeled_objects_number:

            class_id = label[0]
            current_number = label[1]
            max_number = label[2]

            current_dict[int(class_id)] = int(current_number)
            max_number_dict[int(class_id)] = int(max_number)



    while True:
        if Processed_frames.qsize() > 0:

            internal_counter += 1
            (bboxes, txt_name, (hight, width), counter) = Processed_frames.get()

            # showing some information regarding processing time
            if internal_counter % 50 == 0:
                procent = (internal_counter / total_counter) * 100
                logger.info(
                    "Processed: "
                    + str(procent)
                    + " %"
                    + " in: "
                    + str(timedelta(seconds=int(time.time() - start_time)))
                )

                if internal_time != 0:
                    logger.info(
                        "Average time per image: "
                        + str((time.time() - internal_time) / 50)
                    )

                internal_time = time.time()

            file_content = open(txt_name, "a")
            urlist_len = len(bboxes)
            counter = 1
            label = False

            for box in bboxes:
                x_min = box[0]
                x_max = box[2]
                y_min = box[1]
                y_max = box[3]

                new_bbx = [x_min, x_max, y_min, y_max]

                # the 4-th box returns the confidence of the detection
                if box[4] > labeling_threshold:
                    class_id = int(box[5]) + offset
                    label = True

                    # check if the id in the id dictionary
                    if map_classes:
                        if class_id in maping_dictionary:
                            index = list(maping_dictionary.keys()).index(class_id)
                            old_class_id = class_id
                            class_id = maping_dictionary[class_id]

                            if max_number:
                                current_val = current_dict[old_class_id]
                                if current_val <= max_number_dict[old_class_id]:
                                    current_dict[old_class_id] += 1
                                else:
                                    print("Max val reached for : ", class_id)
                                    print("Total number: ", current_dict[old_class_id])
                                    continue
                        else:
                            continue
                    elif max_number:
                        current_val = current_dict[class_id]
                        if current_val <= max_number_dict[class_id]:
                            current_dict[class_id] += 1

                            if (current_val + 1) % 50 == 0:
                                logger.info(
                                    "Labeled : "
                                    + str(current_val + 1)
                                    + " from class: "
                                    + str(class_id)
                                )

                    x, y, w, h = convert((width, hight), new_bbx)

                    # check if reached the last element, so we don't print with \n
                    if counter == urlist_len:
                        # print("Reached last element")
                        file_content.write(
                            str(class_id)
                            + " "
                            + str(round(x, 4))
                            + " "
                            + str(round(y, 4))
                            + " "
                            + str(round(w, 4))
                            + " "
                            + str(round(h, 4))
                        )
                    else:
                        file_content.write(
                            str(class_id)
                            + " "
                            + str(round(x, 4))
                            + " "
                            + str(round(y, 4))
                            + " "
                            + str(round(w, 4))
                            + " "
                            + str(round(h, 4))
                            + "\n"
                        )

                    # print("File name: ", txt_name)
                    # print("X, Y, w, h: ", x,y,w,h)
                    # print(" ----------")

                counter += 1

            # comment this if you don't want prints for debugging
            if label is True:
                print("Image : " + txt_name + " was labeled")

            file_content.close()


@logger.catch
def detect_images_multi_process(
    files_list,
    confidence,
    threshold,
    check_labeled=False,
    classes_number=0,
    offset=0,
    total_number=0,
    already_labeled_number=0,
    bigger_detector=None,
):

    
    input_size = 608 # input size must match converted network size
    logger.start("file_{time}.log", rotation="500 MB", enqueue=True)

    frames_data = Queue()
    predicted_data = Queue()
    processed_frames = Queue()

    total_images = len(files_list)

    maping_dictionary = collections.OrderedDict()

    # check if bigger det
    if bigger_detector is not None:
        if bigger_detector[0] is True:
            classes_number = 0
            offset = 0
            check_labeled = False

        # create dictionary from lists
        for index, val in enumerate(bigger_detector[1]):
            maping_dictionary[val] = bigger_detector[2][index]

    # check if dataset is already labeled
    if classes_number != 0:
        label_interval = (offset, classes_number)
    else:
        label_interval = None

    if len(total_number) == 0:
        labeled_objects_number = None
    else:
        if len(total_number) != 0:
            labeled_objects_number = []

            if bigger_detector is None:
                for i in range(offset, offset + classes_number):
                    object_list = [
                        i,
                        already_labeled_number[i - offset],
                        total_number[i - offset],
                    ]
                    labeled_objects_number.append(object_list)
            else:
                counter = 0
                for key in maping_dictionary:
                    object_list = [
                        key,
                        already_labeled_number[counter],
                        total_number[counter]
                    ]
                    counter += 1
                    labeled_objects_number.append(object_list)
        else:
            labeled_objects_number = None

    start_time = time.time()

    p1 = Process(
        target=predict_bbox_mp,
        args=(
            frames_data,
            predicted_data,
            label_interval,
            check_labeled,
            maping_dictionary,
        ),
    )

    p2 = Process(
        target=postprocess_mp,
        args=(predicted_data, processed_frames, input_size, confidence, threshold),
    )

    p3 = Process(
        target=save_image_mp,
        args=(
            processed_frames,
            offset,
            start_time,
            total_images,
            labeled_objects_number,
            maping_dictionary,
        ),
    )

    p1.start()
    p2.start()
    p3.start()

    # use a sleeper to allow all dependencies loading
    time.sleep(40)

    for counter, element in enumerate(files_list):
        txt_file = element[:-3] + "txt"
        image = cv2.imread(element)

        if image is None:
            logger.error("Image " + element + " could not be read")
        else:
            h, w = image.shape[:2]
            
            # insert maximum 10 elements in queue to not occupy all RAM
            while frames_data.qsize() > 10:
                time.sleep(2)

            image_data = image_preprocess(np.copy(image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            frames_data.put((image_data, txt_file, (h, w), counter + 1))

    # after sending all image in pipeline
    # wait for process to finsh their job

    while True:
        if (
            frames_data.qsize() == 0
            and predicted_data.qsize() == 0
            and processed_frames.qsize() == 0
        ):
            p1.terminate()
            p2.terminate()
            p3.terminate()
            break
