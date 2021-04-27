# The-YOLO-Wizard :dizzy:

## Description
 The YOLO Wizzard is an awesome auto-labeling dataset tool.

 **What is this?**

 YOLO wizzard automates the laborious and time consuming process of data labeling.
 It runs on GPU and uses TensorflowRT.  Using an already trained network, you can label another datasets and use them for retrain or whatever you wish.
 This project is based on: https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3

**What do I need?**
1. You need a trained yolo model
2. You need an unlabeled dataset containing objects that the model was trained to detect.
3. Feed the wizzard the unlabeled dataset and trained weights and he will do *MAGIC* for you!
(OFC, it won't be as good as manual labeling, but still can save you some precious hours)

**The most paintfull part**

Installing all the dependencies is, from my point of view, the most time consuming part.
If everything is set, you just need to convert your weights to trt format and run the script on dataset.

## Performance
 
Testing benchmark:
* Ubuntu 18.04 (Nvidia)
* GeForce™ GTX 1080Ti
* 4 vCPUs
* 12 GB RAM
* 80 GB disk

Testing versions:
* CUDA 10.1
* cuDNN v7.6.5
* 

Average detection time per image of 0.20 s using a FP16 network.

**Image obtainted after detecting with full yolov4 weights using darknet**
![Using YOLO weights darknet detection](<images/darknet_predictions.jpg>)

**Image obtained after detecting with converted network using the wizzard**

You can notice the other classes represented too along with the new labels.
Altough the bounding box is not as precise for the second semaphore compare to the first picture,
the result is more than resonable.
![Using YOLO weights darknet detection](<images/wizzard_predictions.jpg>)


## Dependencies

In order to run this on GPU the weights have to be resized to pb and trt formats. It also reduces the network overall detection performance, but highly increases it's speed.

 
## HOW TO

1. Install all dependencies (CUDA, cuDNN, tf-gpu, etc.)
See requirements in

2. Convert YOLO weights to TRT format

    1. set in yolov3/configs.py YOLO_TRT_QUANTIZE_MODE = "FP16"
    (from what I've tested it works the best, you can use FP32, but it will take a bit more to process the dataset)

    2. set in yolov3/configs.py YOLO_CUSTOM_WEIGHTS = False

    3. set path to obj.names in yolov3/configs.py YOLO_COCO_CLASSES = "path_to_obj.names"

    4. set path to your custom yolo weights YOLO_V4_WEIGHTS="path_to_your_weights.weights"
    
    5. set YOLO_INPUT_SIZE to 608 or your network size

    6. set YOLO_FRAMEWORK = "trt"

    7. run python3 ./tools/Convert_to_pb.py

    8. run python3 ./tools/Convert_to_TRT.py

3. 1. modify YOLO_CUSTOM_WEIGHTS with your path to the saved converted weights
    YOLO_CUSTOM_WEIGHTS = "checkpoints/yolov4-trt-608"
    
    2. copy utils.py to yolov3/ in forked: https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
    
    3. run ./detect_tf_gpu.py specifying the directory of the dataset, minimum confidence and threshold


## Possible configurations

The detection and labeling allows for several configurations:
* **specify a class offset**: Suppose your network detects 3 objects and they will be labeled by default 0,1,2. In your dataset you already have other objects labeled with these ids. Specifying an offset will convert the final_id to initial_id + offset)
* **selective labeling**: Suppose your network detects car. If your dataset already contains some labeled cars you won't want to label those pictures because most likely they will be better labeled)
* **maping id's to detected id's**: Offset will work only if you need all objects detected by a network. In case your network is capable of detecting multiple objects, some of them might not be usefull, so you want to ignore those, and map your desired ids to the a part of network output ids).
* **allow max number of labeling**: Specify a maxmum number of labels you want for each class. This can be usefull to create a balance dataset)

Convert yolo weights to TRT format. 
Step by step tutorial here: https://pylessons.com/YOLOv4-TF2-TensorRT/



## Dependencies

