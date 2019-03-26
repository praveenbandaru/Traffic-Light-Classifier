# Traffic-Light-Classifier

## Abstract

This project implements the TensorFlow Object Detection API to solve a real-time problem such as traffic light detection. It uses the Microsoft Common Objects in Context (COCO) pre-trained model called Single Shot Multibox Detector MobileNet from the TensorFlow Zoo for transfer learning. We used an NVIDIA GeForce GTX 1070 GPU to retrain the model for 20000 steps using the image data from the rosbags provided by Udacity. At the end of this, we obtained an accurate model that was able to identify the traffic signals at more than 90 percent accuracy.

## Introduction

With the advancements in technology, there has been a rapid increase in the development of autonomous cars or smart cars. Accurate detection and recognition of traffic lights is a crucial part in the development of such cars. The concept involves enabling autonomous cars to automatically detect traffic lights using the least amount of human interaction. Automating the process of traffic light detection in cars would also help to reduce accidents.

Traditional approaches in machine learning for traffic light detection and classification are being replaced by deep learning methods to provide state-of-the-art results. However, these methods create various challenges. For example, the distortion or variation in images due to orientation, illumination, and speed fluctuation of vehicles could result in false recognition.

This project was implemented using transfer learning of the Microsoft Common Objects in Context (COCO) pre-trained model called Single Shot Multibox Detector (SSD) with MobileNet. A subset of the ImageNet dataset, which contains traffic lights, was used for further training to improve the performance. For this particular experiment, the entire training and the inferencing was done on an GeForce GTX 1070 GPU.

![](https://raw.githubusercontent.com/praveenbandaru/Traffic-Light-Classifier/master/results/red.png?token=AleOERYl9emAyx00DDs8vZWq0gsPmJeUks5cmdAIwA%3D%3D)

## Software Configuration

The development of this project had the following dependencies as shown below.

|Library|Version|
|--------|------|
|TensorFlow|1.4.0|
|Python|3.6|
|Operating system|Windows 10|
|Protobuf|3.4|
|Pillow|5.3.0|
|Lxml|4.2.5|
|Matplotlib|3.0.2|

## Installation

### Building and Installing TensorFlow

TensorFlow can be installed and used with several combinations of development tools and libraries on a variety of platforms. The following are the steps to install TensorFlow on Windows 10.

### Windows 10

1.  Install TensorFlow version 1.4 by executing the following statement in the Command Prompt (this assumes you have python.exe set in your PATH environment variable)
    
    ```
    pip install tensorflow==1.4
    
    ```
    
2.  Install the following python packages
    
    ```
    pip install pillow lxml matplotlib
    
    ```
    
3.  [Download protoc-3.4.0-win32.zip from the Protobuf repository](https://github.com/google/protobuf/releases) (It must be version 3.4.0!)
    
4.  Extract the Protobuf .zip file e.g. to `C:\Program Files\protoc-3.4.0-win32`
    
5.  Create a new directory somewhere and name it `tensorflow`
    
6.  Clone TensorFlow's _models_ repository from the `tensorflow` directory by executing
    
    ```
    git clone https://github.com/tensorflow/models.git
    
    ```
    
7.  Navigate to the `models` directory in the Command Prompt and execute
    
    ```
    git checkout f7e99c0
    
    ```
    
    This is important because the code from the `master` branch won't work with TensorFlow version 1.4. Also, this commit has already fixed broken models from previous commits.
    
8.  Navigate to the `research` folder and execute    
    
    `"C:\Program Files\protoc-3.4.0-win32\bin\protoc.exe" object_detection/protos/*.proto --python_out=. `
    
9.  If step 8 executed without any error then execute `python object_detection/builders/model_builder_test.py`
    
10.  In order to access the modules from the research folder from anywhere, the `models`, `models/research`, `models/research/slim` & `models/research/object_detection` folders need to be added to PATH.

### Installing LabelImg

Download the latest version of LabelImg, an annotation tool for Microsoft Windows. Extract the zip file, and then rename the folder as LabelImg.

## Solution Design

The solution was implemented with the TensorFlow Object Detection API. The detection pipeline is as given below.

### Traffic detection pipeline

    Algorithm 1: Detection Pipeline 
    boxAssigned← false
    while true do
	    f ←nextFrame
	    while boxAssigned == false do
		    InvokeDetection(f)
	    if Bounding Box is detected then
		    boxAssigned ← true
		    class ← identfiedClass
		    if class is Trafficlight then
			    drawBoundingBox
		    end if
	    end if
    end while

### Why choose TensorFlow Object Detection API?

TensorFlow’s Object Detection API is a powerful tool that makes it easy to construct, train, and deploy object detection models. In most of the cases, training an entire convolutional network from scratch is time consuming and requires large datasets. This problem can be solved by using the advantage of transfer learning with a pre-trained model using the TensorFlow API. Before getting into the technical details of implementing the API, let’s discuss the concept of transfer learning.

Transfer learning is a research problem in [machine learning](https://en.wikipedia.org/wiki/Machine_learning "Machine learning") that focuses on storing the knowledge gained from solving one problem and applying it to a different but related problem. Transfer learning can be applied three major ways:

Convolutional neural network (ConvNet) as a fixed feature extractor: In this method the last fully connected layer of a ConvNet is removed, and the rest of the ConvNet is treated as a fixed feature extractor for the new dataset.

Fine-tuning the ConvNet: This method is similar to the previous method, but the difference is that the weights of the pre-trained network are fine-tuned by continuing backpropagation.

Pre-trained models: Since modern ConvNets takes weeks to train from scratch, it is common to see people release their final ConvNet checkpoints for the benefit of others who can use the networks for fine-tuning. For example, TensorFlow Zoo is one such place where people share their trained models/checkpoints.

In this project, we used a pre-trained model for the transfer learning. The advantage of using a pre-trained model is that instead of building the model from scratch, a model trained for a similar problem can be used as a starting point for training the network. Many pre-trained models are available. This experiment used the COCO pre-trained model/checkpoints SSD MobileNet from the TensorFlow Zoo. This model was used as an initialization checkpoint for training. The model was further trained with images of traffic lights from ImageNet. This fine-tuned model was used for inference.

Now let’s look at how to implement the solution. The TensorFlow Object Detection API has a series of steps to follow, as shown in below figure.
![](https://raw.githubusercontent.com/praveenbandaru/Traffic-Light-Classifier/master/results/imageedit_5_6898013867.png?token=AleOEeTR8pMU4QfR7gRjjVLE2ORMXD2sks5cmdadwA%3D%3D)
### 1. Dataset download

#### 2.1 Extract images from a ROSbag file

Udacity provides the students with a ROSbag file from their Car named Carla where (our and) your capstone project will be tested on the code/procedure for extracting images will be (mostly) the same. **The steps below assume you have ros-kinetic installed either on your local machine (if you have Linux as an operating system) or in a virtual environment (if you have Windows or Mac as an operating system)**

1.  Open a terminal and launch ROS
     
     `roscore`
    
3.  Open another terminal (but do NOT close or exit the first terminal!) and play the ROSbag file
    
	   ` rosbag play -l path/to/your_rosbag_file.bag`
    
4.  Create a directory where you want to save the images
    
5.  Open another, third terminal and navigate to the newly created directory and...
    
    1.  execute the following statement if you have a ROSbag file from Udacity's simulator:
        
        `rosrun image_view image_saver _sec_per_frame:=0.01 image:=/image_color`
        
    2.  execute the following statement if you have a ROSbag file from Udacity's Car Carla:
        
        `rosrun image_view image_saver _sec_per_frame:=0.01 image:=/image_raw`
    

These steps will extract the (camera) images from the ROSbag file into the folder where the code is executed. Please keep in mind that the ROSbag file is in an infinite loop and won't stop when the recording originally ended so it will automatically start from the beginning. If you think you have enough data you should interrupt one of the open terminals.

If you can't execute step 4.1 or 4.2 you probably don't have `image_view` installed. To fix this install `image_view` with `sudo apt-get install ros-kinetic-image-view`.

Hint: You can see the recorded footage of your ROSbag file by opening another, fourth terminal and executing `rviz`.

### 2. Image Annotation

1.  Configuring the LabelImg tool. Before starting with the annotation of images, the classes for labelling needs to be defined in the `LabelImg/data/predefined_classes.txt` file. In this case, there’s four classes which are `Red, Green, Yellow and Unknown`.
2.  Launch labelimg.exe and then select the dataset folder by clicking the **OpenDir** icon on the left pane.
3.  For each image that appears, draw a rectangular box across each traffic light by clicking the **Create RectBox** icon. These rectangular boxes are known as bounding boxes. Select the category **trafficlight** from the drop-down list that appears.
4.  Repeat this process for every traffic light present in the image. Figure 2 shows an example of a completely annotated image.

![Annotated image](https://software.intel.com/sites/default/files/managed/a7/94/traffic-light-detection-using-tensorflow-object-detection-api-fig2.png)
Once the annotations for an image are completed, save the image to any folder.

The corresponding eXtensible Markup Language (XML) files will be generated for each image in the specified folder. XML files contain the coordinates of the bounding boxes, filename, category, and so on for each object within the image. These annotations are the ground truth boxes for comparison. Figure 3 represents the XML file of the corresponding image in Figure 2.

### 3. Label map preparation

Each dataset requires a label map associated with it, which defines a mapping from string class names to integer class IDs. Label maps should always start from ID 1.

As there are only four classes, the label map for this project file has the following structure:

    item {
      id: 1
      name: 'Green'
    }
    
    item {
      id: 2
      name: 'Red'
    }
    
    item {
      id: 3
      name: 'Yellow'
    }
    
    item {
      id: 4
      name: 'Unknown'
    }

### 4. TensorFlow records (TFRecords) generation

TensorFlow accepts inputs in a standard format called a TFRecord file, which is a simple record-oriented binary format. Eighty percent of the input data is used for training and 20 percent is used for testing. The split dataset of images and ground truth boxes are converted to train and test TFRecords. Here, the XML files are converted to csv, and then the TFRecords are created. Sample scripts for generation are available [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md).

We can use the below script to achieve this. For datasets with **.xml** files execute:

```
python create_tf_record.py --data_dir=path/to/green/lights,path/to/red/lights,path/to/yellow/lights --annotations_dir=labels --output_path=your/path/filename.record --label_map_path=path/to/your/label_map.pbtxt

```

You will know that everything worked fine if your .record file has nearly the same size as the sum of the size of your images. Also, you have to execute this script for your training set, your validation set (if you have one) and your test set separately.

### 5. Pipeline configuration

This section discusses the configuration of the hyperparameters, and the path to the model checkpoints, ft. records, and label map. The protosun files are used to configure the training process that has a few major configurations to be modified. A detailed explanation is given in [Configuring the Object Detection Training Pipeline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md). The following are the major settings to be changed for the project.
You will need to [download the .config file for the model you've chosen](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)
-   In the `**model config**`, the major setting to be changed is the `num_classes` that specifies the number of classes in the dataset. Change `num_classes: 90` to the number of labels in your `label_map.pbtxt`. This will be `num_classes: 4` . Set the default `max_detections_per_class: 100` and `max_total_detections: 300` values to a lower value for example `max_detections_per_class: 10` and `max_total_detections: 10`
-   The `**train config**` is used to provide model parameters such as `batch_size`, `num_steps` and `fine_tune_checkpoint`. `fine_tune_checkpoint` field is used to provide path to the pre-existing checkpoint. Change `fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"` to the directory where your downloaded model is stored e.g.: `fine_tune_checkpoint: "models/your_tensorflow_model/model.ckpt"`. Set `num_steps: 200000` down to `num_steps: 20000`. Set `batch_size` appropriately.
-   The `**train_input_config**` and `**eval_input_config**` fields are used to provide paths to the TFRecords and the label map for both train as well as test data. Change the `PATH_TO_BE_CONFIGURED` placeholders in `input_path` and `label_map_path` to your .record file(s) and `label_map.pbtxt`

### 6. Training
Now it's time to select a model which you will train. You can [see the stats of and download the Tensorflow models from the model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).  We selected the following model [ssd_mobilenet_v1_coco_11_06_2017](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz) for our training.
The final task is to assemble all that has been configured so far and run the training job (see Figure 4). Once the above steps are completed, the training file is executed. By default, the training job will continue to run until the step count reaches 20000. The models will be saved at various checkpoints.

![Training pipeline](https://raw.githubusercontent.com/praveenbandaru/Traffic-Light-Classifier/master/results/imageedit_7_5493518341.png?token=AleOESal-mYMg9dRTN3QQNlZpSQYn_kTks5cmeCOwA%3D%3D)
Follow the steps to train the model:
-   Clone your classification repository and create the folders `models` & `data` (in your project folder).
    
-   Copy the tfrecords to the `data` folder
        
-   Navigate to the `models` folder in your project folder and download your tensorflow model with
    
    wget http://download.tensorflow.org/models/object_detection/your_tensorflow_model.tar.gz
    tar -xvzf your_tensorflow_model.tar.gz
    
-   Copy the file `train.py` from the `tensorflow/models/research/object_detection` folder to the root of your project folder
    
-   Train your model by executing the following statement in the root of your project folder
    
    ```
    python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=./config/your_tensorflow_model.config
    
    ```
### Freezing the graph
When training is finished the trained model needs to be exported as a frozen inference graph. Udacity's Carla has TensorFlow Version 1.3 installed. However, the minimum version of TensorFlow needs to be Version 1.4 in order to freeze the graph but note that this does not raise any compatibility issues. If you've trained the graph with a higher version of TensorFlow than 1.4, don't panic! As long as you downgrade Tensorflow to version 1.4 before running the script to freeze the graph you should be fine. To freeze the graph:

1.  Copy `export_inference_graph.py` from the `tensorflow/models/research/object_detection` folder to the root of your project folder
    
2.  Now freeze the graph by executing
    
    ```
    python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/your_tensorflow_model.config --trained_checkpoint_prefix ./models/train/model.ckpt-20000 --output_directory models
    
    ```
    
    This will freeze and output the graph as `frozen_inference_graph.pb`.
    
### 7. Inference

The sample images were extracted from the rosbag files. These images were given to our model trained using transfer learning. After the images pass through the Object Detection pipeline, the bounding boxes will be drawn on the detected images.

![Inference pipeline](https://raw.githubusercontent.com/praveenbandaru/Traffic-Light-Classifier/master/results/imageedit_9_2584405846.png?token=AleOEaizk_zlvCZ4xAMlpqYIrbLGWj4Kks5cmeChwA%3D%3D)
## Experimental Results

We were able to get good results on our sample images and the model was able to detect the traffic light states with more than 90% accuracy.

Below are some sample images.
![](https://raw.githubusercontent.com/praveenbandaru/Traffic-Light-Classifier/master/results/red.png?token=AleOERYl9emAyx00DDs8vZWq0gsPmJeUks5cmdAIwA%3D%3D)![](https://raw.githubusercontent.com/praveenbandaru/Traffic-Light-Classifier/master/results/yellow.png?token=AleOEX98rjzGGPdFNPXKRj1pqIcL6Bjzks5cmeMUwA%3D%3D)![](https://raw.githubusercontent.com/praveenbandaru/Traffic-Light-Classifier/master/results/green.png?token=AleOEbsGzxYpNYufJylTsHoz8dvix0X0ks5cmeM6wA%3D%3D)

**[Take a look at the Jupyter Notebook](https://github.com/praveenbandaru/Traffic-Light-Classifier/blob/master/traffic_light_classification.ipynb) to see the results.**
