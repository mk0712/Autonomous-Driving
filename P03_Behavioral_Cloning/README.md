#Udacity Self-Driving Car Nanodegree
#Term 1, Project 3: Behavioral Cloning 

#Introduction 
The goal of this project is to first record images from a driving car and then use those images to train a model that allows the car to drive by itself. Therefore, three types of images are taken: left side of the car, center view, right side of the car. The model is based on Keras.

#YouTube
The recorded video of the final model can be viewed on YouTube under the following URL:
https://youtu.be/ifxIFyX5x3M

#List of submitted files:
- README.md 
     - Explains the structure of your network and training approach. While we recommend using English for good practice, writing in any language is acceptable (reviewers will translate). There is no minimum word count so long as there are complete descriptions of the problems and the strategies. See the rubric for more details about the expectations.
- behavioral_cloning_0.0.1.2.ipynb, behavioral_cloning_0.0.1.2.html
     - The script used to create and train the model.
- model.json
     - The model architecture. 
- model.h5
     - The model weights.
- drive.py
     - The script to drive the car. You can feel free to resubmit the original drive.py or make modifications and submit your modified version.

- Subfolder “Supplementary material”: 
     - model_ownData.json, model_ownData.h5:
         - The same model trained exclusively on my own data. Unfortunately, this model does not steer the car through track 1 without leaving the track.
     - model_UdacityData.json, model_UdacityData.h5:
         - The same model trained exclusively on Udacity’s sample data. Unfortunately, this model does not steer the car through track 1 without leaving the track.
     - training_data_preprocessed_visualization.png
         - Visualization of loaded data and pre-processed images. 

#Training data 
I recorded training data. However, the trained model was not able to drive by itself yet. Therefore, I extended the recorded training data with Udacity’s sample data.

In particular, I recorded the following data:
-	Track 1:
    - 3 full laps forward direction
    - 1 full lap backwards
    - 2 laps forward leaving and entering the lap and driving to the middle (only recording while entering)
    - Additional images from the curves (Note: This is consistent with the NVIDIA paper. To reduce bias from the majority of straight roads, NVIDIA focused on curves when training the model. [paper p.5])

- Track 2: 
  - 2 “laps” forward (i.e., driving downwards to the valley)
  - 1 “lap” backward (i.e., driving from the valley up the mountain)

#Approach

##Load and pre-process data:

I load the data from the file driving_log.csv which was is created when recording driving data. It is important to note, that the driving_log.csv file provides the paths and file names of the created images (which represent features in this project) and stores them with the recorded steering angles (which represent labels in this project that shall be predicted). 
The loaded data is stored in the variables X and y. 

To reduce biasing noise in the images, I only loaded an extract of each image which shows the track and the boundaries of the track. Specifically, I focused on the y-axis pixel coordinates 50 to 150. I did not cut the x-axis and kept all three color channels. I think that reducing color channels might work well in this simulator because there is only the gray track and red/white or yellow markings. However, I did not want to do it because (a) NVIDIA also did not recommend it in their paper and (b) I think that color in general would be an important feature (e.g., further lines, traffic signs etc.). Thus, the loaded image (without slicing) was 100 x 320 x 3 pixels. 
Furthermore, I reduced the size of the image because 160 x 320 would have been too many pixels for my laptop’s memory (8 GB).

Specifically, I tried three different image sizes. In general, my goal was to keep the largest possible image size that fits in my memory when loading around 15000 images. Therefore, I first reduced the size by 1-100%/2 = 50%. This created images of 50x160x3 pixels. Second, I reduced the size by 1-100%/5 = 80%. This created images of 20x64x3 pixels. Third, I reduced the size by 1-100%/10 = 90%. This created images of 10x32x3pixels. 

The attached folder "Supplementary material" contains a file called "training_data_preprocessed_visualization.png". This file displays sample images of (a) an original image, (b) focused and 50% size reduction, (c) focused and 80% size reduction, and (d) focused and 90% reduction.

Note: My final model used the 80% size reduction. Unfortunately, my laptop was not able to store the desired number of images if they are only reduced by 50%. However, 80% reduction allowed me to create a model that is able to steer the car through track 1 without hitting the curbs. 

##Split data: 
Second, the data is split into training (63%), validation (27%), and testing data (10%). 

I used very little testing data, because I noticed that the accuracy is a rather bad indicator. In fact, accuracy got really high (up to 100%) when I loaded many images in which the car is simply driving straight (i.e., steering angle is 0). However, this obviously is a huge overfitting issue. Even in curves, many images have steering angle 0 because when driving through the track there is no “very hard left/right” curve. In fact, you can drive through all curves with only hitting left/right very shortly. Thus, there are always several datapoints collected within curves during which cars are driving “straight”. In the forum, several students recommended gamepads to overcome this issue because gamepads allow to steer “softly left” and “softly right”. However, I used the alternative approach: increase the training data and it worked. 

##Model parameters:
I used 25 epochs and a batch size of 32. Before this, I tried a batch sizes of 128 and 64, but the final model was not able to drive one lap without leaving the track. I did not try other amounts of epochs because 25 epochs worked well. 

##Model layers:
Following the NVIDIA model, I used 5 convolutional layers of sizes 48, 48, 32, 20, and 12. I also adopted the kernel sizes from the NVIDIA paper. That is, convolutional layers 1-3 use 5x5 kernel size and convolutional layers 4-5 use 3x3 kernel size. In detail, the resulting image size per layer is as follows:

Image - 20x64x3 
Convolution layer 1: 48, 5x5 kernel size - 16x60x48 
Convolution layer 2: 48, 5x5 kernel size - 12x56x48 
Convolution layer 3: 32, 5x5 kernel size - 8x52x32 
Convolution layer 4: 20, 3x3 kernel size - 6x50x20 
Convolution layer 5: 12, 3x3 kernel size - 4x48x12 
Max. pooling: 2x2 - 2x24x12 
Flatten - 576 

Note: further, detailed information is also provided by the keras output.

After these convolutional layers, I used max pooling with a pool size of 2x2, 25% drop-out to reduce overfitting risk, a then a flatten layer. The output of this layer is a feature of size 576. After this, I added dense layers to further reduce the size to 16, a second drop-out layer (50%) and a final dense layer to reduce the feature size to 1. 

The total parameter number of my model was 117473. 

Note that besides drop-out layers, the model also used additional training data from track 2 and data from driving track 1 and 2 backwards. This training data also helps to avoid overfitting because it increases the probability that the model is rather generalizable than tailored to one specific track. 

##Model training: 
I used Adam MSE to train the model. The model is stored in model.json and the weights of the trained model are stored in model.h5. 

#Discussion of trained model:
The accuracy is relatively low compared to other exercises. Depending on the ratio of curves in the loaded training images, accuracy was between 40% (almost only driving curves) and 100% (almost only driving straight). This is not surprising because predicting the exact steering angle is very difficult. However, the model was able to drive the car through track 1 without hitting the curbs. This indicates that the simulator does not need a perfect steering angle prediction for each datapoint for driving the car on track 1. Although the goal was to create a model for track 1, I also tested the model on track 2. Unfortunately, the model was not able to drive track 2 without crashing into the left side or right side of the track. 

(Note: NVIDIA does not report any accuracy measures in their paper. Thus, it is difficult to interpret my accuracy values. However, I noticed that I had the same issues that NVIDIA mentions when they argue that “to remove bias towards driving straight the training data includes a higher proportion of frames that represent road curves”. [paper p.5])

In addition, it was interesting that training the model only with training data collected by myself did not create a model that was able to steer the car through track 1 without leaving the track. Similar, training the model only with Udacity’s training data was also not enough. In both cases, the car left the track in the left curve after the bridge. Only when using both, my own data as well as Udacity’s data, the model was able to steer the car through the track. (I attached the model based on only Udacity’s data and only my data in the submission.)

#Further minor notes: 
-	For further information, code parts are commented. In my opinion this is more convenient and detailed for readers than describing all code in the README.md file. 

