**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####0. General information for the reviewer

First, all my code is written in the submitted Jupyter notebook. 

Second, I want to acknowledge that many functions in the notebook are extended versions of functions introduced in the Udacity "Vehicle Detection and Tracking" lecture. 

Third, I create an object called Params. It is declared in the third cell in the notebook. This object contains all parameters (excep paths and file names). Thus, whenever I refer to parameter in the following, I am refering to the Params object.

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.


The function classify() trains my model and starts by calling the function extract_features(). This function reads the images. In particular, the vehicle- and non-vehicle images are read using mpimg from matplotlib.image. After reading each image, the function single_img_features() is called on that image which, besides others, calls my function get_hog_features(). This function is very similar to the get_hog_features() function from class. That is, it calls the hog() function of the scikit-image package. Furthermore, my Params-object provides the orientations, pixel per cell, and cells per block values. 

My get_hog_features() function is called by the functions single_img_features() and hog_visualization(). The function hog_visualiation could have been integrated in get_hog_features(). However, I created a separate function, because the get_hog_features() function is already quite "loaded" and, in particular within a single Jupyter notebook, I believe that it is simply clearer to provide a separate function in case all hog-channels across all color-spaces shall be displayed (--> this is what hog_visualization() is doing).


I started by reading in all the `vehicle` and `non-vehicle` images.  The Jupyter notebook shows an example vehicle- and non-vehicle image as well as the three hog-channels for each color-space of those two example images. 


####2. Explain how you settled on your final choice of HOG parameters.


I ran the notebook several time and noticed that the YCrCb appeared to be performing best overall. I then tried a some parameter combinations and ended up with orient=8 (tried also 9), pixels_per_cell=(8,8) (tried also 16,16), and cells_per_block=(2,2) (tried also 4,4).

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

My function classify() represents my training-function. Although the Udacity class did not introduce a so-called classify() function, the function is based on several lines of code from the lessons 26 (Color classify), lesson 27 (Hog classify), and lesson 32 (Search and classify). 

After receiveing the features from my extract_features() function, I use a StandardScaler and then split the dataset randomly into a trainset and testset. I use the same ratio as in class (i.e., 80% train and 20% test).

After that, a linear SVC model is created and trained. The accuracy on the training data is 0.9864 and the accuracy on the test data is 0.9827.

Short remark: 
Yes, I also used color-features in addition to my hog features. In fact, the function extract_features() runs the function single_img_features() on every image. The function single_img_features(), in turn, runs my bin_spatial(), color_his(), and get_hog_features() functions. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My function images_process_func() processes all test images and video frames. Besides other things, it calls the function search_image(). The function search_image() then runs the function slide_window() with all possible combinations of my parameters xy_windows and xy_overlaps (as provided by my Params-object). Howevery, it only searches in the lower half of the image because cars are more likely to be driving on the street than flying in the sky :=)

In particular, the function slide_window() returns subimages (i.e., "sub-windows") back to the function search_image(). These "sub-windows" are then provided to the function search_windows_with_cars_in_image() which only returns those "sub-windows" that show cars.

I started with relatively large image sizes for sliding (i.e., 256x256) but then reduced it further as I noticed that smaller sizes yield better result on the test images and the test video. 

Regarding overlap, I started with 50% overlap on both axes because 50% was used in class. I reduced this value to 25% and also increased to 75% and 100%. However, the results did not noticeably improve on the test images and the test video. Thus, I decided to go back to 50% overlap and focus on the other parameters. 

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The Jupyter notebook shows the following output for each of the test images: 
(1) images with identified boxes, (2.1) image with heatmap, (2.2) image with heatmap and inclusion threshold being applied, (2.3) image with car labels, and (3) image with boxes around cars (i.e., the final output image). 

Although both, the training accuracy as well as the test accuracy of my classifier were very high (i.e., above 98% accuracy), false positives seem to be a big problem when running my model on the images and videos. Therefore, to optimize my classifier, I used the decision_function() function which is recommended in the forums. Also, I tried several C-values for my SVC-model and found that 0.0004 performs relatively good.

---

### Video Implementation

####1. Provide a link to your final video output.  

Test video: https://youtu.be/cZ3YDEd4yMo

Full project video: https://youtu.be/RNMLu_xT8vw 

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I generated heatmaps of car labels (i.e., "sub-windows" for which the inclusion threshold was met). These heatmaps improved the results remarkably. However, when looking at the heatmap visualizations, I can see that often trees and very straight lines (either from lane lines or lane boundaries) are relatively "hot" areas in the images besides cars, too.


The Jupyter notebook shows the following output for each of the test images: 
(1) images with identified boxes, (2.1) image with heatmap, (2.2) image with heatmap and inclusion threshold being applied, (2.3) image with car labels, and (3) image with boxes around cars (i.e., the final output image).

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Overall, I mostly had issues wiht false positives. When looking at the heatmap visualizations, I can see that often trees and very straight lines (either from lane lines or lane boundaries) are relatively "hot" areas in the images besides cars. To tackle this issue, I would like to experiment with a combination approach. That is, I would like to combine the linear SVC classifier with a different approach such as a neural net. I believe that NNs could perform particularly well in this project. 

In addition, improvements could leverage the labeled Udacity dataset. This could help train a more generalizable classifier because, although images are randomly divided into training data and testing data, many images are somewhat similar but different from the video frames (which, e.g., show many trees).

