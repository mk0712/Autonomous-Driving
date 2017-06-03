#Udacity Self-Driving Car Nanodegree
#Term 1, Project 4: Advanced Lane Finding

#Introduction 
The goal of this project is to create a program pipeline that identifies lane boundaries from a car's front-facing camera. 
In particular, the project first uses a set of chessboard images to learn the camera's distortion coefficients. 
Subsequently, the pipeline is built. It first uses color transforms to create a thresholded binary image and then conducts perspective transformation.
Next, the pipeline searches for lane pixels and determines lane lines. It also determines radius of the lane and the driver's distance from the center of the lane.
The lanes (i.e., lane area, left lane boundary, right lane boundary) are warped back on the original video stream, which, together with curve radius and center distance information, is stored as output.

#YouTube
The final output video is provided together with the submission. In addition, it is also uploaded to following URL:
https://youtu.be/sbG_FAUjBSw

#List of submitted files:
- README.md 
     - Explains the code.
- ale.py, ale.ipynb
     - The script/Jupyter notebook used to create and train the model. Note: The Jupyter notebook also shows the processed test images!
- project_video_result.mp4
     - The generated video (the same video is also uploaded on YouTube; see above)

	 
# Project Description

## Setup
Definition of all path variables and inclusion of all supplementary packages. 

## Distortion
The chessboard test images are loaded and transformed to grayscale. 
A 6*9 cell chessboard is "searched" using cv2.findChessboardCorders(...) and cv2.calibrateCamera(...). 
The results are used to distort the test images using cv2.undistort(...).
The results are visualized together with the original image.

## Pipeline Preparation
Functions for identification of linear lane segments (i.e., polynomial degree = 1, https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html) and a data storage object for the lanes are defined.

## Pipeline
The pipeline is the function getImageWithProjectionAndText( source_image ). 

The interesting areas of the screen (i.e., bottom left, bottom right) for finding potential lanes is defined. Distance estimates are introduced and params are defined (e.g., pixel margin, min lane pixels for segment etc).

Next, the video frame is distorted using the distortion coefficients for the front-facing camera.

Next, the video frame is converted to grayscale and the Sobel filter is used to determine image gradients and edges. In particular, cv2.Sobel(...) is used. The usage follows cv2's recommendations (http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html).

Next, using cv2.bitwise_or(...) a binary image is generated.

Next, the image is transformed using cv2.perspectiveTransform(...) and, subsequently, warped using cv2.warpPerspective(...). Empty warp projections are generated that will be used to draw lanes boundaries and the lane area. 

Next, the lanes are searched and if found put on the warped slides (first left lane, then right lane). Lane curves are fitted using the polynomial of degree 2 [--> np.polyfit(...)]. https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html

In addition, the curve radius is estimated for each lane separately and, then, averaged and multiplied by the estimated distance-factor (i.e., 300 pixel vertically divided by 15m vertically). For the lanes (left, right) and the lane area (between the lane boundaries), three warps are created. The lane area is filled using cv2.fillPoly(...). Finally, the three warps are stacked together. The lane area uses 50% transparency and the lane borders 20%.

Finally, the coords of the car are computed using the center of the image in pixel and the estimated horizontal factor (--> horizontal_factor = 0.6*1280*2.5 #1280 pixel * 0.6 because lane is approx. 60% of screen at bottom * 2.5 because lane is approx. 2.5 wide).

The curve degree and distance to center are written on the image using cv2.putText(...).

Finally, the image is returned as the pipeline's result.

## Image Test
The function img_test(path) for running the pipeline and showing the result is introduced and ran on all 6 test images. All results look good and can be viewed in the Jupyter notebook or when executing the code again.

## Running Pipeline on Video
The video stream is loaded using cv2.VideoCapture(...).read(). In particular, each frame is read individually and the pipeline is executed for each frame. 
After running the pipeline (approx. 10 minutes), MoviePy is used to store the resulting video.
The video is included in the Jupyter notebook, the submission, and uploaded to YouTube (see above for URL).
