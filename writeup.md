
## Vehicle Detection and Tracking Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in section 1.1 and 1.2 of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

#### 2. Explain how you settled on your final choice of HOG parameters.

I first implemented part of the vehicle detection and tracking pipeline including data preparation and classifier training until the point of multi-detection, i.e., section 2.1 (Detect vehicle regions with sliding windows) so I can directly see the end result when I tune the parameters. I then tried various combinations of parameters, and decided to keep or change the parameters by visually checking the end result. For example, when changing `color_space` from "RGB" to "YCrCb", the vehicle detection in images became much more accurate (I also noted that the improvement of the end result does not necessarily correspond to an imrovement of the accuracy of the trained classifier), not only the actually vehicle location is detected, but false positive is also minimized, thus this paramter is kept.

Eventually the following parameters are chosen (in cell 8):

color_space = 'YCrCb'   # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

orient = 9              # HOG orientations

pix_per_cell = 8        # HOG pixels per cell

cell_per_block = 2      # HOG cells per block

hog_channel = "ALL"     # Can be 0, 1, 2, or "ALL"

spatial_size = (16, 16) # Spatial binning dimensions

hist_bins = 32          # Number of histogram bins

spatial_feat = True     # Spatial features on or off

hist_feat = True        # Histogram features on or off

hog_feat = True         # HOG features on or off

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After reading in all the images I performed (in section 1.3 Data Preparation) the following steps to prepare the data (both training and test data) used for classifier traning:
1) feature extraction 
2) feature normalization 
3) data labeling 
4) data random shuffling

In section 1.4 (Classifier Training), I trained a linear SVM by first creating the SVM `svc = LinearSVC()` (imported from sklearn.svm) and then training it using `svc.fit(X_train, y_train)`. The tested accuracy obtained by `svc.score(X_test, y_test)` is around 0.99. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In section 2.1 (Detect vehicle regions with sliding windows) I implemented a fixed sized sliding window search, with the original window size of 64 and scale 1.5, cells_per_step is chosen to be 2. As stated above, these parameters are also selected in a trial-and-error manner.  


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

Also I implemented a tracking mechanism (in cell 16) to track and filter the detected vehicle's centroid position, velocity and its size. For every new frame, I first (1) predicted the current vehicle position based on its last step position and velocity, then I iterate through the detected vehicles at this step and try to (2) associate the current detection with previous (filtered) detections, then (3) update those vehicles with current step measurement, and fianlly, (4) if some detected vehicle is lost for too long, delete them.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

By including the tracking mechanism the result is much improved. Actually I probably spent more time than I should in the tracking (filtering) part and less time than I should in the detecting part. If I had more time I'd tune/train my classifier to be more robust and improve the sliding window search and classify to make detection more robust also. 


