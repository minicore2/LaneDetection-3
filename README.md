# LaneDetection

## Purpose
Detect the lane lines and overlay them on the camera images.

## Methods
It uses a series of computer vision algorithms for the lane line detection task. 
the pipeline are listed as below:

	1. Cropping image to the most valid region
	2. Enhancing image constrast
	3. Performing color thresholding and obtaining the color mask
	4. Projecting the image to the ground plane
	5. Obtaining edges using 2D line filtering
	6. Scanning vertically to extract right and left lane points
	7. Spline RANSAC fitting
	8. Stablizing the predicted lanes (from noisy environment) using Kalman filter
	
## Dependencies
	- OpenCV
	
	