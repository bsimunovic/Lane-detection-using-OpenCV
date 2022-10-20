# Lane-detection-using-OpenCV
Basic image segmentation for lane detection

The problem of this paper was the description of the work and the creation of an algorithm 
for the detection of traffic lanes. OpenCV, Matplotlib and NumPy libraries are used for creation 
of the algorithm. It is described in a few steps, the most important of which is image processing. 
First, it is essential to transform perspective. After that, the image is processed using the HSL color 
space and the noise is reduced with that. This results in a binary image and the application of the 
Canny edge detection algorithm. To describe the image processing, routs representing traffic lanes 
are then found using the Hough transform. From the obtained routs, one average rout is created in 
order to show the best possible traffic lanes. Finally, we return the filtered image to normal 
perspective and merge it with the input image. This is a brief description of the algorithm, after 
which tests are performed. The algorithm is fast enough to run in real time, but not efficient enough 
for use in real traffic.
