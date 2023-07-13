# Image-Stitching

I implemented Image-Stitching as a term project.

The goal of the project is to implement Image Stitchin and to implement it directly without using built-in functions and to increase understanding.

First of all, Level A1a is to determine the direction of the two input images. It was solved by an algorithm that determines the direction using thresholding in the Homography matrix.

Level A1b is the task of stitching the four input images entered in order. Since no information on the size and height of the image was given, I sliced and solved it with an algorithm that was stitched in the order of the bottom → the top.

Level 2 is the task of stitching four input images that are entered without order information. As shown in the algorithm flow chart below, I solved the case by dividing it into three.
