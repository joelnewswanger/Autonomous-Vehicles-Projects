# **Finding Lane Lines on the Road**

### The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on the work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Description of pipeline

My pipeline() function consisted of 6 steps. First, I converted the images to grayscale and applied a gaussian blur with a kernel size of 5. Then, I applied the Canny transform with a lower threshold of 75 and upper threshold of 150. I selected separate areas of interest for the left and right lane lines, and fed both into the Hough lines function. The Hough lines function sends final lines to the draw_lines() function, which draws them onto a canvas the size of the original picture. The weighted_img() function then overlays the lines onto the original picture, producing the final product returned by pipeline().

I modified the Hough lines function to find the lines separately for the left and right, and for each side to find the average slope and x coordinate at y=400 for all lines with a slope of 70 +- 20. I then extrapolated along the average slope from the average (x,400) point to the horizon (x,325) and the bottom of the picture (x,540), and sent this line to the draw_lines() function.



### 2. Potential shortcomings of current pipeline


One potential shortcoming is the heavy reliance on areas of interest. If the car were to become misaligned with the road it could loose sight of the lines. Additionally, if other lines (cracks/lines in asphalt, debris, etc.) come into the area of interest they are not well filtered out.

Another shortcoming is that currently if it cannot find a line in a frame it throws an error and quits. It would be important to handle the exception in a better way before implementing this pipeline.


### 3. Possible improvements to the pipeline

A possible improvement would be to guess based on previous frames if a line cannot be found in a frame, or at least continue on to the next frame without throwing an error.

Another potential improvement could be to use machine learning to determine what lines are lane lines and which are distractions. * See project "Advanced lane finding" *
