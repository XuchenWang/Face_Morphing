# Xuchen Wang - CIS 581 Project 2

This project morphes faces from one to another. The result will be in a gif animation. 

## Getting Started

The folder contains pictures and scripts: 

"face_eval-20.gif" and "face_eval-60.gif" are the two morphing results from the source images, "face_1.png" and "face_2.png". The only difference between the two gif's is that "face_eval-20.gif" has 20 frames and "face_eval-60.gif" has 60 frames. "cat_eval-20.gif" is the morphing result from the source images, "cat_1.png" and "cat_2.png".

For scripts, faceMorphing.py is the main file that puts everything together and generate the gif in the end. cpselect.py and interp.py are the python file provided by the TA. click_correspondences.py calls cpselect.py and lets the user click the corresponding important points (like eyes, nose, and mouth) on the two images. morph_tri.py warps the two images first and then cross dissolve them. 

## Running the tests

Go to faceMorphing.py. Change variables 'im1' and 'im2' to the image you would like to morph. Change variable 'frame' to the number of frames you wish to generate. The default value is 60, which may cause a longer time to generate the gif. 

The im1_pts and im2_pts are the corresponding important points between "face_1.png" and "face_2.png". When the two source images are "face_1.png" and "face_2.png", you could choose the points yourself by calling click_correspondences.py, or you could also uncomment out the coordinator list in faceMorphing.py. 

Run morph_tri.py and check your folder for the morphing gif, 'eval_morphimg.gif'.

## Acknowledgments

* Collaberated with Lishuo Pan for this project.

