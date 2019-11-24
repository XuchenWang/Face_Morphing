'''
  File name: morph_tri.py
  Author: Xuchen Wang
  Date created: Oct 12, 2019
'''

import numpy as np
from click_correspondences import click_correspondences
from morph_tri import morph_tri
import imageio
import scipy.misc
from PIL import Image

def main(im1, im2):
    frames = 10

    h1 = im1.shape[0]-1
    w1 = im1.shape[1]-1
    mid1h = np.round(h1/2)
    mid1w = np.round(w1/2)
    surround1 = np.array([[0,0],[w1,h1],[0,h1],[w1,0],[0,mid1h],[mid1w,0],[w1,mid1h],[mid1w,h1]])
    h2 = im2.shape[0]-1
    w2 = im2.shape[1]-1
    mid2h = np.round(h2/2)
    mid2w = np.round(w2/2)
    surround2 = np.array([[0,0],[w2,h2],[0,h2],[w2,0],[0,mid2h],[mid2w,0],[w2,mid2h],[mid2w,h2]])
    im1_pts, im2_pts = click_correspondences(im1, im2)
    print("im1_pts: ", im1_pts)
    print("im2_pts: ", im2_pts)

    # im1_pts = np.array([[146.40322581, 22.82258065],
    #                     [220.91935484,  61.14516129],
    #                     [236.88709677, 121.82258065],
    #                     [227.83870968, 188.35483871],
    #                     [185.25806452, 239.4516129 ],
    #                     [119.25806452, 206.98387097],
    #                     [ 97.43548387, 137.79032258],
    #                     [ 67.09677419, 138.85483871],
    #                     [ 76.67741935,  72.32258065],
    #                     [166.62903226,  91.48387097],
    #                     [151.72580645, 119.16129032],
    #                     [190.58064516, 113.83870968],
    #                     [141.61290323, 140.4516129 ],
    #                     [175.14516129, 132.46774194],
    #                     [201.75806452, 133.53225806],
    #                     [178.87096774, 167.59677419],
    #                     [154.38709677, 198.46774194],
    #                     [201.22580645, 192.08064516],
    #                     [ 68.16129032, 225.61290323],
    #                     [167.16129032, 274.58064516],
    #                     [241.67741935, 271.91935484],
    #                     [139.48387097, 163.87096774],
    #                     [217.19354839, 152.16129032]])
    #
    # im2_pts = np.array([[148.87096774,  38.25806452],
    #                     [204.22580645,  69.12903226],
    #                     [209.01612903, 121.82258065],
    #                     [197.83870968, 159.61290323],
    #                     [157.91935484, 187.29032258],
    #                     [120.12903226, 171.85483871],
    #                     [ 98.83870968, 126.61290323],
    #                     [ 83.93548387, 123.9516129 ],
    #                     [ 91.38709677,  69.66129032],
    #                     [152.59677419,  96.27419355],
    #                     [141.9516129,  113.30645161],
    #                     [166.43548387, 112.24193548],
    #                     [130.24193548, 134.06451613],
    #                     [158.4516129,  128.20967742],
    #                     [181.87096774, 131.93548387],
    #                     [160.58064516, 147.37096774],
    #                     [141.9516129,  164.40322581],
    #                     [173.35483871, 163.87096774],
    #                     [116.40322581, 201.12903226],
    #                     [148.33870968, 205.91935484],
    #                     [205.82258065, 210.17741935],
    #                     [131.83870968, 146.30645161],
    #                     [186.66129032, 141.51612903]])
    #

    im1_pts = np.append(im1_pts, surround1, axis=0)
    im2_pts = np.append(im2_pts, surround2, axis=0)

    warp_frac = np.linspace(0.0, 1.0, num=frames)
    dissolve_frac = np.linspace(0.0, 1.0, num=frames)
    morph_im = morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac)

    morph_list = []
    for i in range(len(warp_frac)):
        # image = im.fromarray(morph_im[i, :, :, :], 'RGB')
        morph_list.append(morph_im[i, :, :, :])
        imageio.mimsave('./eval_morphimg.gif', morph_list)



if __name__ == "__main__":
    im1 = np.array(Image.open('pic_1.jpg').convert('RGB'))
    im2 = np.array(Image.open('pic_2.jpg').convert('RGB'))
    img1 = scipy.misc.imresize(im1,[300,300])
    img2 = scipy.misc.imresize(im2,[300,300])
    main(img1, img2)

