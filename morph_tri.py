'''
  File name: morph_tri.py
  Author: Xuchen Wang
  Date created: Oct 11, 2019
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''

from scipy.spatial import Delaunay
import numpy as np
from interp import interp2

def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
  h1 = im1.shape[0]
  w1 = im1.shape[1]
  h2 = im2.shape[0]
  w2 = im2.shape[1]
  morphed_im = []

  for t in range(len(warp_frac)):
    warp_value = warp_frac[t]
    dissolve_value = dissolve_frac[t]

    # find triangles for intermediate image
    # print("========intermediate image=========")
    intermediate_pts = (1-warp_value)*im1_pts + warp_value * im2_pts
    Tri = Delaunay(intermediate_pts)
    simplices = Tri.simplices


    # find the inverse of matrix A for each intermediate triangle, #inverse_matrix=#triangles
    # print("========inverse of matrix A for intermediate triangle=========")
    temp = intermediate_pts[simplices].transpose((0,2,1))    #get coordinates of convex points
    temp_1 = np.zeros([simplices.shape[0],3,3])
    for t in range(len(temp)):
      temp_1[t,:,:] = np.linalg.inv(np.vstack((temp[t],[1,1,1])))


    # find matrix A for img 1, #matrix=#triangles
    # print("========find matrix A for img 1=========")
    temp_img1 = im1_pts[simplices].transpose((0,2,1))
    temp_1_img1 = np.zeros([simplices.shape[0],3,3])
    for t in range(len(temp_img1)):
      temp_1_img1[t,:,:] = np.vstack((temp_img1[t],[1,1,1]))

    # find matrix A for img 2, #matrix=#triangles
    # print("========find matrix A for img 2=========")
    temp_img2 = im2_pts[simplices].transpose((0,2,1))
    temp_1_img2 = np.zeros([simplices.shape[0],3,3])
    for t in range(len(temp_img2)):
      temp_1_img2[t,:,:] = np.vstack((temp_img2[t],[1,1,1]))

    # get alpha beta gamma
    # print("========homogeneous coordinators=========")
    x,y = np.meshgrid(np.arange(w1), np.arange(h1))
    x = x.flatten()
    y = y.flatten()
    simplex = Tri.find_simplex(list(zip(x,y)))  # the index of triangle each pixel is in
    pt_A = temp_1[simplex]   #get interm_A_inverse for each point
    alpha = pt_A[:,0,0]*x + pt_A[:,0,1]*y + pt_A[:,0,2]
    beta = pt_A[:,1,0]*x + pt_A[:,1,1]*y + pt_A[:,1,2]
    gamma = pt_A[:,2,0]*x + pt_A[:,2,1]*y + pt_A[:,2,2]


    # get coordinate in im1
    # print("========get coordinate in im1=========")
    pt_A_img1 = temp_1_img1[simplex]   #get im1_A matrix for each point
    x_t_1 = pt_A_img1[:,0,0]*alpha + pt_A_img1[:,0,1]*beta + pt_A_img1[:,0,2]*gamma
    y_t_1 = pt_A_img1[:,1,0]*alpha + pt_A_img1[:,1,1]*beta + pt_A_img1[:,1,2]*gamma
    z_t_1 = alpha + beta + gamma
    x_t_1 = x_t_1/z_t_1
    y_t_1 = y_t_1/z_t_1
    im1_warped = im1.copy()
    im1_warped[:,:,0] = np.reshape(interp2(im1[:,:,0],x_t_1,y_t_1), [h1, w1])
    im1_warped[:,:,1] = np.reshape(interp2(im1[:,:,1],x_t_1,y_t_1), [h1, w1])
    im1_warped[:,:,2] = np.reshape(interp2(im1[:,:,2],x_t_1,y_t_1), [h1, w1])
    # print(sum(im1_warped[:,:,0] == im1[:,:,0]))
    # print(sum(im1_warped[:,:,1] == im1[:,:,1]))
    # print(sum(im1_warped[:,:,2] == im1[:,:,2]))


    # get coordinate in im2
    # print("========get coordinate in im2=========")
    pt_A_img2 = temp_1_img2[simplex]   #get im2_A matrix for each point
    x_t_2 = pt_A_img2[:,0,0]*alpha + pt_A_img2[:,0,1]*beta + pt_A_img2[:,0,2]*gamma
    y_t_2 = pt_A_img2[:,1,0]*alpha + pt_A_img2[:,1,1]*beta + pt_A_img2[:,1,2]*gamma
    z_t_2 = alpha + beta + gamma
    x_t_2 = x_t_2/z_t_2
    y_t_2 = y_t_2/z_t_2
    # im2 = im1
    im2_warped = im2.copy()
    im2_warped[:,:,0] = np.reshape(interp2(im2[:,:,0],x_t_2,y_t_2), [h2, w2])
    im2_warped[:,:,1] = np.reshape(interp2(im2[:,:,1],x_t_2,y_t_2), [h2, w2])
    im2_warped[:,:,2] = np.reshape(interp2(im2[:,:,2],x_t_2,y_t_2), [h2, w2])
    # print(sum(im2_warped[:,:,0] == im2[:,:,0]))
    # print(sum(im2_warped[:,:,1] == im2[:,:,1]))
    # print(sum(im2_warped[:,:,2] == im2[:,:,2]))


    # cross-dissolve
    # print("========cross-dissolve=========")
    morphed = im1_warped.copy()
    morphed[:,:,0] = (1-dissolve_value)*im1_warped[:,:,0] + dissolve_value* im2_warped[:,:,0]
    morphed[:,:,1] = (1-dissolve_value)*im1_warped[:,:,1] + dissolve_value* im2_warped[:,:,1]
    morphed[:,:,2] = (1-dissolve_value)*im1_warped[:,:,2] + dissolve_value* im2_warped[:,:,2]
    # print(sum(morphed[:,:,0] == im1[:,:,0]))
    # print(sum(morphed[:,:,1] == im1[:,:,1]))
    # print(sum(morphed[:,:,2] == im1[:,:,2]))
    # print(sum(morphed[:,:,0] == im2[:,:,0]))
    # print(sum(morphed[:,:,1] == im2[:,:,1]))
    # print(sum(morphed[:,:,2] == im2[:,:,2]))
    # print(morphed.max(), morphed.min())

    # putting together the morphed images
    # print("========put together=========")
    morphed_im.append(morphed)

  morphed_im = np.array(morphed_im)
  return morphed_im.astype('uint8')
