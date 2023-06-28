import cv2
import os
import scipy
import skimage
import time
import re
import ast
import io

import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path
from tqdm import tqdm
from itertools import product, combinations
from scipy.ndimage import convolve
from skimage.measure import label, regionprops_table
from skimage.morphology import reconstruction
from numpy import matlib


def aruco_display(corners, ids, rejected, image):
    
    if len(corners) > 0:
		
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            print(corners)           
            (topL, topR, bottomR, bottomL) = corners
           

            tr = (int(topR[0]), int(topR[1]))
            br = (int(bottomR[0]), int(bottomR[1]))
            bl = (int(bottomL[0]), int(bottomL[1]))
            tl = (int(topL[0]), int(topL[1]))

            return image, tr, br, bl, tl
        


def draw_markers(corners, ids, rejected, image):
    if len(corners) > 0:
		
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            print(corners)           
            (topL, topR, bottomR, bottomL) = corners
           

            tr = (int(topR[0]), int(topR[1]))
            br = (int(bottomR[0]), int(bottomR[1]))
            bl = (int(bottomL[0]), int(bottomL[1]))
            tl = (int(topL[0]), int(topL[1]))
            
            cv2.line(image, tl, tr, (0,255,0), 10)
            cv2.line(image, tr, br, (0,255,0), 10)
            cv2.line(image, br, bl, (0,255,0), 10)
            cv2.line(image, bl, tl, (0,255,0), 10)

            cX= int((tl[0] + br[0] + tr[0]  + bl[0])/4.0)
            cY= int((tl[1] + br[1] + tr[1]  + bl[1])/4.0)
            cv2.circle(image, (cX, cY), 20, (0,0,255), -1)
            cv2.putText(image, str(markerID), (tl[0], tl[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0,255,0), 2)

    return image


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (lm1, lm2) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    (rm1, rm2) = rightMost[np.argsort(rightMost[:, 1]), :]
    return np.array([lm1, lm2, rm1, rm2])



def get_displacements(h_matrix, dest_cn, p_length) : 

        # x value를 기준으로 sort 진행
        dest_cn = order_points(dest_cn)

        #############################################
        ## dest 전처리
        dest_vec1 = np.array([[dest_cn[0][0]], [dest_cn[0][1]], [1]])
        dest_vec2 = np.array([[dest_cn[1][0]], [dest_cn[1][1]], [1]])
        dest_vec3 = np.array([[dest_cn[2][0]], [dest_cn[2][1]], [1]])
        dest_vec4 = np.array([[dest_cn[3][0]], [dest_cn[3][1]], [1]])

        ## dest에 h_matrix 적용 (with inner product)
        dist_1 = np.dot(h_matrix, dest_vec1)
        dist_1 = dist_1 / dist_1[2]
        
        dist_2 = np.dot(h_matrix, dest_vec2)
        dist_2 = dist_2 / dist_2[2]

        dist_3 = np.dot(h_matrix, dest_vec3)
        dist_3 = dist_3 / dist_3[2]

        dist_4 = np.dot(h_matrix, dest_vec4)
        dist_4 = dist_4 / dist_4[2]

        ## 결과 도출 - result
        result = (dist_1 + dist_2 + dist_3 + dist_4) / 4 - np.array([[p_length/2], [p_length/2], [1]])

        return result[:2]   




def get_homography_transform(corners, p_length) : 
        
        cn_matrix = np.array([
            [np.array(corners[0][0]), np.array(corners[0][1])],
            [np.array(corners[1][0]), np.array(corners[1][1])],
            [np.array(corners[2][0]), np.array(corners[2][1])],
            [np.array(corners[3][0]), np.array(corners[3][1])]
        ])
        
        # x value를 기준으로 sort 진행
        cn_matrix = order_points(cn_matrix)


        # p_length에 따른 weight matrix 생성
        wc = np.array([
            [0, 0],
            [0, p_length],
            [p_length, 0],
            [p_length, p_length]
        ])
        
        # findHomography(src, desc, ...) 이용해서 호모그래피 matrix 찾기
        h_matrix = findProjectiveTransform(cn_matrix, wc).T

        return h_matrix    





def homography_transformation(corners, dest_cn, p_length) : 
        
        cn_matrix = np.array([
            [np.array(corners[0][0]), np.array(corners[0][1])],
            [np.array(corners[1][0]), np.array(corners[1][1])],
            [np.array(corners[2][0]), np.array(corners[2][1])],
            [np.array(corners[3][0]), np.array(corners[3][1])]
        ])
        
        # x value를 기준으로 sort 진행
        cn_matrix = order_points(cn_matrix)
        dest_cn = order_points(dest_cn)


        # p_length에 따른 weight matrix 생성
        wc = np.array([
            [0, 0],
            [0, p_length],
            [p_length, 0],
            [p_length, p_length]
        ])
        
        # findHomography(src, desc, ...) 이용해서 호모그래피 matrix 찾기
        h_matrix = findProjectiveTransform(cn_matrix, wc).T

        

        
        #############################################
        ## dest 전처리
        dest_vec1 = np.array([[dest_cn[0][0]], [dest_cn[0][1]], [1]])
        dest_vec2 = np.array([[dest_cn[1][0]], [dest_cn[1][1]], [1]])
        dest_vec3 = np.array([[dest_cn[2][0]], [dest_cn[2][1]], [1]])
        dest_vec4 = np.array([[dest_cn[3][0]], [dest_cn[3][1]], [1]])

        ## dest에 h_matrix 적용 (with inner product)
        dist_1 = np.dot(h_matrix, dest_vec1)
        dist_1 = dist_1 / dist_1[2]
        
        dist_2 = np.dot(h_matrix, dest_vec2)
        dist_2 = dist_2 / dist_2[2]

        dist_3 = np.dot(h_matrix, dest_vec3)
        dist_3 = dist_3 / dist_3[2]

        dist_4 = np.dot(h_matrix, dest_vec4)
        dist_4 = dist_4 / dist_4[2]

        ## 결과 도출 - result
        result = (dist_1 + dist_2 + dist_3 + dist_4) / 4 - np.array([[p_length/2], [p_length/2], [1]])

        return h_matrix, result[:2]     



def findProjectiveTransform(uv, xy):

    """
    This code is a Python implementation of Matlab code
    
    'findProjectiveTransform' in 'fitgeotrans' 
    
    Reference : to-be added here
    
    """
        
    uv, normMatrix1 = normalizeControlPoints(uv);
    xy, normMatrix2 = normalizeControlPoints(xy);

    minRequiredNonCollinearPairs = 4;
    M = xy.shape[0]
    x = xy[:, 0][:, np.newaxis]
    y = xy[:, 1][:, np.newaxis]
    vec_1 = np.ones((M, 1))
    vec_0 = np.zeros((M, 1))
    u = uv[:, 0][:, np.newaxis]
    v = uv[:, 1][:, np.newaxis]

    U = np.vstack([u, v])
    X = np.hstack([[x, y, vec_1, vec_0, vec_0, vec_0, np.multiply(-u, x), np.multiply(-u, y)],
                   [vec_0, vec_0, vec_0, x, y, vec_1, np.multiply(-v, x), np.multiply(-v, y)]]).squeeze().T

    # We know that X * Tvec = U
    if np.linalg.matrix_rank(X) >= 2 * minRequiredNonCollinearPairs:
        Tvec = np.linalg.lstsq(X, U, rcond=-1)[0]
    #     else :
    #         error(message('images:geotrans:requiredNonCollinearPoints', minRequiredNonCollinearPairs, 'projective'))

    #      We assumed I = 1;
    Tvec = np.append(Tvec, 1)
    Tinv = np.reshape(Tvec, (3, 3));

    Tinv = np.linalg.lstsq(normMatrix2, np.matmul(Tinv, normMatrix1), rcond=-1)[0]
    T = np.linalg.inv(Tinv)
    T = T / T[2, 2]

    return T



def normalizeControlPoints(pts):

    # Define N, the number of control points
    N = pts.shape[0]

    # Compute [xCentroid,yCentroid]
    cent = np.mean(pts, 0)

    # Shift centroid of the input points to the origin.
    ptsNorm = np.zeros((4, 2))
    ptsNorm[:, 0] = pts[:, 0] - cent[0]
    ptsNorm[:, 1] = pts[:, 1] - cent[1]

    sumOfPointDistancesFromOriginSquared = np.sum(np.power(np.hypot(ptsNorm[:, 0], ptsNorm[:, 1]), 2))

    if sumOfPointDistancesFromOriginSquared > 0:
        scaleFactor = np.sqrt(2 * N) / np.sqrt(sumOfPointDistancesFromOriginSquared)
        
    #     else:
    # % If all input control points are at the same location, the denominator
    # % of the scale factor goes to 0. Don't rescale in this case.
    #         if isa(pts,'single'):
    #             scaleFactor = single(1);
    #         else:
    #             scaleFactor = 1.0;
    # % Scale control points by a common scalar scale factor
    
    ptsNorm = np.multiply(ptsNorm, scaleFactor)

    normMatrixInv = np.array([
        [1 / scaleFactor, 0, 0],
        [0, 1 / scaleFactor, 0],
        [cent[0], cent[1], 1]])

    return ptsNorm, normMatrixInv