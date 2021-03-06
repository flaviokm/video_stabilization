import cv2
import numpy as np
import copy
import math
from copy import deepcopy
from filters import KalmanFilter
import argparse

class Tracker():
    def __init__(self,image,fit = False):
        self.reference_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        self.feature_params =  dict(maxCorners=100,
                                                         qualityLevel=0.2,
                                                         minDistance=7,
                                                         blockSize=7)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=6,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.reference_points = cv2.goodFeaturesToTrack(
            self.reference_image, mask=None, **self.feature_params)
        self.affine_matrix = None
        self.kalman_filter = None
        self.fix_fit  = fit
        self.fit_angle = 30

    def change_reference(self,image):
        self.reference_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        self.reference_points = cv2.goodFeaturesToTrack(
            self.reference_image, mask=None, **self.feature_params)

    def check_matrix(self,matrix):
        if matrix is None:
            matrix = self.affine_matrix
        else: 
            self.affine_matrix = copy.deepcopy(matrix)
        return matrix

    def calc_affine(self,image):
        dst = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.reference_image, gray, self.reference_points, None, **self.lk_params)
        affine_matrix = cv2.estimateRigidTransform(p1,self.reference_points,fullAffine=False)
        affine_matrix = self.check_matrix(affine_matrix)
        if self.kalman_filter is not None:
            old_affine = deepcopy(affine_matrix)
            affine_matrix = self.kalman_filter.apply(affine_matrix)
            # print ("difference:{}".format(np.sum(old_affine-affine_matrix)))
        else :
            self.kalman_filter = KalmanFilter(affine_matrix,affine_matrix.shape)
            affine_matrix = self.kalman_filter.apply(affine_matrix)
        translation = affine_matrix[:,-1]
        rotation = affine_matrix[:,:-1]
        rotation_angle = math.asin(1.0) if rotation[0,1]>1.0 else math.asin(rotation[0,1]) if rotation[0,1] > -1.0 else math.asin(-1.0)
        dst = cv2.warpAffine(image, affine_matrix, (image.shape[1], image.shape[0]),flags=cv2.INTER_AREA)
        if self.fix_fit:
            return self.fit_rotation(dst,self.fit_angle*math.pi/180.0)
        else:
            return dst,(dst.shape[1],dst.shape[0])

    def fit_rotation(self,image,angle):
        def rotatedRectWithMaxArea(w, h, angle):
            """
            from stackoverflow user coproc
            Given a rectangle of size wxh that has been rotated by 'angle' (in
            radians), computes the width and height of the largest possible
            axis-aligned rectangle (maximal area) within the rotated rectangle.
            """
            if w <= 0 or h <= 0:
                return 0,0
            width_is_longer = w >= h
            side_long, side_short = (w,h) if width_is_longer else (h,w)
            sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
            if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
                x = 0.5*side_short
                wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
            else:
                cos_2a = cos_a*cos_a - sin_a*sin_a
                wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
            return wr,hr
        h,w = image.shape[0],image.shape[1]
        center = (w//2,h//2)
        wn,hn = rotatedRectWithMaxArea(w, h, -angle)
        x1,y1 = int(center[0]-wn//2),int(center[1]-hn//2)
        x2,y2 = int(center[0]+wn//2),int(center[1]+hn//2)
        return cv2.resize(image[y1:y2,x1:x2],(w,h), interpolation=cv2.INTER_AREA),(int(wn),int(hn))

    def adjust_fitted_to_show(self,image,size):
        image_superposted = np.zeros(image.shape)
        resized_image = cv2.resize(image,size)
        x0 = image_superposted.shape[0]//2-resized_image.shape[0]//2
        y0 = image_superposted.shape[1]//2-resized_image.shape[1]//2
        image_superposted[x0:x0+resized_image.shape[0],y0:y0+resized_image.shape[1]] = resized_image
        return image_superposted.astype(image.dtype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Video Stabilizer')
    parser.add_argument(
        '-f','--fit',
        nargs='?', const='', default='', type=bool,
        help='fix the fited window')
    args = parser.parse_args()

    windows = ['image','new_image','comparison']
    for win_num,window in enumerate(windows):
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window, win_num*640, 0) 
    tracker = None
    cap = cv2.VideoCapture(1)
    while(1):
        ret ,original_image = cap.read()
        if ret == True:
            new_image = copy.deepcopy(original_image)
            if tracker is not None: 
                new_image,size = tracker.calc_affine(original_image)
                cv2.imshow(windows[0], original_image)
                cv2.imshow(windows[1], new_image)
                if args.fit:
                    cv2.imshow(windows[2], cv2.addWeighted(tracker.adjust_fitted_to_show(new_image,size),0.5,cv2.cvtColor(tracker.reference_image,cv2.COLOR_GRAY2BGR),0.5,1))#   new_image)
                else:
                    cv2.imshow(windows[2], cv2.addWeighted(new_image,0.5,cv2.cvtColor(tracker.reference_image,cv2.COLOR_GRAY2BGR),0.5,1))#   new_image)
                for window in windows:
                    cv2.resizeWindow(window, 640, 360)
                k = cv2.waitKey(60) & 0xff
                if k == 27:
                    break
                if k == 32:
                    tracker.change_reference(original_image)
            else:
                tracker = Tracker(copy.deepcopy(original_image),fit = args.fit)
           
