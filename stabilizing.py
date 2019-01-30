import cv2
import numpy as np
import copy

class Tracker():
    def __init__(self,image):
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
        dst = cv2.warpAffine(image, affine_matrix, (image.shape[1], image.shape[0]),flags=cv2.INTER_CUBIC)
        return dst

if __name__ == "__main__":
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
                new_image = tracker.calc_affine(original_image)
            else:
                tracker = Tracker(copy.deepcopy(original_image))
            cv2.imshow(windows[0], original_image)
            cv2.imshow(windows[1], new_image)
            cv2.imshow(windows[2], cv2.addWeighted(new_image,0.5,cv2.cvtColor(tracker.reference_image,cv2.COLOR_GRAY2BGR),0.5,1))#   new_image)
            for window in windows:
                cv2.resizeWindow(window, 640, 360)
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            if k == 32:
                tracker.change_reference(original_image)
    
