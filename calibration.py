"""
 Based on the following tutorials:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
   https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
   https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
   https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html
"""
import numpy as np
import cv2
import glob

import os
import matplotlib.pyplot as plt
import pickle

# Define the rows and columns
rows = 2
cols = 2

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('calibration/IMG_*.jpg')

# Write code that makes use of OpenCV to load each image file in turn
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # # find Harris corners
    # gray = np.float32(gray)
    # dst = cv2.cornerHarris(gray, 5, 3, 0.04)
    # dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # dst = cv2.dilate(dst, None)
    # # Threshold for an optimal value, it may vary depending on the image.
    # ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    # ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    # dst = np.uint8(dst)
    # # find centroids
    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        # # define the criteria to stop and refine the corners
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        # corners2 = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
        # detect the outer corners of the stamp
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        # result_dir = "contour"
        # outfile = os.path.join(result_dir, os.path.basename(fname))
        # cv2.imwrite(outfile, img)

        cnts = [item for sublist in contours for item in sublist]
        cnts = np.array(cnts)
        rect = cv2.minAreaRect(cnts)
        box = cv2.boxPoints(rect)
        box = np.reshape(box, (4, 1, 2))
        imgpoints.append(box)

        box = np.int0(cv2.boxPoints(rect))
        x1 = box[0][0]
        y1 = box[0][1]
        x2 = box[1][0]
        y2 = box[1][1]
        x3 = box[2][0]
        y3 = box[2][1]
        x4 = box[3][0]
        y4 = box[3][1]
        cv2.circle(img, (x1, y1), 3, (0, 0, 255), -1)
        cv2.circle(img, (x2, y2), 3, (0, 0, 255), -1)
        cv2.circle(img, (x3, y3), 3, (0, 0, 255), -1)
        cv2.circle(img, (x4, y4), 3, (0, 0, 255), -1)
        result_dir = "calibration_results"
        outfile = os.path.join(result_dir, os.path.basename(fname))
        cv2.imwrite(outfile, img)

cv2.destroyAllWindows()

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

images = glob.glob('calibration/IMG_*.jpg')
for idx, fname in enumerate(images):
    # Test undistortion on an image
    img = cv2.imread(fname)
    img_size = (img.shape[1], img.shape[0])
    # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,img_size,1,img_size)
    # undistort
    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # Then use the co-ordinates of those corners to produce (to a good approximation)
    # a perspective corrected output image for each input image
    # i.e. the output should show a "top-down" view of stamp without major distortions.
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # crop the image
    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]
    result_dir = "calibration_undist"
    outfile = os.path.join(result_dir, os.path.basename(fname))
    cv2.imwrite(outfile, dst)
    # cv2.imwrite('calibration/test_undist.jpg', dst)

tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    tot_error += cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
print("total error: ", tot_error / len(objpoints))

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {"mtx": mtx, "dist": dist}
pickle.dump(dist_pickle, open("calibration/dist_pickle.p", "wb"))
np.savez('calibration/calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()
