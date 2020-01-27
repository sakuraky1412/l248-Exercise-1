"""
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
"""
import numpy as np
import cv2
import glob

import matplotlib.pyplot as plt
import pickle

"""
Write code that makes use of OpenCV to load each image file in turn and detect the outer corners of the stamp. 

Then use the co-ordinates of those corners to produce (to a good approximation) a perspective corrected output image for each input image, 
i.e. the output should show a "top-down" view of stamp without major distortions. 
"""
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define the chess board rows and columns
rows = 8
cols = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('calibration_wide/GOPR003*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (rows, cols), corners2, ret)
        write_name = 'calibration_wide_results/corners_found' + str(idx) + '.jpg'
        cv2.imwrite(write_name, img)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Do camera calibration given object points and image points
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# img = cv2.imread('left12.jpg')
# Test undistortion on an image
img = cv2.imread('calibration_wide/test_image.jpg')
img_size = (img.shape[1], img.shape[0])
# h,  w = img.shape[:2]
# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,img_size,1,img_size)
# undistort
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
dst = cv2.undistort(img, mtx, dist, None, mtx)

# crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png',dst)
cv2.imwrite('calibration_wide/test_undist.jpg', dst)

tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    tot_error += cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)

print("total error: ", tot_error / len(objpoints))

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {"mtx": mtx, "dist": dist}
pickle.dump(dist_pickle, open("calibration_wide/wide_dist_pickle.p", "wb"))
# np.savez('../data/calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()
