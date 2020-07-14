import numpy as np
import cv2
import matplotlib.pyplot as plt
# matplotlib qt

img = cv2.imread("friends.jpg")
img_copy= np.copy(img)
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
#plt.imshow(img_copy)
gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
#plt.imshow(gray)

#Locate the key points

#set parameter for max key points and the pyramid decimation ratio
orb = cv2.ORB_create(200, 2.0)

#find key points in gray scale img and compute their orb descriptor
#None para is indicating we aren't using mask
keypoints, descriptor = orb.detectAndCompute(gray, None)

img_without_size = np.copy(img_copy)
img_with_size = np.copy(img_copy)


#draw key pts without size or orientation
cv2.drawKeypoints(img_copy, keypoints, img_without_size , color = (0, 255, 0))
plt.imshow(img_without_size)

#draw key pts with size or orientation
cv2.drawKeypoints(img_copy, keypoints, img_with_size , flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color = (2, 255, 0))
plt.imshow(img_with_size)
