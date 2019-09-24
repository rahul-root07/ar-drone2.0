import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import ps_drone
import time
import math
# File storage in OpenCV

cv_file = cv2.FileStorage("data/calib.yaml", cv2.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode("camera_matrix").mat()
dist_matrix = cv_file.getNode("dist_coeff").mat()
cv_file.release()

mtx = camera_matrix
dist = dist_matrix




drone = ps_drone.Drone()                                     # Start using drone
drone.startup() 
drone.reset()                                                # Sets drone's status to good (LEDs turn green when red)
while (drone.getBattery()[0] == -1): 
	time.sleep(0.1)      
	   # Waits until drone has done its reset
print("Battery:"+str(drone.getBattery()[0])+"%"+str(drone.getBattery()[1]))	# Gives a battery-status
drone.useDemoMode(True)                                      # Just give me 15 basic dataset per second (is default anyway)

##### Mainprogram begin #####
drone.setConfigAllID()                                       # Go to multiconfiguration-mode
drone.hdVideo()                                              # Choose lower resolution (hdVideo() for...well, guess it)
drone.frontCam()                                             # Choose front view
CDC = drone.ConfigDataCount
while (CDC == drone.ConfigDataCount):       
	time.sleep(0.0001) # Wait until it is done (after resync is done)
drone.startVideo()                                           # Start video-function
drone.showVideo()
#drone.trim()
drone.getSelfRotation(3)

drone.takeoff()
time.sleep(3)
drone.stop()
time.sleep(2) 

while(True):
    
    	frame = drone.VideoImage 
    	if drone.VideoImageCount > 100:
	    	



		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
		parameters =  aruco.DetectorParameters_create()		
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict,parameters=parameters)
			                                    
		# SUB PIXEL DETECTION
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
		for corner in corners:
		    cv2.cornerSubPix(gray, corner, winSize = (11,11), zeroZone = (-1,-1), criteria = criteria)

		frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
		size_of_marker =  0.13 # side lenght of the marker in meter
		rvecs,tvecs,obj = aruco.estimatePoseSingleMarkers(corners, size_of_marker , mtx, dist)
		#rvecs,tvecs = aruco.estimatePoseSingleMarkers( mtx, dist)
		length_of_axis = 0.01
		imaxis = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
		
		key=drone.getKey()
		if key ==" ":
			drone.land()						
		try:
				tvecs = tvecs*10
				xyz=tvecs[0]
				data = pd.DataFrame(data = tvecs.reshape(len(tvecs),3), columns = ["tx", "ty", "tz"],
                		index = ids.flatten())
				data.index.name = "marker"
				data.sort_index(inplace= True)
	
				x=xyz[0][0]
				y=xyz[0][1]
				z=xyz[0][2]
				print("the value of z ",z)
				if (z < 15 and z > 13) :
					drone.stop()					
					print("stay in it's positions")

				elif (z > 15) :	
					drone.moveForward(0.1)
					time.sleep(1)
					print("moving forward in")
					drone.stop()


				else:
					drone.moveBackward(0.1)
					time.sleep(1)
   					print("moving backward")	
					drone.stop()
		
				 
				
		except:
			drone.stop()
			continue
			print("landing")
			
