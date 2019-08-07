import cv2
import numpy as np
from math import e, sqrt, pi
img = cv2.imread('task2.jpg',0)
image = cv2.imread('task2.jpg',1)

height,width, = img.shape

#Function to plot keypoints on Grayscale original image
def plotKeypoints(orig,key1,key2):
	height,width = orig.shape
	for i in range(height):
		for j in range(width):
			if(key1[i][j]==255 or key2[i][j]==255):
				orig[i][j]=255
	return orig
#Function to plot keypoints on Coloured original image	
def plotKeypoints_color(orig,key1,key2):
	height,width,_ = orig.shape
	for i in range(height):
		for j in range(width):
			if(key1[i][j]==255 or key2[i][j]==255):
				orig[i,j]=[255,255,255]
	return orig
#Function to calculate the coordinates of at-most 'n' leftmost keypoints.
def cal(c,num):
	n = 0
	left = []
	height, width = c.shape
	
	
	for k in range(width):
		for l in range(height):
			row=[]
			if(c[l][k]==255 and n<num):
				n = n + 1
				row.append([k,l])
				left.append(row)
	return left

#Function to generate Gaussian kernel of any size.
def kernel(size,s):
	j=0
	i=0
	c = 0
	d = 0
	g_kernel = [[ 0 for i in range(size)] for j in range(size)]
	y = j+ int(size/2) + 1
	for i in range(size):
		x = -int(size/2)
		y = y - 1
		for j in range(size):
			gauss = (1/(2*pi*s*s))*e**(-0.5*((float(x*x) + float(y*y))/(s*s)))
			g_kernel[i][j] = gauss
			x = x + 1
	for i in range(size):
		for j in range(size):
			c = c + g_kernel[i][j]
			
	co = (float)(1/c)
	
	for i in range(size):
		for j in range(size):
			g_kernel[i][j] = float(co*g_kernel[i][j])
	
	for i in range(size):
		for j in range(size):
			d = d + g_kernel[i][j]
					
	k_arr = np.asarray(g_kernel)
	return k_arr

#Function to apply Gaussian Blur
def blur(img, s,height,width):
	a = [[ 0 for x in range(width+6)] for w in range(height+6)]
	b = [[ 0 for x in range(width)] for w in range(height)]
	
	for i in range(height):
	    for j in range(width):
	        a[i+3][j+3] = img[i][j]

      	  
	a = np.asarray(a)
	ker_arr = kernel(7,s)
	for i in range(height):
	    for j in range(width):
	        sum1=0
	        for k in range(7):
	            for l in range(7):
	                sum1 += ker_arr[k][l]*a[i+k][j+l]
	        b[i][j] = sum1     
	fin = np.asarray(b)
	return fin

#Function to resize image to generate scale space. 
def resize(img,height,width):
	r = [[ 0 for x in range(int((width/2 )))] for w in range(int(height/2))]
	k,l=0,0
	
	for i in range(0,height-1,2):
		for j in range(0,width-1,2):
			
			r[k][int(j/2)] = img[i][j]
			#print(i,int(j/2))
		k = k + 1 
	r = np.asarray(r)	 
	return r

#Resizing coloured image
'''
height, width,_ = image.shape

#For Octave 2
image = resize(image, height,width)
#cv2.imwrite('Second-octave.jpg',image)

#For Ocatve 3
height, width,_ = image.shape
image = resize(image, height,width)
#cv2.imwrite('Third-octave.jpg',image)

#For Octave 4
height, width,_ = image.shape
image = resize(image, height,width)
#cv2.imwrite('Fourth-octave.jpg',image)
'''
#Resizing Grayscale image
'''
height, width = img.shape
#For Octave 2
img = resize(img, height,width)
#cv2.imwrite('Second-octave-GS.jpg',image)

#For Octave 3
height, width = img.shape
img = resize(img, height,width)
#cv2.imwrite('Third-octave-GS.jpg',image)

#For Octave 4
height, width = img.shape
img = resize(img, height,width)
#cv2.imwrite('Fourth-octave-GS.jpg',image)
'''
#Five different scales for gaussian kernel with bandwidth parameters.

#Octave1
k1 = blur(img,0.707,height,width)
k2 = blur(img,1,height,width)
k3 = blur(img,1.414,height,width)
k4 = blur(img,2,height,width)
k5 = blur(img,2.828,height,width)

'''
#Octave 2
height, width = img.shape
img = resize(img, height,width)
height, width = img.shape
k1 = blur(img,1.414,height,width)
k2 = blur(img,2,height,width)
k3 = blur(img,2.828,height,width)
k4 = blur(img,4,height,width)
k5 = blur(img,5.656,height,width)


#Octave 3
height, width = img.shape
img = resize(img, height,width)
height, width = img.shape
img = resize(img, height,width)
height, width = img.shape
k1 = blur(img,2.828,height,width)
k2 = blur(img,4,height,width)
k3 = blur(img,5.656,height,width)
k4 = blur(img,8,height,width)
k5 = blur(img,11.312,height,width)

#Octave 4
height, width = img.shape
img = resize(img, height,width)
height, width = img.shape
img = resize(img, height,width)
height, width = img.shape
img = resize(img, height,width)
height, width = img.shape
k1 = blur(img,5.656,height,width)
k2 = blur(img,8,height,width)
k3 = blur(img,11.312,height,width)
k4 = blur(img,16,height,width)
k5 = blur(img,22.624,height,width)
'''
#Difference of Gaussian Images
d1 = k1 - k2
d2 = k2 - k3
d3 = k3 - k4
d4 = k4 - k5 

#Show and save DoG images
'''
cv2.imshow('DoG_1',d1)
cv2.waitKey(0)
d1p = d1 * 255 
cv2.imwrite('DoG_3.1.jpg',d1p)

cv2.imshow('DoG_2',d1)
cv2.waitKey(0)
d1p = d2 * 255 
cv2.imwrite('DoG_3.2.jpg',d1p)

cv2.imshow('DoG_3',d1)
cv2.waitKey(0)
d1p = d3 * 255 
cv2.imwrite('DoG_3.3.jpg',d1p)

cv2.imshow('DoG_4',d1)
cv2.waitKey(0)
d1p = d4 * 255 
cv2.imwrite('DoG_3.4.jpg',d1p)
cv2.destroyAllWindows()
'''
#Function to calculate maxima and minima
def maxmin(dog1,dog2,dog3):
	count = 0
	height,width = dog1.shape
	dogp1 = [[ 0 for i in range(width+2)] for j in range(height+2)]
	dogp2 = [[ 0 for i in range(width+2)] for j in range(height+2)]
	dogp3 = [[ 0 for i in range(width+2)] for j in range(height+2)]
	alt = [[ 0 for i in range(width)] for j in range(height)]
	final = [[ 0 for x in range(width)] for w in range(height)]

	#Padding the DoG Images
	#print("Before Padding:",dogp1.shape)
	for i in range(height):
		for j in range(width):
			dogp1[i+1][j+1] = dog1[i][j]
			
	for i in range(height):
		for j in range(width):
			dogp2[i+1][j+1] = dog2[i][j]
			
	for i in range(height):
		for j in range(width):
			dogp3[i+1][j+1] = dog3[i][j]
	#print("After padding",dog1.shape)		
	for i in range(height):
		for j in range(width):
			#Finding Maxima
			#Same layer
			if(dogp2[i+1][j+1]>dogp2[i][j]):
				if(dogp2[i+1][j+1]>dogp2[i][j+1]):
					if(dogp2[i+1][j+1]>dogp2[i][j+2]):
						if(dogp2[i+1][j+1]>dogp2[i+1][j]):
							if(dogp2[i+1][j+1]>dogp2[i+1][j+2]):
								if(dogp2[i+1][j+1]>dogp2[i+2][j]):
									if(dogp2[i+1][j+1]>dogp2[i+2][j+1]):
										if(dogp2[i+1][j+1]>dogp2[i+2][j+2]):
											#Below layer
											if(dogp2[i+1][j+1]>dogp1[i][j]):
												if(dogp2[i+1][j+1]>dogp1[i][j+1]):
													if(dogp2[i+1][j+1]>dogp1[i][j+2]):
														if(dogp2[i+1][j+1]>dogp1[i+1][j]):
															if(dogp2[i+1][j+1]>dogp1[i+1][j+2]):
																if(dogp2[i+1][j+1]>dogp1[i+2][j]):
																	if(dogp2[i+1][j+1]>dogp1[i+2][j+1]):
																		if(dogp2[i+1][j+1]>dogp1[i+2][j+2]):
																			if(dogp2[i+1][j+1]>dogp1[i+1][j+1]):
																				#Upper Layer
																				if(dogp2[i+1][j+1]>dogp3[i][j]):
																					if(dogp2[i+1][j+1]>dogp3[i][j+1]):
																						if(dogp2[i+1][j+1]>dogp3[i][j+2]):
																							if(dogp2[i+1][j+1]>dogp3[i+1][j]):
																								if(dogp2[i+1][j+1]>dogp3[i+1][j+2]):
																									if(dogp2[i+1][j+1]>dogp3[i+2][j]):
																										if(dogp2[i+1][j+1]>dogp3[i+2][j+1]):
																											if(dogp2[i+1][j+1]>dogp3[i+2][j+2]):
																												if(dogp2[i+1][j+1]>dogp3[i+1][j+1]):
																													alt[i][j] = 255
																												
			#Finding Minima																									
			#Same layer						
			if(dogp2[i+1][j+1]<dogp2[i][j]):
				if(dogp2[i+1][j+1]<dogp2[i][j+1]):
					if(dogp2[i+1][j+1]<dogp2[i][j+2]):
						if(dogp2[i+1][j+1]<dogp2[i+1][j]):
							if(dogp2[i+1][j+1]<dogp2[i+1][j+2]):
								if(dogp2[i+1][j+1]<dogp2[i+2][j]):
									if(dogp2[i+1][j+1]<dogp2[i+2][j+1]):
										if(dogp2[i+1][j+1]<dogp2[i][j+2]):
											#Below layer
											if(dogp2[i+1][j+1]<dogp1[i][j]):
												if(dogp2[i+1][j+1]<dogp1[i][j+1]):
													if(dogp2[i+1][j+1]<dogp1[i][j+2]):
														if(dogp2[i+1][j+1]<dogp1[i+1][j]):
															if(dogp2[i+1][j+1]<dogp1[i+1][j+2]):
																if(dogp2[i+1][j+1]<dogp1[i+2][j]):
																	if(dogp2[i+1][j+1]<dogp1[i+2][j+1]):
																		if(dogp2[i+1][j+1]<dogp1[i][j+2]):
																			if(dogp2[i+1][j+1]<dogp1[i+1][j+1]):
																				#Upper Layer
																				if(dogp2[i+1][j+1]<dogp3[i][j]):
																					if(dogp2[i+1][j+1]<dogp3[i][j+1]):
																						if(dogp2[i+1][j+1]<dogp3[i][j+2]):
																							if(dogp2[i+1][j+1]<dogp3[i+1][j]):
																								if(dogp2[i+1][j+1]<dogp3[i+1][j+2]):
																									if(dogp2[i+1][j+1]<dogp3[i+2][j]):
																										if(dogp2[i+1][j+1]<dogp3[i+2][j+1]):
																											if(dogp2[i+1][j+1]<dogp3[i][j+2]):
																												if(dogp2[i+1][j+1]<dogp3[i+1][j+1]):
																													alt[i][j] = 255	
							
						
		
	alt = np.asarray(alt)			
	final = np.asarray(final)
	return alt				

#Keypoint Image 1 
keyp1 = maxmin(d1,d2,d3)	
m=0
for i in range(height):
	for j in range(width):
		if(keyp1[i][j]>m):
			m = keyp1[i][j]

print("Coordinates of atmost five-leftmost keypoints in Keypoint Image 1:")
list = cal(keyp1,5)	
print(list)	
if(m>0):
	keyp = keyp1/m
	cv2.imshow('image',keyp)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	keyp =keyp1* 255
	cv2.imwrite('Keypoint4.1.jpg',keyp)
else:
	print("No Keypoints in Image 1")

#Keypoint Image 2 
keyp2 = maxmin(d2,d3,d4)	
m=0
for i in range(height):
	for j in range(width):
		if(keyp2[i][j]>m):
			m = keyp2[i][j]

print("Coordinates of atmost five-leftmost keypoints in Keypoint Image 2:")
list = cal(keyp2,5)	
print(list)	
if(m>0):
	keyp = keyp2/m
	cv2.imshow('image',keyp)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	keyp = keyp2* 255
	cv2.imwrite('Keypoint4.2.jpg',keyp)
else:
	print("No Keypoint in Image 2")

#Keypoints on Grayscale Original Image.
final = plotKeypoints(img,keyp1,keyp2)
cv2.imshow('Final',final)
cv2.waitKey(0)
cv2.imwrite('Final_Fourth-Octave.jpg',final)

#Keypoints on Coloured original Image
final_color = plotKeypoints_color(image,keyp1,keyp2)
cv2.imshow('Final',final_color)
cv2.waitKey(0)
cv2.imwrite('Final_color_Fourth-Octave-.jpg',final_color)


cv2.destroyAllWindows()
