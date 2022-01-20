import pandas as pd
import numpy as np
import skimage
import cv2
from skimage.feature import greycomatrix, greycoprops
import os

patch_size = 10

locations = []

for i in range(28, 224, 28):
	for j in range(28, 224, 28):
		locations.append([i,j])

def calculate_features(resized_image):
	req_patches = []
	for loc in locations:
		req_patches.append(resized_image[loc[0]:loc[0] + patch_size, loc[1]:loc[1] + patch_size])

	contrasts = []
	correlations = []
	dissimilarity = []

	for patch in req_patches:
		glcm = greycomatrix(patch, distances= [2], angles=[0, np.pi/2, np.pi, 3*np.pi/2])
		correlation = greycoprops(glcm, 'correlation')
		correlations.append(correlation)
		contrasts.append(greycoprops(glcm, 'contrast'))
		dissimilarity.append(greycoprops(glcm, 'dissimilarity'))

	contrasts = np.asarray(contrasts).reshape(-1,)
	correlations = np.asarray(correlations).reshape(-1)
	dissimilarity = np.asarray(dissimilarity).reshape(-1)
	all_glcm_features = np.concatenate([contrasts, correlations, dissimilarity], axis=0)
	return all_glcm_features

if __name__ == '__main__':
	base_path = "Testing/"
	data = pd.read_csv("glcm_labelled_test.csv")
	filenames = data['filename']
	label = data['label']

	i = 0
	for filename in filenames:
		image_name = filename.split(".")[0]
		print(image_name)
		image = cv2.imread(base_path + filename)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		resized_image = cv2.resize(image, (224, 224))
		
		all_glcm_features = calculate_features(resized_image)
		print(all_glcm_features.shape)
		np.save(os.path.join("testing", image_name + ".npy"), all_glcm_features)
		i += 1 
		print(i)


