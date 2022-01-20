# Brain-Tumor-Classification-Using-Machine-Learning

The purpose of the project is to classify brain tumors into 4 classes - Glioma, Meningioma, Pituitary and No Tumor. The dataset is taken from Kaggle -- https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset.

Generated data and weigths can be found here -- https://cmu.box.com/s/6diszv2wk71i3a9li7zocwmkehq9elby.

3 different feature engineering methods are compared, namely, Gray Level Co-Occurance Matrix (GLCM), Histogram Oriented Features (HOG) and Principal Component Analysis (PCA). We test these features on Random Forest Classifier, SVM and a Deep Neural Network. Highest test accuracy is achieved with HOG + PCA features trained on SVM equal to around 96%. We also compare the number of False Positives and False Negatives produced. HOG + PCA produced the lowest number of False Positives and False Negatives with an average Recall score = 0.96.

We benchmark these results using ResNet50 and DenseNet169 architectures. A test accuracy of 99.2% is achieved using DenseNet169. However, our generated features perform better than ResNet50, prodcuing a max test accuracy equal to 96% wheras ResNet50 produces a max test accuracy equal to only 93%.

