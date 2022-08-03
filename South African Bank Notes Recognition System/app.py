"""
Name: Sashen Moodley
Student Number: 219006946
"""

import cv2
import warnings
import glob
import time
import numpy as np
import mahotas as mt
from matplotlib import pyplot as plt
from matplotlib import rcParams
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from math import copysign, log10

warnings.filterwarnings('ignore')
rcParams.update({'figure.autolayout': True})

"""
Name: Sashen Moodley
Student Number: 219006946
"""

"""
============================================= HOW TO RUN ======================================================= 
This application is built in a modular fashion, that is every technique is represented as a function

This application is broken into segments corresponding to each major activity:
    1 - Pre-processing
    2 - Segmentation
    3 - Feature extraction
    4 - Classification

You can simply run this file with its current configuration

The current configuration is:
    - Pre-processing: 
        (1) Scaling 
        (2) Grayscale conversion 
        (3) Histogram equalization 
        (4) Global binary thresholding
        (5) Median Filtering
    - Segmentation: Region-based segmentation
    - Feature extraction: 5 Haralick GLCM features
    - Classification: Both the SVM and the KNN

You are at liberty to specify your own runtime configuration as follows:
     - To configure the pre-processing techniques, refer to the 'pre_process_image()' function
     - To configure which segmentation technique to use, refer to the 'get_training_data()' and 
       'get_testing_data()' functions
     - To configure which feature extraction technique to use refer to the 'get_training_data()' 
       and 'get_testing_data()' functions

    Both classifiers are utilized in the main application, if you wish to utilize only one during 
    runtime the best way to achieve this would be to simply comment-out the classifier in the main application

These instructions are additionally provided on the GitHub Repository page
================================================================================================================
"""


# ============================ HELPER FUNCTIONS ===========================================
def get_image_info(directory_path):
    """
    This function will take the directory path and provide the denomination, orientation, and  age of the note
    Therefore it is imperative that the notes are put into their correct directory to get the correct label
    :param directory_path: The directory path to the image (make sure the image is in one of the pre-created folders)
    :return: Denomination (R10, R20, R50, R100, R200),
             Orientation (Front/Back), and
             Age (Old, New)
    """
    denomination = None
    age = None
    orientation = None

    # Getting denomination
    if "R10 Notes" in directory_path:
        denomination = 10
    elif "R20 Notes" in directory_path:
        denomination = 20
    elif "R50 Notes" in directory_path:
        denomination = 50
    elif "R100 Notes" in directory_path:
        denomination = 100
    elif "R200 Notes" in directory_path:
        denomination = 200

    if "New" in directory_path:
        age = 'New'
    elif "Old" in directory_path:
        age = 'Old'

    if "Front" in directory_path:
        orientation = 'Front'
    elif "Back" in directory_path:
        orientation = 'Back'

    return denomination, age, orientation


def max_probability(coocurance_matrix):
    """
    Gets the maximum probability from the normalized GLCM
    :param coocurance_matrix: Normalized GLCM
    :return: Maximum probability within the GLCM
    """
    max_prob = 0.0
    for a in coocurance_matrix:
        for b in a:
            for c in b:
                if c > max_prob:
                    max_prob = c
    return max_prob


def get_flat_accuracy(y_pred, y_true):
    total = len(y_true)
    correct = sum(prediction == y_true[i] for i, prediction in enumerate(y_pred))

    return (correct / total) * 100


# ==========================================================================================


# ============================ 1 - PREPROCESSING ===========================================
def scale_image(image):
    return cv2.resize(image, (400, 200), interpolation=cv2.INTER_AREA)


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def histogram_equalize(image):
    return cv2.equalizeHist(image)


def median_filter(image):
    return cv2.medianBlur(image, 3)


def average_smoothing(image):
    return cv2.blur(src=image, ksize=(3, 3))


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (7, 7), 0)


def binary_thresholding(image):
    return cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]


def adaptive_thresholding(image):
    return cv2.adaptiveThreshold(src=image,
                                 maxValue=255,
                                 adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                 thresholdType=cv2.THRESH_BINARY,
                                 blockSize=9,
                                 C=1)


def pre_process_image(image):
    """
    These are the pre-processing techniques that will be applied to the images when running the main application
    You have complete control over which functions you wish to add in this section
    :param image: Image to be enhanced
    :return: Resultant image from pre-processing
    """
    # THESE SHOULD NOT BE CHANGED
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = scale_image(processed_image)
    processed_image = to_grayscale(processed_image)
    # INSERT CUSTOM PRE-PROCESSING FUNCTIONS BELOW
    processed_image = histogram_equalize(processed_image)
    processed_image = binary_thresholding(processed_image)
    processed_image = median_filter(processed_image)

    return processed_image


# =========================================================================================


# ============================ 2 - SEGMENTATION ===========================================
def canny_edge_detection(image):
    # Canny uses a default sobel kernel size of 3x3
    edges = cv2.Canny(image, threshold1=100, threshold2=200,  # 100 # 200
                      L2gradient=False)  # if we change from false to true we get different results
    image = cv2.bitwise_and(image, image, edges)
    return image


def sobel_edge_detection(image):
    """Add some more input parameters to specify which one we want"""
    # sobel detection on the x-axis
    sobelx = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    # sobel detection on the y-axis
    sobely = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    # combined x and y sobel edge detection
    sobelxy = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)

    return sobelx, sobely, sobelxy


def region_segmentation(image, image_info):
    """
    :param image: Input image to be segmented into regions
    :param image_info: List of info in form [denomination, age, orientation]
    :return: image segments
    """
    # Image ROIs
    front_new_note = {
        'Mandela': [32, 164, 198, 303],
        'Left Value': [127, 179, 64, 139],
        'Discrim Right Value': [127, 176, 309, 377],
        'Word Value': [14, 124, 359, 379],
        'Right Pattern': [15, 124, 328, 359],
        'Left Pattern': [33, 73, 111, 160],
        'Bottom Pattern': [179, 199, 96, 369],
        'Animal': [143, 176, 21, 78],
        'COA Symbol': [7, 55, 9, 53],
        'Reserve Bank Text': [14, 30, 44, 283],
        'Left Lines': [123, 174, 0, 20],
        'Right Lines': [127, 172, 379, 499]
    }

    back_new_note = {
        'Animal': [37, 171, 95, 204],
        'Left Value': [9, 35, 35, 71],
        'Discrim Right Value': [121, 174, 205, 276],
        'Word Value': [11, 102, 376, 399],
        'Animal Half': [140, 175, 330, 374],
        'Animal Pattern': [123, 184, 282, 329],
        'Top Text': [13, 31, 65, 259],
        'Bottom Text': [169, 186, 32, 258],
        'Left Pattern': [46, 173, 35, 92],
        'Right Pattern': [41, 86, 209, 287],
        'Top Pattern': [0, 13, 35, 204],
        'Bottom Pattern': [187, 199, 35, 204]
    }

    front_old_note = {
        'Animal': [19, 138, 235, 335],
        'Small Animal': [35, 111, 108, 190],
        'Discrim Left Value': [115, 173, 105, 190],
        'Left Text Value': [3, 114, 0, 24],
        'Right Text Value': [3, 117, 374, 399],
        'Circle Left': [115, 171, 7, 69],
        'Circle Right': [117, 170, 329, 390],
        'Dots': [175, 193, 2, 59]
    }

    back_old_note = {
        'Left Text Value': [0, 120, 0, 21],
        'Right Text Value': [0, 120, 363, 399],
        'Left Value': [166, 199, 0, 22],
        'Right Value': [167, 199, 371, 399],
        'Discrim Middle Value': [111, 182, 186, 299],
        'Large Pattern': [29, 114, 52, 287],
        'Small Pattern': [114, 182, 27, 185],
        'Circle': [116, 176, 331, 392]
    }

    image_region_subsets = []
    discriminant_region_subets = []
    chosen_dict = None
    if image_info[1] == 'New' and image_info[2] == 'Front':
        chosen_dict = front_new_note
    elif image_info[1] == 'New' and image_info[2] == 'Back':
        chosen_dict = back_new_note
    elif image_info[1] == 'Old' and image_info[2] == 'Front':
        chosen_dict = front_old_note
    elif image_info[1] == 'Old' and image_info[2] == 'Back':
        chosen_dict = back_old_note

    for dict_element in chosen_dict.items():
        region_mask = dict_element[1]
        image_region_subsets.append(image[region_mask[0]:region_mask[1], region_mask[2]:region_mask[3]])
        if 'Discrim' in dict_element[0]:
            discriminant_region_subets.append(image[region_mask[0]:region_mask[1], region_mask[2]:region_mask[3]])

    return image_region_subsets, discriminant_region_subets


# ==========================================================================================


# ============================ 3 - FEATURE EXTRACTION ======================================
def haralick_features_5(image):
    matrix_coocurance = graycomatrix(image, distances=[1], angles=[0],
                                     levels=256, normed=True, symmetric=False)

    features = []
    features.append(graycoprops(matrix_coocurance, 'energy')[0][0])
    features.append(graycoprops(matrix_coocurance, 'contrast')[0][0])
    features.append(graycoprops(matrix_coocurance, 'dissimilarity'))
    features.append(graycoprops(matrix_coocurance, 'homogeneity')[0][0])
    features.append(graycoprops(matrix_coocurance, 'correlation')[0][0])

    return features


def haralick_features_13(image):
    textures = mt.features.haralick(image)
    return textures.mean(axis=0)


def hu_7(image):
    moments = cv2.moments(image)
    huMoments = cv2.HuMoments(moments)

    return [-1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])) for i in range(7)]


# ==========================================================================================


# ============================ 4 - CLASSIFICATION ==========================================
def get_training_data():
    """
    This function will generate the training data for the classifiers using the specified pre-processing, segmentation,
    and feature extraction technique which you can specify in the function body
    :return: Training data in the form of a feature vector along with their respective label
    """
    x_train = []
    y_train = []

    training_directories = ['Bank Notes Dataset/Training Notes/R10 Notes/New/Back',
                            'Bank Notes Dataset/Training Notes/R10 Notes/New/Front',
                            'Bank Notes Dataset/Training Notes/R10 Notes/Old/Back',
                            'Bank Notes Dataset/Training Notes/R10 Notes/Old/Front',
                            'Bank Notes Dataset/Training Notes/R20 Notes/New/Back',
                            'Bank Notes Dataset/Training Notes/R20 Notes/New/Front',
                            'Bank Notes Dataset/Training Notes/R20 Notes/Old/Back',
                            'Bank Notes Dataset/Training Notes/R20 Notes/Old/Front',
                            'Bank Notes Dataset/Training Notes/R50 Notes/New/Back',
                            'Bank Notes Dataset/Training Notes/R50 Notes/New/Front',
                            'Bank Notes Dataset/Training Notes/R50 Notes/Old/Back',
                            'Bank Notes Dataset/Training Notes/R50 Notes/Old/Front',
                            'Bank Notes Dataset/Training Notes/R100 Notes/New/Back',
                            'Bank Notes Dataset/Training Notes/R100 Notes/New/Front',
                            'Bank Notes Dataset/Training Notes/R100 Notes/Old/Back',
                            'Bank Notes Dataset/Training Notes/R100 Notes/Old/Front',
                            'Bank Notes Dataset/Training Notes/R200 Notes/New/Back',
                            'Bank Notes Dataset/Training Notes/R200 Notes/New/Front',
                            'Bank Notes Dataset/Training Notes/R200 Notes/Old/Back',
                            'Bank Notes Dataset/Training Notes/R200 Notes/Old/Front'
                            ]

    for directory in training_directories:
        glob_path = glob.iglob(f'{directory}/*')
        for note_file in glob_path:
            if note_file is not None:
                image_info = get_image_info(note_file)
                image = cv2.imread(note_file)

                # pre-process the image
                image = pre_process_image(image)  # Uses the configured pre-processing techniques

                # segmentation of the image
                sobel_segmentation = sobel_edge_detection(image)
                canny_segmentation = canny_edge_detection(image)
                discrim_regions = region_segmentation(canny_segmentation, image_info)[1]

                # feature extraction based on which segmentation technique you want to use
                hu_7_transformed = hu_7(discrim_regions[0])
                five_haralick_features = haralick_features_5(discrim_regions[0])
                thirteen_haralick_features = haralick_features_13(discrim_regions[0])

                x_train.append(five_haralick_features)  # Append which feature extraction method you one you want
                y_train.append(image_info[0])

    return x_train, y_train


def get_testing_data():
    """
    This function will generate the testing data for the classifiers using the specified pre-processing, segmentation,
    and feature extraction technique which you can specify in the function body
    :return: Testing data in the form of a feature vector along with their respective label
    """
    x_test = []
    y_test = []

    testing_directories = ['Bank Notes Dataset/Testing Notes/R10 Notes/New/Back',
                           'Bank Notes Dataset/Testing Notes/R10 Notes/New/Front',
                           'Bank Notes Dataset/Testing Notes/R10 Notes/Old/Back',
                           'Bank Notes Dataset/Testing Notes/R10 Notes/Old/Front',
                           'Bank Notes Dataset/Testing Notes/R20 Notes/New/Back',
                           'Bank Notes Dataset/Testing Notes/R20 Notes/New/Front',
                           'Bank Notes Dataset/Testing Notes/R20 Notes/Old/Back',
                           'Bank Notes Dataset/Testing Notes/R20 Notes/Old/Front',
                           'Bank Notes Dataset/Testing Notes/R50 Notes/New/Back',
                           'Bank Notes Dataset/Testing Notes/R50 Notes/New/Front',
                           'Bank Notes Dataset/Testing Notes/R50 Notes/Old/Back',
                           'Bank Notes Dataset/Testing Notes/R50 Notes/Old/Front',
                           'Bank Notes Dataset/Testing Notes/R100 Notes/New/Back',
                           'Bank Notes Dataset/Testing Notes/R100 Notes/New/Front',
                           'Bank Notes Dataset/Testing Notes/R100 Notes/Old/Back',
                           'Bank Notes Dataset/Testing Notes/R100 Notes/Old/Front',
                           'Bank Notes Dataset/Testing Notes/R200 Notes/New/Back',
                           'Bank Notes Dataset/Testing Notes/R200 Notes/New/Front',
                           'Bank Notes Dataset/Testing Notes/R200 Notes/Old/Back',
                           'Bank Notes Dataset/Testing Notes/R200 Notes/Old/Front'
                           ]

    for directory in testing_directories:
        glob_path = glob.iglob(f'{directory}/*')
        for note_file in glob_path:
            if note_file is not None:
                image_info = get_image_info(note_file)
                image = cv2.imread(note_file)

                # pre-process the image
                image = pre_process_image(image)

                # segmentation of the image
                sobel_segmentation = sobel_edge_detection(image)
                canny_segmentation = canny_edge_detection(image)
                discrim_regions = region_segmentation(canny_segmentation, image_info)[1]

                # feature extraction based on which segmentation technique you want to use
                hu_7_transformed = hu_7(discrim_regions[0])
                five_haralick_features = haralick_features_5(discrim_regions[0])
                thirteen_haralick_features = haralick_features_13(discrim_regions[0])

                x_test.append(five_haralick_features)  # Append which feature extraction method you one you want
                y_test.append(image_info[0])

    return x_test, y_test


def k_nearest_classifier(no_of_neighbours, train_x, train_y, test_x):
    knn_model = KNeighborsClassifier(no_of_neighbours)
    print("Fitting knn training data. . . ")
    start_time = time.time()
    knn_model.fit(train_x, train_y)
    print(f"KNN took {time.time() - start_time}s to fit data")

    print("Making predictions. . .")
    start_time = time.time()
    predictions = [knn_model.predict([testing_image])[0] for testing_image in test_x]
    print(f"KNN took {time.time() - start_time}s to make predictions")

    return predictions


def svm_classifier(x_train, y_train, test_x):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    # svm.setKernel(cv2.ml.SVM)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    print("Training SVM. . . ")
    start_time = time.time()
    svm.train(np.array(x_train, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(y_train))
    print(f"SVM took {time.time() - start_time}s to fit training data")

    print("Making Predictions. . .")
    predictions = []
    start_time = time.time()
    for testing_vector in test_x:
        testing_vector_array = [testing_vector]
        numpy_vector = np.array(testing_vector_array, dtype=np.float32)
        predictions.append(svm.predict(numpy_vector)[1].tolist())
    print(f"SVM took {time.time() - start_time}s to make predictions")
    predictions = [int(pred[0][0]) for pred in predictions]

    return predictions


# ==========================================================================================
# ================================ MAIN APPLICATION ========================================
# ==========================================================================================

# Getting the training and testing vectors (X) and their respective labels (Y)
x_train, y_train = get_training_data()
print("*Training samples successfully loaded*")
x_test, y_test = get_testing_data()
print("*Testing samples successfully loaded*")

# =================================== SVM ==================================================
# Setting up and making predictions with the SVM classifier
svm_classifier_predictions = svm_classifier(x_train, y_train, x_test)
print(f"Correct labels: {y_test}")
print(f"Predictions: {svm_classifier_predictions}")

# Getting the confusion matrix and classification report for the SVM classifier
svm_confusion_matrix = confusion_matrix(y_true=y_test, y_pred=svm_classifier_predictions)
disp = ConfusionMatrixDisplay(svm_confusion_matrix)
disp.plot()
plt.title("SVM Confusion Matrix")
plt.show()
print("---------- SVM CLASSIFICATION REPORT --------------- ")
print(classification_report(y_true=y_test, y_pred=svm_classifier_predictions))
print(f"Flat SVM accuracy: {get_flat_accuracy(y_pred=svm_classifier_predictions, y_true=y_test)}%\n")

# ===================================== KNN =================================================
# Setting up and making predictions with the KNN classifier
knn_classification_predictions = k_nearest_classifier(1, x_train, y_train, x_test)
print(f"Correct labels: {y_test}")
print(f"Predictions: {knn_classification_predictions}")

# Getting the confusion matrix and classification report for the KNN classifier
knn_confusion_matrix = confusion_matrix(y_true=y_test, y_pred=knn_classification_predictions)
disp = ConfusionMatrixDisplay(knn_confusion_matrix)
disp.plot()
plt.title("K-NN Confusion Matrix")
plt.show()
print("------------- KNN CLASSIFICATION REPORT -------------- ")
print(classification_report(y_true=y_test, y_pred=knn_classification_predictions))
print(f"Flat KNN accuracy: {get_flat_accuracy(y_pred=knn_classification_predictions, y_true=y_test)}%")

# ==========================================================================================
# ========================================= END=============================================
# ==========================================================================================
