from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np
import os
import cv2

def run_model(distance_training, training_results, distance_tests, test_results):
    '''
    This model runs using the scikit Random Forest classifier for image classification
    '''
    model = RandomForestClassifier(n_estimators=100)
    model.fit(distance_training, training_results)      # preparing the model with training information
    predictions = model.predict(distance_tests)        # compare prediction to expected value
    cm = confusion_matrix(test_results, predictions)    # confusion matrix to give us the amount of 
                                                        # true negative, palse positive, false negative and true positive predictions
    true_neg, false_pos, false_neg, true_pos = cm.ravel()
    frr = false_neg / (false_neg + true_pos)  # calculate False Reject Rate
    far = false_pos / (false_pos + true_neg)  # calculate False Accept Rate
    return frr, far


def load_data(directory):
    itemPairs = set()

    for item in os.listdir(directory):    #loops over all files in the directory
        if item.endswith('.png'):         #takes only .png files
            initial = item[0]             #keep track of first letter (f/s)
            rInitial = 's' if initial == 'f' else 'f'    #reverses Initial letter in the file name (f <-> s)
            matching_item = rInitial + item[1:]          #find name of the matching fingerprint
            if matching_item in os.listdir(directory):         #if the matching item exists, pair both of them
                pair = tuple(sorted((item, matching_item)))
                itemPairs.add(pair)
    
    itemPairs = list(itemPairs)
    return itemPairs

def find_minutia(path):     
    '''
    To find minutia, we need to find contours first, then use them to make a convex hull
    '''
	  
    """
    the first part reads images, makes them grayscale, blurs them then uses the cv2 method to creade contours
    """
    grayscale_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)    #reads the image in grayscale
    ret, binary_image = cv2.threshold(grayscale_image, 128, 255, cv2.THRESH_BINARY)    #sets a threshold of 128, any pixel brighter
                                                                                       #than that is set to white, anything less is
                                                                                       #set to black
    binary_image = cv2.medianBlur(binary_image, 3)                                     #blurs the image to get rid of granular abnormalities
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    #creates contours for minutia in the fingerprint
    minutia = []

    """
    creates a convex hull that goes around contours
    """
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)                             #sets epsilon to be around a fifth of the perimeter
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(approx_contour, returnPoints=True)
        for point in hull:
            if len(minutia) < 50:
                minutia.append(tuple(point[0]))

    """
    randomly pick points on the hull and generate offests of the hull
    this randomness helps to better learn the shape of the contours in the minutia
    """
    while len(minutia) < 50:
        index = np.random.randint(0, len(minutia))
        point = minutia[index]
        offset = np.random.uniform(-5, 5, size=2)
        extra_point = tuple(point[0] + offset)
        minutia.append(extra_point)

    return minutia

def get_distances(m1, m2):
    """
    calculates distance between the minutia
    """
    dist = []
    for p1 in m1:
        for p2 in m2:
            distance = np.linalg.norm(np.array(p1) - np.array(p2))
            dist.append(distance)

    return dist


def main():

    training_location = '/home/student/lab4/lab4/train'
    testing_location = '/home/student/lab4/lab4/test'
    print('Training Phase...')
    training_images = load_data(training_location)

    # setup lists for confusion matrix
    distance_training, training_results, distance_tests, test_results = [], [], [], []


    train_progress = tqdm(desc='Training', total=1500)
    for pair in training_images:
      train_progress.update(1)
      
      ''' get a pair of minutia for training '''
      minutia1 = find_minutia(os.path.join(training_location, pair[0]))
      minutia2 = find_minutia(os.path.join(training_location, pair[1]))
      
      ''' calculate the distances between the minutia '''
      distances = get_distances(minutia1, minutia2)
      distance_training.append(distances)
      avg_dist = sum(distances) / len(distances)
      if avg_dist < 210:          # if average distance less than 210 then it is considered success
          training_results.append(1)
      else:
          training_results.append(0)
      
    train_progress.close()
    
    
    print('Testing Phase...')
    testing_images = load_data(testing_location)
    
    
    test_progress = tqdm(desc="Testing...", total=500)
    for pair in testing_images:
        test_progress.update(1)
        
        ''' get a pair of minutia for testing '''
        minutia1 = find_minutia(os.path.join(testing_location, pair[0]))
        minutia2 = find_minutia(os.path.join(testing_location, pair[1]))
        
        ''' calculate the distances between the minutia '''
        distances = get_distances(minutia1, minutia2)
        distance_tests.append(distances)
        avg_dist = sum(distances) / len(distances)
        if avg_dist < 210:      # if average distance less than 210 then it is considered success
            test_results.append(1)
        else:
            test_results.append(0)
    test_progress.close()

    print('Modeling...')
    '''
    This is where we finalize the lists for the confusion matrix
    '''
    distance_training = np.array(distance_training).reshape(len(distance_training), -1)
    distance_tests = np.array(distance_tests).reshape(len(distance_tests), -1)

    training_results = np.array(training_results)
    test_results = np.array(test_results)

    
    frr_values = []
    far_values = []

    for _ in range(10):
        '''
        testing the model by running it multiple times and keeping track 
        of all False Rejection and False Acceptance Rates from each run 
        '''
        frr, far = run_model(distance_training, training_results, distance_tests, test_results)
        frr_values.append(frr)
        far_values.append(far)

    frr_max, frr_min, frr_avg = max(frr_values), min(frr_values), sum(frr_values) / len(frr_values)
    far_max, far_min, far_avg = max(far_values), min(far_values), sum(far_values) / len(far_values)

    eer = (far_avg + frr_avg)

    print(f"Max FRR: {frr_max}, Min FRR: {frr_min}, Avg FRR: {frr_avg}")
    print(f"Max FAR: {far_max}, Min FAR: {far_min}, Avg FAR: {far_avg}")
    print(f"EER: {eer}")

    return distance_training, distance_tests, training_results, test_results, frr_max, frr_min, far_max, far_min

if __name__ == "__main__":
    main()
