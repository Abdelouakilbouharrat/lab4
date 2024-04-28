import SuppVector
import RandForest
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
import HGBC

# initializing values
frrs=[]
fars=[]
Hybrid_pred = []
print()
def main ():
    print('RF METHOD:')
    '''
    use random forest method for image classification
    this runs it and gets the predictions made by that
    method
    '''
    rf_model = RandomForestClassifier(n_estimators=100)
    # get the required values from the RF method
    distance_training, distance_tests, training_results, test_results, frr_max, frr_min, far_max, far_min = RandForest.main()
    rf_model.fit(distance_training, training_results)
    rf_pred = rf_model.predict(distance_tests)    #extract RF predictions
    frrs.extend([frr_max, frr_min])           #keep track of minimum and maximum FRR and FAR
    fars.extend([far_max, far_min])

    print('SVC METHOD:')
    '''
    use Support Vector method for image classification
    this runs it and gets the predictions made by that
    method
    '''
    svc_model = SVC(kernel='linear')
    # get the required values from the SVC method
    distance_training, distance_tests, training_results, test_results, frr_max, frr_min, far_max, far_min = SuppVector.main()
    svc_model.fit(distance_training, training_results)
    svc_pred = svc_model.predict(distance_tests)   #extract SVC predictions
    frrs.extend([frr_max, frr_min])            #keep track of minimum and maximum FRR and FAR
    fars.extend([far_max, far_min])
    
    print('HistGradient METHOD')
    '''
    use Histogram-based Gadient Boosting method for image classification
    this runs it and gets the predictions made by that
    method
    '''
    Hist_model = HistGradientBoostingClassifier()
    # get the required values from the HGBC method
    distance_training, distance_tests, training_results, test_results, frr_max, frr_min, far_max, far_min = HGBC.main()
    Hist_model.fit(distance_training, training_results)
    H_pred = Hist_model.predict(distance_tests)     #extract HGBC predictions
    frrs.extend([frr_max, frr_min])            #keep track of minimum and maximum FRR and FAR
    fars.extend([far_max, far_min])

    '''
    The part responsilbe of comparing all predictions of the 3 methods,
    looking for consensus between the 3 and only allowing the ones with
    majority agreement to be predicted as correct by the hybrid method
    '''
    for rf_pred, svc_pred, H_pred in zip(rf_pred, svc_pred, H_pred):
        value = rf_pred + svc_pred + H_pred           #calculates the value for consensus
        final_pred = 1 if value >= 2 else 0     # if the majority deemed it positive, then it is positive, else it is negative
        Hybrid_pred.append(final_pred)     # add the hybrid solution's prediction to the prediction list

    print('Hybrid Solution: ')
    '''
    The Hybrid method testsits predictions against the expected results from training
    The confusion matrix finds the True negative, True positive, False negative and
    False positive predictions, which will then be used to calculate the False Rejection rate,
    False Acceptance Rate and Equal Error Rate
    '''
    cm = confusion_matrix(test_results, Hybrid_pred)
    true_neg, false_pos, false_neg, true_pos = cm.ravel()
    frr = false_neg / (false_neg + true_pos)                #False Rejection Rate
    far = false_pos / (false_pos + true_neg)                #False Acceptance Rate
    eer = (frr + far)/2                                     #Equal Error Rate
    hybrid_max_frr = max(frrs)
    hybrid_min_frr = min(frrs)
    hybrid_max_far = max(fars)
    hybrid_min_far = min(fars)

    print(f"False Reject Rate (FRR): {frr}")
    print(f"False Alarm Rate (FAR): {far}")
    print(f"Equal Error Rate (EER): {eer}")
    print(f"Max FRR: {hybrid_max_frr}")
    print(f"Min FRR: {hybrid_min_frr}")
    print(f"Max FAR: {hybrid_max_far}")
    print(f"Min FAR: {hybrid_min_far}")
    
    return
    
if __name__ == "__main__":
    main()
