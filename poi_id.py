#!/usr/bin/python
import pprint
import sys
import pickle
sys.path.append("../tools/")
import matplotlib 
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import pprint

        
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    if "NaN" in [poi_messages,all_messages]:
        return 0
       
    fraction = float(poi_messages)/all_messages
 
    return fraction

def createFraction(data_dict):
    """
    Creates two new features into the enron data dictionary.
    1) "from_poi_fraction" is the fraction of total received
    emails a person received from a POI.
    2) "to_poi_fraction" is the fraction of total sent emails
    a person sent to a POI
    """
    for name in data_dict:
        from_poi = data_dict[name]['from_poi_to_this_person']
        to_poi = data_dict[name]['from_this_person_to_poi']
        total_received = data_dict[name]['to_messages']
        total_sent = data_dict[name]['from_messages']

        data_dict[name]["from_poi_fraction"] = computeFraction(from_poi,total_received)
        data_dict[name]["to_poi_fraction"] = computeFraction(to_poi,total_sent)
       

def gather_values(data_dict,feature):
    """
    Returns all the feature values for POIs and not POIs in two separate lists
    """
    
    poi_values = [data_dict[x][feature] for x in data_dict if data_dict[x]["poi"]]
    not_poi_values = [data_dict[x][feature] for x in data_dict if not data_dict[x]["poi"]]
    return poi_values,not_poi_values


def best_pairs(max_NaN = 18,f1=False,dt=False):
    """
    Finds the 20 best feature pairs that best predict a person being a POI or not.

    Parameters:
    max_NaN - the maximum allowed number of missing feature values among POIs.
    All the features that have more NaN values among POIs are removed from analysis.
    f1 - if true then f1 score is considered; if false then precision is considered
    dt - if true then the Decision Tree algorithm is used; if false then Gaussian Naive
    Bayes is used.
    
    """   
    def train_and_predict(first,second):
        #trains the model and returns the value of desired evaluation metric
        
        features_list = ["poi",first,second]
        data = featureFormat(my_dataset, features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)

        from sklearn.naive_bayes import GaussianNB
        from sklearn import tree

        if dt:
            clf = tree.DecisionTreeClassifier()
        else:
            clf = GaussianNB()

        if f1:
            return test_classifier(clf, my_dataset, features_list,return_F1=True)
        else:
            return test_classifier(clf, my_dataset, features_list,return_precision=True)
    

    results_dict = {}
    count = 0

    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
    createFraction(data_dict)
    
    example_entry = data_dict["TOTAL"]
    fields = [x for x in example_entry if x != "poi"]
    data_dict.pop("TOTAL",0)
    my_dataset = data_dict

    #the following two loops are for removing features with too many NaN values
    bad_fields = [] 
    for f in fields:
        poi_values,_ = gather_values(data_dict,f)
        if poi_values.count("NaN") > max_NaN:
            bad_fields.append(f)
    for f in bad_fields:
        fields.remove(f)
        
    #Try the performance of all the possible feature pairs in predicting POIs
    #Write results into a dictionary
    for x in fields:    
        for y in fields:
            if (x,y) not in results_dict and (y,x) not in results_dict and x!=y:                    
                try:
                    value = train_and_predict(x,y)
                except:
                    value = 0

                results_dict[(x,y)] = value
        count+=1
        print count,"/",len(fields)

    #Print 20 best pairs from the results dictionary
    from operator import itemgetter
    sorted_result = sorted(results_dict.items(), key=itemgetter(1),reverse=True)
    for i in range(20):
        print i, sorted_result[i]


def decision_surface(first,second):
    """
    Draws a scatter plot for two features with decision surface for classifying persons into POI/not POI
    """
    
    features_list = ['poi',first,second]
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

    createFraction(data_dict)

    features = data_dict["TOTAL"]
    
    data_dict.pop("TOTAL",0)

    for i in features:
        poi,notpoi = gather_values(data_dict,i)
        print i,  round(poi.count("NaN")/18.0,2), round(notpoi.count("NaN")/127.0,2), poi.count("NaN") > 5

    data = featureFormat(data_dict, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()# Provided to give you a starting point. Try a varity of classifiers.

    clf.fit(features,labels)
    predictions = clf.predict(features)

    from sklearn.metrics import classification_report
    print classification_report(labels,predictions)

    x = data[:,1]
    y = data[:,2]
    color = data[:,0]
    
    xlim = (int(min(x)*0.9),int(max(x)*1.1))
    ylim = (int(min(y)*0.9),int(max(y)*1.1))

    import numpy as np
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 71),
                     np.linspace(ylim[0], ylim[1], 81))

    z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    z = z[:, 1].reshape(xx.shape)


    plt.scatter(x,y,c=color,s=50)
    plt.contour(xx,yy,z,[0.5],colors="k")

    plt.show()
 
   


def test_features(features,GNB=False,cycle=False,MSS=2,cycles=50,return_f1=False):

    """
    Uses either the Decision Tree or the Gaussian Naive Bayes algorithm to train a model
    for predicting if a person is a POI or not.

    Parameters:
    features - list of features that the model will use
    GNB - if true the Gaussian Naive Bayes is used; if false then decision tree is used
    cycle - if true then several min_samples_split parameter values are tried; from 2 up to
    the number given by the cycles parameter
    MSS - if cycle is false then this value specifies the min_samples_split parameter of dt
    return_f1 - if true then f1 score is returend for the use in
    the best_f1_for_all_combinations function
    """
    
    features_list = features  # You will need to use more features
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
    
    ### Task 2: Remove outliers
    data_dict.pop("TOTAL",0)
    
    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

   
    ### Create two new features
    createFraction(data_dict)
    
    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    
    labels, features = targetFeatureSplit(data)

    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    
    from sklearn import tree
    from sklearn.naive_bayes import GaussianNB
    
    if GNB:
        clf = GaussianNB()        
    
    elif cycle:
        
        precision_val = []
        recall_val = []
        f1_val = []
        minsams = list(range(2,cycles))

        for i in minsams:
            clf = tree.DecisionTreeClassifier(min_samples_split = i)# Provided to give you a starting point. Try a varity of classifiers.

            clf.fit(features,labels)
        #predictions = clf.predict(features)

            precision_val.append(test_classifier(clf, my_dataset, features_list,return_precision=True))
            recall_val.append(test_classifier(clf, my_dataset, features_list,return_recall=True))
            f1_val.append(test_classifier(clf, my_dataset, features_list,return_F1=True))
            print i, "/",cycles

       # print precision_val
       # print minsams

        plt.plot(minsams,precision_val,color="b",label="precision")
        plt.plot(minsams,recall_val,color="r",label="recall")
        plt.plot(minsams,f1_val,color="k", label="f1")

        plt.axis([0,cycles,0,1])
        plt.xlabel("Min_samples_split")
        plt.ylabel("test score")
        plt.legend()
        plt.show()

    else:
        clf = tree.DecisionTreeClassifier(min_samples_split = MSS)

    clf.fit(features,labels)
    #print clf.feature_importances_
    if return_f1:
        return test_classifier(clf,my_dataset,features_list,return_F1=True)
    else:
        print test_classifier(clf,my_dataset,features_list)
    
    dump_classifier_and_data(clf, my_dataset, features_list)


    
def best_f1_from_all_combinations(feature_list,MSS_list):

    """
    Tries all the possible feature combinations from the given feature list and returns the
    best 20 combinations based on f1 metric. Decision tree algorithm is used.

    Parameters:
    feature_list - list of features to be tested
    MSS_list - list of min_samples_splil values to be tested

    """
    
    import itertools

    result = []
    for nr in range(1,len(feature_list)):
        print nr,"/", len(feature_list)
        for i in itertools.combinations(feature_list,nr):
            for mss in MSS_list:
                result.append([test_features(["poi"]+list(i),MSS=mss,return_f1=True),i,mss])

    pprint.pprint(sorted(result,reverse=True)[:20])    
    
    

    
#features from the top pairs and the two features created by myself for use combinational analysis.
features2= ["expenses","from_this_person_to_poi","other","bonus","to_messages","exercised_stock_options","from_poi_fraction","to_poi_fraction"]

#best_f1_from_all_combinations(features2,[2,5,10,20])

#decision_surface('deferred_income', 'exercised_stock_options')
#normal_gaussian_flow()
#best_pairs(max_NaN=7,f1=True)


features = ["poi",'expenses', 'from_this_person_to_poi', 'exercised_stock_options']
print test_features(features,MSS=20)

