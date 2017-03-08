import numpy as np
import pandas as pd
from time import time

from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"

n_students = student_data.size

n_features = student_data.shape[1]

n_passed = student_data[student_data.passed == 'yes'].size

n_failed = student_data[student_data.passed == 'no'].size

grad_rate = float(n_passed) / float(n_students)

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1]

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()


def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix=col)

            # Collect the revised columns
        output = output.join(col_data)

    return output


X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

from sklearn.model_selection import train_test_split

num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=47)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)
    return end - start


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes'), (end - start)


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))

    # Train the classifier
    training_time = train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    training_f1, training_prediction_time = predict_labels(clf, X_train, y_train)
    test_f1, predict_time = predict_labels(clf, X_test, y_test)
    print "F1 score for training set: {:.4f}.".format(training_f1)
    print "F1 score for test set: {:.4f}.".format(test_f1)
    return {'train_time': training_time, 'predict_time': predict_time, 'f1_train': training_f1, 'f1_test': test_f1}


from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

clf_A = GaussianNB()
clf_B = KNeighborsClassifier()
clf_C = SVC()

classifiers = [GaussianNB(), GradientBoostingClassifier(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(),
               AdaBoostClassifier(),
               RandomForestClassifier(), BaggingClassifier()]
trainingSizes = [100, 200, 300]

results = {}

for clf in classifiers:
    sub_results = {}
    for num_train in trainingSizes:
        # Set the number of testing points
        sub_results[num_train] = train_predict(clf, X_train[:num_train], y_train[:num_train], X_test, y_test)
    results[clf.__class__.__name__] = sub_results

for item in results.items():
    print "Classifier : {}".format(item[0])
    print "Training Size\tTraining Time\tPrediction\tF1 Train\tF1 Test"
    for result in sorted(item[1].items()):
        print "{}\t\t\t\t{:.4f}\t\t\t{:.4f}\t\t\t{:.4f}\t\t{:.4f}".format(result[0],
                                                                          result[1]['train_time'],
                                                                          result[1]['predict_time'],
                                                                          result[1]['f1_train'],
                                                                          result[1]['f1_test'])

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

parameters = {'n_neighbors': range(1, 20, 1),
              'weights': ('uniform', 'distance'),
              'algorithm': ('ball_tree', 'kd_tree', 'brute'),
              'leaf_size': range(2, 18, 2),
              'p': range(1, 2, 0.1)
              }

clf = KNeighborsClassifier()

f1_scorer = make_scorer(f1_score, pos_label="yes")

grid_obj = GridSearchCV(clf, parameters, f1_scorer)

grid_obj = grid_obj.fit(X_all, y_all)

# Get the estimator
clf = grid_obj.best_estimator_

print "Best Params : {}".format(grid_obj.best_params_)

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train)[0])
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test)[0])
