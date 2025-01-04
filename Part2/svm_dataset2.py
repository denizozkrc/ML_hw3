import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils import shuffle

dataset, labels = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

scaler = StandardScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)

parameters = [{'model__kernel': ['rbf', 'poly'], 'model__C': [1, 10]}]
model = SVC()

pipeline = Pipeline([('scaler',  scaler), ('model', model)])  # must do for scaling like said in the pdf

grid_search = GridSearchCV(pipeline, parameters, cv=10, scoring='accuracy')
params_len = len(parameters[0]['model__kernel']) * len(parameters[0]['model__C'])
means = np.zeros(params_len)
stds = np.zeros(params_len)
CIs = np.zeros(params_len)
n_test = len(dataset)/10
Number_of_iterations = 5
for i in range(Number_of_iterations):
    dataset, labels = shuffle(dataset, labels, random_state=0)
    grid_search.fit(dataset, labels)  # makes 10fold startifiedkfold itself. Scales on training data
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    means += np.array(grid_search.cv_results_['mean_test_score'])
    stds += np.array(grid_search.cv_results_['std_test_score'])
    print(f" for iteration {i+1}, best parameters: {best_params}, best score: {best_score}")

means = means/Number_of_iterations
stds = stds/Number_of_iterations
CIs = 1.96 * stds / np.sqrt(n_test)  # subtract/add from/to mean
print("\n parameter configurations: ", grid_search.cv_results_["params"], "\n")
print("average accuracies: ", means)
print("CI values to be adde/subtracted from mean values: ", CIs)

