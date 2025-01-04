import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils import shuffle

dataset, labels = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

scaler = StandardScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)

parameters = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'C': [0.1, 1, 10, 100]}
model = SVC()

grid_search = GridSearchCV(model, parameters, cv=10, scoring='accuracy')

for i in range(5):
    dataset, labels = shuffle(dataset, labels, random_state=0)
    grid_search.fit(dataset, labels)  # makes 10fold startifiedkfold itself
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f" for iteration {i+1}, best parameters: {best_params}, best score: {best_score}")
