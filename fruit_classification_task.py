"""Import required modules and load data file"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

fruits = pd.read_table("fruit_data_with_colors.txt")
# print(fruits)    To print Table
print("Shape of data:", fruits.shape)
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
print("Label and fruit names: ", lookup_fruit_name)

"""Create train-test split"""
X = fruits[['mass', 'width', 'height', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

"""Graph"""
def graph():
    # plotting a 3D scatter plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c=y_train, marker='o', s=100)
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    ax.set_zlabel('color_score')
    plt.show()


"""Create classifier object"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)   # k=5

"""Train the Classifier using training data"""
knn.fit(X_train, y_train)

"""Estimate the accuracy of the classifier on the future data, using the test data"""
print("Accuracy of KNN Classifier", knn.score(X_test, y_test))

"""Use the trained k-NN classifier model to classify new, previously unseen objects"""
fruit_prediction = knn.predict([[20, 4.3, 5.5, 0.55]])
print(fruit_prediction)
print(lookup_fruit_name[fruit_prediction[0]])

"""Plot the decision boundaries of the k-NN classifier"""
def decisionBoundaries():
    from adspy_shared_utilities import plot_fruit_knn
    plot_fruit_knn(X_train, y_train, 5, 'uniform')   # we choose 5 nearest neighbors
