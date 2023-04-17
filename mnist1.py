import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import graphviz

# load the Iris dataset
mnist = load_digits()
X, y = mnist.data, mnist.target
classNames = (mnist.target_names).astype(str)

# create and print the decision tree
dtc = tree.DecisionTreeClassifier(criterion="entropy")
dtc.fit(X, y)
tree.plot_tree(dtc)
dot_data = tree.export_graphviz(dtc, out_file=None,
feature_names=mnist.feature_names,
class_names=classNames,
filled=True, rounded=True,
special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("mnisttree")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_pred = dtc.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
