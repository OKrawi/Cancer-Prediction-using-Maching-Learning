# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score, \
    accuracy_score, recall_score, precision_score
from sklearn.model_selection import RepeatedKFold, cross_val_score
from matplotlib import pyplot as plt
import seaborn as sns

# Reading the dataset
df = pd.read_csv("cervical_cancer_complete.csv", index_col=0)

# Creating a dataframe with all training data except the target columns
X = df.drop(columns=["Cancer", "Hinselmann", "Schiller", "Citology", "Biopsy"])

# Separating target values
y = df["Cancer"]

# Applying a standard scaler to the data
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Creating the decision tree classifier and fitting the data
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Applying the classifier to the testing data
y_pred = dt_classifier.predict(X_test)

# Checking the accuracy, recall, precision, and the f1 score of the model on the test data
print("Accuracy of DT algorithm: {:.3f} %".format(accuracy_score(y_test, y_pred)))
print("F1 Score of DT algorithm: {:.3f} %".format(f1_score(y_test, y_pred)))
print("Recall Score of DT algorithm: {:.3f} %".format(recall_score(y_test, y_pred)))
print("Precision Score of DT algorithm: {:.3f} %".format(precision_score(y_test, y_pred)))

# Getting the Cross Validation score
cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
scores = cross_val_score(dt_classifier, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(f'Cross Validation Accuracy of DT algorithm: {np.mean(scores):.3f} ({np.std(scores):.3f})')

# Getting the AUC score
pred_prob = dt_classifier.predict_proba(X_test)
auc_score = roc_auc_score(y_test, pred_prob[:, 1])
print("AUC Score of DT algorithm: {:.3f} %".format(auc_score))

# Plotting the roc curve
plt.figure(1, figsize=(12, 6))
fpr, tpr, thresh = roc_curve(y_test, pred_prob[:, 1], pos_label=1)
plt.plot(fpr, tpr, linestyle='--', color='orange', label='DT')
plt.title('ROC curve for DT Algorithm')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('ROC DT', dpi=300)

# Creating and graphing the confusion matrix
cm = confusion_matrix(y_test, y_pred)

group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)

plt.figure(2, figsize=(12, 6))
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

ax.set_title('Confusion Matrix for DT Algorithm\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])

plt.savefig('CM DT', dpi=300)

plt.show()
