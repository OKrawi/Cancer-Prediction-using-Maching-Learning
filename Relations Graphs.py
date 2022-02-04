# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Reading the dataset
df = pd.read_csv('cervical_cancer_complete.csv')

# Setting the style for our graphs to be tight
plt.tight_layout()

#
plt.figure(1, figsize=(20, 10))

order = [1, 2, 3, 4, 5, 6, 7, 8]
sns.countplot(data=df[df["Cancer"] == 1], x="Age Group", order=order)
plt.title("Age Group vs. Cancer")
plt.xlabel("Age Group")
x_labels = ["Younger than 18", "18 to 24", "24 to 30", "30 to 36", "36 to 42", "42 to 50", "50 to 60", "Older than 60"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)
plt.ylabel("Cancer Cases")

plt.figure(2, figsize=(20, 10))

sns.countplot(data=df[df["Cancer"] == 1], x="Age")
plt.title("Age vs. Cancer")
plt.xlabel("Age")
plt.ylabel("Cancer Cases")

plt.show()
