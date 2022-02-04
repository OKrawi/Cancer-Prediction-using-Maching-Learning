# Importing required libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Reading the dataset and dropping unwanted target variables
df = pd.read_csv('cervical_cancer_complete.csv')
df = df.drop(["Hinselmann", "Schiller", "Citology", "Biopsy"], axis=1)

# Creating two correlation graphs, one for most correlated features, and one for least correlated features
corr = df.corr()
corr_target_head = abs(corr["Has Cancer"]).sort_values(ascending=False).head(15).index.values
plot_corr_head = df[corr_target_head.tolist()].corr()

corr_target_tail = abs(corr["Has Cancer"]).sort_values(ascending=False).tail(14).index.values
corr_target_tail = np.insert(corr_target_tail, 0, "Has Cancer", axis=0)
plot_corr_tail = df[corr_target_tail.tolist()].corr()

plt.figure(1, figsize=(12, 9))
plt.title("Correlation Graph for Most Correlated Features")
cmap = sns.diverging_palette(1000, 120, as_cmap=True)
mask = np.triu(np.ones_like(plot_corr_head, dtype=bool))
ax = sns.heatmap(plot_corr_head, annot=True, fmt='.1%',  linewidths=.05, cmap=cmap, mask=mask)
ax.set_xlabel('\nFeatures')
plt.tight_layout()

plt.figure(2, figsize=(12, 9))
plt.title("Correlation Graph for Least Correlated Features")
cmap = sns.diverging_palette(1000, 120, as_cmap=True)
mask = np.triu(np.ones_like(plot_corr_tail, dtype=bool))
ax = sns.heatmap(plot_corr_tail, annot=True, fmt='.1%',  linewidths=.05, cmap=cmap, mask=mask)
ax.set_xlabel('\nFeatures')
plt.tight_layout()

plt.show()
