# Importing the required libraries
import pandas as pd
import numpy as np

# Reading the dataset
df = pd.read_csv("risk_factors_cervical_cancer.csv", na_values=["?"])

# Checking if there are any missing data
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 400)

# Exploring the dataset
print("The dataset has {} rows and {} columns".format(df.shape[0], df.shape[1]))
print("Statistical Description of the data:")
print(df.describe().transpose())
print("\n")
print(df.info())
print("\n")
print(df.isnull().sum())

# Dropping the columns with more than 90% missing data or where all the data is zero
df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis', 'STDs:AIDS',
         'STDs:cervical condylomatosis'], axis=1, inplace=True)

print("The dataset has {} rows and {} columns".format(df.shape[0], df.shape[1]))

# Filling the missing numerical data in the dataset using the median
numerical_features = ["Number of sexual partners", "First sexual intercourse", "Num of pregnancies"]

for feature in numerical_features:
    df[feature] = df[feature].fillna(df[feature].median())

# Filling the missing categorical data in the dataset using mode
categorical_features = ["STDs", "Smokes", "Hormonal Contraceptives", "IUD"]

for feature in categorical_features:
    df[feature] = df[feature].fillna(df[feature].value_counts().index[0])

# Filling the missing dependent variables with zero if the column they are dependent on is zero
df.loc[df['IUD'] == 0, 'IUD (years)'] = 0
df.loc[df['STDs'] == 0, 'STDs (number)'] = 0
df.loc[df['Smokes'] == 0, 'Smokes (years)'] = 0
df.loc[df['Smokes'] == 0, 'Smokes (packs/year)'] = 0
df.loc[df['Hormonal Contraceptives'] == 0, 'Hormonal Contraceptives (years)'] = 0

# Filling the remaining null values of the dependent variables with the median
# of only the values corresponding to non zero values in the variable they are dependent on
df['IUD (years)'] = df['IUD (years)'].fillna(df.loc[df['IUD'] == 1]['IUD (years)'].median())
df['STDs (number)'] = df['STDs (number)'].fillna(df.loc[df['STDs'] == 1]['STDs (number)'].median())
df['Smokes (years)'] = df['Smokes (years)'].fillna(df.loc[df['Smokes'] == 1]['Smokes (years)'].median())
df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(df.loc[df['Smokes'] == 1]['Smokes (packs/year)'].median())
df['Hormonal Contraceptives (years)'] = \
    df['Hormonal Contraceptives (years)'].fillna(df.loc[df['Hormonal Contraceptives'] == 1]
                                                 ['Hormonal Contraceptives (years)'].median())

# Filling the remaining null values of the dependent variables with the mode
# of only the values corresponding to non zero values in the variable they are dependent on
std_categorical_features = ["STDs:condylomatosis", "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis",
                            "STDs:syphilis", "STDs:pelvic inflammatory disease", "STDs:genital herpes",
                            "STDs:molluscum contagiosum", "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV"]

for feature in std_categorical_features:
    df.loc[df['STDs'] == 0, feature] = 0
    df[feature] = df[feature].fillna(df.loc[df['STDs'] == 1][feature].value_counts().index[0])

# Checking if there are any missing data
print("After Imputation: ")
print(df.info())
print(df.isnull().sum())

# Creating the "Cancer" column if one diagnosis method diagnoses the patient with cancer
df["Cancer"] = np.where((df["Hinselmann"] + df["Schiller"] + df["Citology"] + df["Biopsy"]) > 0, 1, 0)

# Creating the Age Group column
df["Age Group"] = np.where(df["Age"] < 18, 1,
                           (np.where((df["Age"] >= 18) & (df["Age"] < 24), 2,
                            (np.where((df["Age"] >= 24) & (df["Age"] < 30), 3,
                             (np.where((df["Age"] >= 30) & (df["Age"] < 36), 4,
                              (np.where((df["Age"] >= 36) & (df["Age"] < 42), 5,
                               (np.where((df["Age"] >= 42) & (df["Age"] < 50), 6,
                                (np.where((df["Age"] >= 50) & (df["Age"] < 60), 7,
                                          8)))))))))))))

# Exporting the new dataset
if df.to_csv("cervical_cancer_complete.csv", index=False):
    print("Preprocessing complete")
