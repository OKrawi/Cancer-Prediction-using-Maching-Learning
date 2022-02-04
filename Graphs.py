# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the dataset
df = pd.read_csv('cervical_cancer_complete.csv')

# Setting the style for our graphs to be tight
plt.tight_layout()

# Visualizing the “Has Cancer” Feature
plt.figure(1, figsize=(20, 10))

sns.countplot(x='Cancer', data=df)
plt.title("Counts of Patients Diagnosed with Cancer")
plt.xlabel("")
x_labels = ["No Cancer", "Has Cancer"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

# Visualizing the “Hinselmann”, "Schiller", "Citology", "Biopsy" Features
plt.figure(2, figsize=(20, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='Hinselmann', data=df)
plt.title("Counts of Patients Diagnosed with Cancer Using Hinselmann Method")
plt.xlabel("")
x_labels = ["No Cancer", "Has Cancer"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 2)
sns.countplot(x='Schiller', data=df)
plt.title("Counts of Patients Diagnosed with Cancer Using Schiller Method")
plt.xlabel("")
x_labels = ["No Cancer", "Has Cancer"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 3)
sns.countplot(x='Citology', data=df)
plt.title("Counts of Patients Diagnosed with Cancer Using Citology Method")
plt.xlabel("")
x_labels = ["No Cancer", "Has Cancer"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 4)
sns.countplot(x='Biopsy', data=df)
plt.title("Counts of Patients Diagnosed with Cancer Using Biopsy Method")
plt.xlabel("")
x_labels = ["No Cancer", "Has Cancer"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

# Visualizing the “Dx:Cancer”, "Dx:CIN", "Dx:HPV", "Dx" Features
plt.figure(3, figsize=(20, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='Dx:Cancer', data=df)
plt.title("Counts of Patients Previously Diagnosed with Cancer")
plt.xlabel("")
x_labels = ["Never Diagnosed with Cancer", "Previously Diagnosed with Cancer"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 2)
sns.countplot(x='Dx:CIN', data=df)
plt.title("Counts of Patients Previously Diagnosed with CIN")
plt.xlabel("")
x_labels = ["Never Had CIN", "Had CIN"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 3)
sns.countplot(x='Dx:HPV', data=df)
plt.title("Counts of Patients Previously Diagnosed with HPV")
plt.xlabel("")
x_labels = ["Never Had HPV", "Had HPV"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 4)
sns.countplot(x='Dx', data=df)
plt.title("Counts of Patients Previously Diagnosed with Cervical Cancer")
plt.xlabel("")
x_labels = ["Never Diagnosed with Cervical Cancer", "Previously Diagnosed with Cervical Cancer"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

# Visualizing the “Age”, "Number of sexual partners", "First sexual intercourse", "Num of pregnancies" Features
plt.figure(4, figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.boxplot(df["Age"])
plt.title("Age of All Patients")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(2, 2, 2)
plt.boxplot(df["Number of sexual partners"])
plt.title("Number of Sexual Partners for All Patients")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(2, 2, 3)
plt.boxplot(df["First sexual intercourse"])
plt.title("Age of First Sexual Intercourse for All Patients")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(2, 2, 4)
plt.boxplot(df["Num of pregnancies"])
plt.title("Number of Past Pregnancies for All Patients")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Visualizing the “Hormonal Contraceptives”, "Hormonal Contraceptives (years)", "IUD", "IUD (years)" Features
plt.figure(5, figsize=(20, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='Hormonal Contraceptives', data=df)
plt.title("Counts of Patients Who Take Hormonal Contraceptives")
plt.xlabel("")
x_labels = ["Doesn't have a Hormonal Contraceptives", "Has a Hormonal Contraceptives"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 2)
plt.boxplot(df["Hormonal Contraceptives (years)"])
plt.title("Years the Patients had been taking Hormonal Contraceptives")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(2, 2, 3)
sns.countplot(x='IUD', data=df)
plt.title("Counts of Patients With an IUD")
plt.xlabel("")
x_labels = ["Doesn't Have an AUD", "Has an IUD"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 4)
plt.boxplot(df["IUD (years)"])
plt.title("Years the Patients had had a IUD")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Visualizing the “Smokes”, "Smokes (years)", "Smokes (packs/year)", "STDs" Features
plt.figure(6, figsize=(20, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='Smokes', data=df)
plt.title("Counts of Patients Who smoke")
plt.xlabel("")
x_labels = ["Doesn't Smoke", "Smokes"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 2)
plt.boxplot(df["Smokes (years)"])
plt.title("Number of Years the Patient Has Been Smoking")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(2, 2, 3)
plt.boxplot(df["Smokes (packs/year)"])
plt.title("Number of Packets Smoked Per Year for All Patients")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(2, 2, 4)
sns.countplot(x='STDs', data=df)
plt.title("Counts of Patients With an STD")
plt.xlabel("")
x_labels = ["Doesn't Have an STD", "Has STD(s)"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

# Visualizing the “STDs (number)”, "STDs: Number of diagnosis", "STDs:condylomatosis", "STDs:vaginal condylomatosis" Features
plt.figure(7, figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.boxplot(df["STDs (number)"])
plt.title("Number of STDs the Patients have")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(2, 2, 2)
plt.boxplot(df["STDs: Number of diagnosis"])
plt.title("The Number of Times the Patients Have Been Diagnosed with an STD")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(2, 2, 3)
sns.countplot(x='STDs:condylomatosis', data=df)
plt.title("Counts of Patients Who Have Condylomatosis")
plt.xlabel("")
x_labels = ["No Condylomatosis", "Has Condylomatosis"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 4)
sns.countplot(x='STDs:vaginal condylomatosis', data=df)
plt.title("Counts of Patients Who Have Vaginal Condylomatosis")
plt.xlabel("")
x_labels = ["No Vaginal Condylomatosis", "Has Vaginal Condylomatosis"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

# Visualizing the “STDs:vulvo-perineal condylomatosis”, "STDs:syphilis", "STDs:genital herpes", "STDs:molluscum contagiosum" Features
plt.figure(8, figsize=(20, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='STDs:vulvo-perineal condylomatosis', data=df)
plt.title("Counts of Patients Who Have Vulvo-Perineal Condylomatosis")
plt.xlabel("")
x_labels = ["No Vulvo-Perineal Condylomatosis", "Has Vulvo-Perineal Condylomatosis"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 2)
sns.countplot(x='STDs:syphilis', data=df)
plt.title("Counts of Patients Who Have Syphilis")
plt.xlabel("")
x_labels = ["No Syphilis", "Has Syphilis"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 3)
sns.countplot(x='STDs:genital herpes', data=df)
plt.title("Counts of Patients Who Have Genital Herpes")
plt.xlabel("")
x_labels = ["No Genital Herpes", "Has Genital Herpes"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 4)
sns.countplot(x='STDs:molluscum contagiosum', data=df)
plt.title("Counts of Patients Who Have Molluscum Contagiosum")
plt.xlabel("")
x_labels = ["No Molluscum Contagiosum", "Has Molluscum Contagiosum"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

# Visualizing the “STDs:pelvic inflammatory disease”, "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV" Features
plt.figure(9, figsize=(20, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='STDs:pelvic inflammatory disease', data=df)
plt.title("Counts of Patients Who Have Pelvic Inflammatory Disease")
plt.xlabel("")
x_labels = ["No Pelvic Inflammatory Disease", "Has Pelvic Inflammatory Disease"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 2)
sns.countplot(x='STDs:HIV', data=df)
plt.title("Counts of Patients Who Have HIV")
plt.xlabel("")
x_labels = ["No HIV", "Has HIV"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 3)
sns.countplot(x='STDs:Hepatitis B', data=df)
plt.title("Counts of Patients Who Have Hepatitis B")
plt.xlabel("")
x_labels = ["No Hepatitis B", "Has Hepatitis B"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.subplot(2, 2, 4)
sns.countplot(x='STDs:HPV', data=df)
plt.title("Counts of Patients Who Have HPV")
plt.xlabel("")
x_labels = ["No HPV", "Has HPV"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)

plt.figure(10, figsize=(20, 10))

order = [1, 2, 3, 4, 5, 6, 7, 8]
sns.countplot(data=df[df["Cancer"] == 1], x="Age Group", order=order)
plt.title("Age Group vs. Cancer")
plt.xlabel("Age Group")
x_labels = ["Younger than 18", "18 to 24", "24 to 30", "30 to 36", "36 to 42", "42 to 50", "50 to 60", "Older than 60"]
plt.xticks(ticks=range(0, len(x_labels)), labels=x_labels)
plt.ylabel("Cancer Cases")

plt.show()
