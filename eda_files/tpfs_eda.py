import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings as wrngs

# Reading in dataset
url = "https://storage.googleapis.com/databucket_seniorproj/TPFS_Data/TPFS.csv"
df = pd.read_csv(url)

# Basic data insights
print(df.head(10))
print(f"Shape of the data \n {df.shape} \n")

print("Dataset Info: ")
df.info()
print("\n")

print(f"Description of numeric data: \n {df.describe()} \n")

print(f"Columns as list for reference: \n {df.columns.tolist()} \n")

df.replace(['-', 'N/A', 'NULL', '', 'None', 'NaN'], np.nan, inplace=True)
print("Counting missing entries")
print(df.isnull().sum())
print("\n")

print("Checking duplicate values: ")
duplicates = df.nunique()
print(duplicates)
print("\n")

plt.figure(figsize=(8,5))
sns.histplot(df['value'], bins=30, kde=True)
plt.title("Distribution of Transportation Financial Values")
plt.xlabel("Value ($)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(y=df['gov_level'], order=df['gov_level'].value_counts().index, hue=df['gov_level'], palette=sns.color_palette())
plt.title("Distribution of Transportation Financial Records by Government Level")
plt.xlabel("Count")
plt.ylabel("Government Level")
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(x=df['cash_flow'], hue=df['cash_flow'], palette="coolwarm")
plt.title("Cash Flow Counts")
plt.xlabel("Cash Flow Type")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8,8))
df['exp_type'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Proportion of Various Expenditure Types")
plt.ylabel("")
plt.show()

plt.figure(figsize=(10,5))
sns.lineplot(x=df['year'], y=df['value'], marker='o')
plt.title("Yearly Trend of Transportation Financial Values")
plt.xlabel("Year")
plt.ylabel("Value ($)")
plt.grid()
plt.show()

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_columns])
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.title("Box Plot of Numeric Variables")
plt.show()

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
sns.pairplot(df[numerical_columns], kind="reg", diag_kind="kde", plot_kws={'scatter_kws': {'alpha': 0.5}})
plt.suptitle("Pair Plot for Transportation Public Financial Statistics", y=1.02)
plt.show()

sns.lmplot(x="value", y="chained_value", data=df, scatter_kws={'alpha':0.5})
plt.title("Relationship between Value and Chained Value")
plt.show()

sns.lineplot(x="year", y="deflator", data=df)
plt.title("Deflator Trend Over the Years")
plt.show()

sns.violinplot(x="gov_level", y="value", data=df)
plt.title("Distribution of Value Across Government Levels")
plt.show()

sns.barplot(x="mode", y="value", data=df, errorbar=None)
plt.title("Comparison of Transportation Modes in Financial Values ($)")
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x="own_supporting", y="value", data=df, estimator=sum, errorbar=None)
plt.title("Revenue Source vs. Financial Value")
plt.xlabel("Revenue Type")
plt.ylabel("Total Financial Value ($)")
plt.xticks(rotation=30)
plt.show()

plt.figure(figsize=(12,10))
sns.boxplot(x="exp_type", y="value", data=df)
plt.title("Expenditure Purpose vs. Financial Value")
plt.xlabel("Expenditure Type")
plt.ylabel("Financial Value ($)")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x="trust_fund", hue="cash_flow", data=df)
plt.title("Count of Trust Funds w/ associated Cash Flow Type")
plt.xlabel("Trust Fund Usage")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x="mode", hue="cash_flow", data=df)
plt.title("Count of Transportation Modes w/ associated Cash Flow Type")
plt.xlabel("Transportation Mode")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12,6))
sns.lineplot(x="year", y="value", hue="mode", data=df, estimator=sum)
plt.title("Yearly Trend of Transportation Spending by Mode")
plt.xlabel("Year")
plt.ylabel("Total Financial Value ($)")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x="user_other", hue="mode", data=df)
plt.title("Count of User-based revenue w/ associated Transportation Mode")
plt.xlabel("Transportation Mode")
plt.ylabel("Count")
plt.show()