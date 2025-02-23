import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import warnings
from pandas_profiling import ProfileReport

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the dataset
used_cars_df = pd.read_csv("used_cars.csv")

# Display basic information about the dataset
print(used_cars_df.head())  # First 5 rows
print(used_cars_df.tail())  # Last 5 rows
print(used_cars_df.info())  # Data types and missing values
print(used_cars_df.nunique())  # Unique values per column
print(used_cars_df.isnull().sum())  # Count of missing values
print((used_cars_df.isnull().sum() / len(used_cars_df)) * 100)  # Percentage of missing values

# Removing the Serial Number column
used_cars_df.drop(['S.No.'], axis=1, inplace=True)
print(used_cars_df.info())

# Feature Engineering: Creating new columns
used_cars_df['Car_Age'] = date.today().year - used_cars_df['Year']  # Calculate car age
used_cars_df['Brand'] = used_cars_df.Name.str.split().str.get(0)  # Extract brand from name
used_cars_df['Model'] = used_cars_df.Name.str.split().str.get(1) + " " + used_cars_df.Name.str.split().str.get(2)  # Extract model
print(used_cars_df[['Name', 'Brand', 'Model']].head())

# Cleaning Brand names for consistency
brand_corrections = {"ISUZU": "Isuzu", "Mini": "Mini Cooper", "Land": "Land Rover"}
used_cars_df["Brand"].replace(brand_corrections, inplace=True)

# Exploratory Data Analysis (EDA)
print(used_cars_df.describe().T)  # Statistical summary
print(used_cars_df.describe(include='all').T)  # Summary including categorical variables

# Separating categorical and numerical columns
categorical_cols = used_cars_df.select_dtypes(include=['object']).columns
numerical_cols = used_cars_df.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:", categorical_cols)
print("Numerical Variables:", numerical_cols)

# Univariate Analysis: Distribution and outliers
for col in numerical_cols:
    print(f"{col}: Skewness -", round(used_cars_df[col].skew(), 2))
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 2, 1)
    used_cars_df[col].hist(grid=False)
    plt.ylabel('Count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=used_cars_df[col])
    plt.show()

# Bar plots for categorical variables
fig, axes = plt.subplots(3, 2, figsize=(18, 18))
fig.suptitle('Bar plot for all Categorical Variables')

sns.countplot(ax=axes[0, 0], x='Fuel_Type', data=used_cars_df, color='blue')
sns.countplot(ax=axes[0, 1], x='Transmission', data=used_cars_df, color='blue')
sns.countplot(ax=axes[1, 0], x='Owner_Type', data=used_cars_df, color='blue')
sns.countplot(ax=axes[1, 1], x='Location', data=used_cars_df, color='blue')
sns.countplot(ax=axes[2, 0], x='Brand', data=used_cars_df, color='blue', order=used_cars_df['Brand'].value_counts().head(20).index)
sns.countplot(ax=axes[2, 1], x='Model', data=used_cars_df, color='blue', order=used_cars_df['Model'].value_counts().head(20).index)

axes[1, 1].tick_params(labelrotation=45)
axes[2, 0].tick_params(labelrotation=90)
axes[2, 1].tick_params(labelrotation=90)

# Data Transformation: Log Transformation Function
def log_transform(df, columns):
    for col in columns:
        if (df[col] == 1.0).all():
            df[col + '_log'] = np.log(df[col] + 1)
        else:
            df[col + '_log'] = np.log(df[col])
    print(df.info())

log_transform(used_cars_df, ['Kilometers_Driven', 'Price'])

# Distribution after log transformation
sns.histplot(used_cars_df["Kilometers_Driven_log"], kde=True)
plt.xlabel("Kilometers Driven (Log)")
plt.show()

# Bivariate Analysis: Pairplot
sns.pairplot(data=used_cars_df.drop(['Kilometers_Driven', 'Price'], axis=1))
plt.show()

# Bar plots comparing price with different factors
fig, axes = plt.subplots(4, 2, figsize=(12, 18))
used_cars_df.groupby('Location')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axes[0, 0])
used_cars_df.groupby('Transmission')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axes[0, 1])
used_cars_df.groupby('Fuel_Type')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axes[1, 0])
used_cars_df.groupby('Owner_Type')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axes[1, 1])
used_cars_df.groupby('Brand')['Price_log'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axes[2, 0])
used_cars_df.groupby('Model')['Price_log'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axes[2, 1])
used_cars_df.groupby('Seats')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axes[3, 0])
used_cars_df.groupby('Car_Age')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axes[3, 1])
plt.subplots_adjust(hspace=1.0, wspace=0.5)
sns.despine()

# Correlation Matrix (Multivariate Analysis)
plt.figure(figsize=(12, 7))
sns.heatmap(used_cars_df.drop(['Kilometers_Driven','Price'], axis=1).select_dtypes(include=['number']).corr(), annot=True, vmin=-1, vmax=1)
plt.show()

# Impute Missing Values
used_cars_df['Mileage'].replace(0.0, np.nan, inplace=True)
used_cars_df['Mileage'].fillna(value=used_cars_df['Mileage'].mean(), inplace=True)
used_cars_df['Seats'] = used_cars_df.groupby(['Model', 'Brand'])['Seats'].transform(lambda x: x.fillna(x.median()))
used_cars_df['Engine'] = used_cars_df.groupby(['Brand', 'Model'])['Engine'].transform(lambda x: x.fillna(x.median()))
used_cars_df['Power'] = used_cars_df.groupby(['Brand', 'Model'])['Power'].transform(lambda x: x.fillna(x.median()))

# Verify missing values are handled
print("Missing values after cleanup:")
print(used_cars_df.isnull().sum())

# Generate EDA Report
profile = ProfileReport(used_cars_df, title="EDA Report", explorative=True)
profile.to_file("eda_report.html")
