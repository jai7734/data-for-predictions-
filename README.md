# data-for-predictions-

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\HP\Downloads\data_for_predictions (1).csv')
df

pd.set_option('display.max_columns', None)

df.head()

# BASIC CHECK

df.info()

df.describe()

df.describe(include='O')

df1 = df.drop(['id'] , axis = 1)

df1.tail()

df.value_counts('id')

## EDA

### Univerient analysis

pip install sweetviz

import sweetviz as sv
my_report = sv.analyze(df)
my_report.show_html

my_report.show_html

df

df.columns

df_cat = []
for i in df:
    if len(df[i].unique()) <= 20:
        df_cat.append(i)

df_cat

len(df_cat)

df_cont = []
for i in df:
    if len(df[i].unique()) > 20:
        df_cont.append(i)

df_cont

df1_cat = df[df_cat]
df1_cat.head()

df2_cont = df[df_cont]
df2_cont.head()

plt.figure(figsize = (10,30))
plotnumber = 1
for i in df1_cat:
    plt.subplot(13,2, plotnumber)
    sns.countplot(x = df1_cat[i])
    plt.xticks(rotation = 20)
    plotnumber = plotnumber + 1
plt.tight_layout()

df.isnull().sum()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\HP\Downloads\clean_data_after_eda.csv')
df

df.info()

df.describe()

df.head(3)

# Convert date columns to datetime
df["date_activ"] = pd.to_datetime(df["date_activ"], format='%Y-%m-%d')
df["date_end"] = pd.to_datetime(df["date_end"], format='%Y-%m-%d')
df["date_modif_prod"] = pd.to_datetime(df["date_modif_prod"], format='%Y-%m-%d')
df["date_renewal"] = pd.to_datetime(df["date_renewal"], format='%Y-%m-%d')


## 3. Feature engineering

Creating new features from the existing data.


# Convert date columns to datetime
df["date_activ"] = pd.to_datetime(df["date_activ"], format='%Y-%m-%d')
df["date_end"] = pd.to_datetime(df["date_end"], format='%Y-%m-%d')
df["date_modif_prod"] = pd.to_datetime(df["date_modif_prod"], format='%Y-%m-%d')
df["date_renewal"] = pd.to_datetime(df["date_renewal"], format='%Y-%m-%d')

# Inspect the DataFrame
print(df.head())
print(df.columns)


# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
print(categorical_columns)

# Encode categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)


