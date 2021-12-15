#import all the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from scipy import stats
import sklearn


df = pd.read_csv("https://raw.githubusercontent.com/dioapw/school-project-clustering-classification/main/kendaraan_train.csv")
#Reading the dataset in a dataframe using Pandas

df.head(10)  #Print first ten observations


df.head() # show first 5 rows


df.tail() # last 5 rows


df.columns # list all column names


df.shape # get number of rows and columns


df.info() # additional info about dataframe


df.describe() # statistical description, only for numeric values


df.value_counts(dropna=False) # count unique values


df.sort_values('Premi',ascending=False).head(10) # Sort the Data frame based on Premi in ascending value and print first 10 observation.


df.sort_values('Lama_Berlangganan', ascending=False).head(10) #Sort the Data frame based on Lama_Berlangganan in ascending value and print first 10 observation.


#Plots in matplotlib reside within a figure object, use plt.figure to create new figure
fig=plt.figure()
#Create 3 subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)

#Plot Histogram by age
#Variable
ax.hist(df['Umur'],bins=range(20, 85))
#Labels and Tit
plt.title('Distribusi Umur')
plt.xlabel('Umur')
plt.ylabel('Pelanggan')
plt.show()


#Plot Histogram by Premi
#Plots in matplotlib reside within a figure object, use plt.figure to create new figure
fig=plt.figure()
#Create 3 subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)

#Variable
ax.hist(df['Premi'],bins=100)
#Labels and Tit
plt.title('Distribusi Premi')
plt.xlabel('Premi')
plt.ylabel('Pelanggan')
plt.show()


#Plot Histogram by Lama_Berlangganan
#Plots in matplotlib reside within a figure object, use plt.figure to create new figure
fig=plt.figure()
#Create 3 subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)

#Variable
ax.hist(df['Lama_Berlangganan'], bins=range(10, 299))
#Labels and Tit
plt.title('Distribusi Lama Berlangganan')
plt.xlabel('Lama Berlangganan')
plt.ylabel('Pelanggan')
plt.show()


#Plot Histogram by Kanal Penjualan
#Plots in matplotlib reside within a figure object, use plt.figure to create new figure
fig=plt.figure()
#Create 3 subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)

#Variable 
ax.hist(df['Kanal_Penjualan'],bins=range(1, 163))
#Labels and Tit
plt.title('Distribusi Kanal Penjualan')
plt.xlabel('Kanal Penjualan')
plt.ylabel('Pelanggan')
plt.show()


# Scatter Plot based on Age and premi
#Plots in matplotlib reside within a figure object, use plt.figure to create new figure 
fig=plt.figure()
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)

#Variable
ax.scatter(df['Umur'],df['Premi'])
#Labels and Tit
plt.title('Umur dan Premi distribution')
plt.xlabel('Umur')
plt.ylabel('Premi')
plt.show()


# Scatter Plot based on premi and Lama_Berlangganan
#Plots in matplotlib reside within a figure object, use plt.figure to create new figure 
fig=plt.figure()
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)

#Variable
ax.scatter(df['Premi'],df['Lama_Berlangganan'])
#Labels and Tit
plt.title('Premi dan Lama Berlangganan distribution')
plt.xlabel('Premi')
plt.ylabel('Lama Berlangganan')
plt.show()


# Box-plot based on age
sns.boxplot(x=df['Umur']) 
sns.despine()


# Box-plot based on Premi
sns.boxplot(x=df['Premi']) 
sns.despine()


# Box-plot based on Lama_Berlanggan
sns.boxplot(x=df['Lama_Berlangganan']) 
sns.despine()


# Box-plot based on Kode_Daerah
sns.boxplot(x=df['Kode_Daerah']) 
sns.despine()


# Box-plot based on Kanal_Penjualan
sns.boxplot(x=df['Kanal_Penjualan']) 
sns.despine()


plt.figure(figsize = (16, 10))
sns.heatmap(df.corr(), annot=True)
plt.show()


df.corr()


freq = df.groupby(['Umur','Sudah_Asuransi']) #Group the Data frame based on Umur and Sudah_Asuransi
freq.size()


freq = df.groupby(['Umur','Premi']) #Group the Data frame based on Umur and Premi
freq.size()


freq = df.groupby(['Umur','Lama_Berlangganan']) #Group the Data frame based on Umur and Lama_Berlangganan
freq.size()


freq = df.groupby(['Umur','Kendaraan_Rusak']) #Group the Data frame based on Umur and Kendaraan_Rusak
freq.size()


group = df.groupby(['Jenis_Kelamin','SIM','Umur_Kendaraan','Kendaraan_Rusak','Sudah_Asuransi'])
group.describe()


# Find a duplicate rows
duplicateDFRow = df[df.duplicated()]
print(duplicateDFRow)


# Identify missing values of dataframe
df.isnull().sum()


#Example to impute missing values
df.dropna(inplace=True) # Using dropna to drop all NaN values in the Dataframe


# Identify missing values of dataframe
df.isnull().sum()


Q1= df['Premi'].quantile(0.25)
Q3 = df['Premi'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR

print("Upper whisker: ", upper_limit)
print("Lower Whisker: ", lower_limit)


df[(df['Premi'] < lower_limit) | (df['Premi'] > upper_limit)]


df = df[(df['Premi'] > lower_limit) & (df['Premi'] < upper_limit)]


sns.boxplot(x=df['Premi']) 
sns.despine()


#Penghapusan fitur-fitur yang tidak diperlukan di dataframe
df = df.drop(['id'], axis = 1)
df


df.info()


df.dtypes # show the datatypes


# convert data type of Umur, Premi, Kanal_Penjualan and Lama_Berlangganan column
# into integer
df.Umur = df.Umur.astype(np.int64)
df.Premi = df.Premi.astype(np.int64)
df.Kanal_Penjualan = df.Kanal_Penjualan.astype(np.int64)
df.Lama_Berlangganan = df.Lama_Berlangganan.astype(np.int64)

# show the datatypes
print(df.dtypes)


df['Jenis_Kelamin'] = df['Jenis_Kelamin'].replace(['Wanita'], 2)
df['Jenis_Kelamin'] = df['Jenis_Kelamin'].replace(['Pria'], 1)

df['Umur_Kendaraan'] = df['Umur_Kendaraan'].replace(['< 1 Tahun'], 1)
df['Umur_Kendaraan'] = df['Umur_Kendaraan'].replace(['1-2 Tahun'], 2)
df['Umur_Kendaraan'] = df['Umur_Kendaraan'].replace(['> 2 Tahun'], 3)

df['Kendaraan_Rusak'] = df['Kendaraan_Rusak'].replace(['Tidak'], 0)
df['Kendaraan_Rusak'] = df['Kendaraan_Rusak'].replace(['Pernah'], 1)

df.head()


# Converting categorical data into object data types
df=df.astype({'Jenis_Kelamin':object, 'SIM': object, 'Sudah_Asuransi': object, 'Kanal_Penjualan': object, 'Kendaraan_Rusak': object})
df.info()
