# EX-06 FEATURE TRANSFORMATION
## Aim:
To read the given data and perform Feature Transformation process and save the data to a file.

## Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## Algorithm:
Step1:
Read the given Data.

Step2:
Clean the Data Set using Data Cleaning Process.

Step3:
Apply Feature Transformation techniques to all the features of the data set.

Step4:
Print the transformed features.

## Program:
Developed By: DELLI PRIYA L

Register No: 212222230029

### Importing libraries and reading csv file:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
df=pd.read_csv("Data_to_Transform.csv")
```
### Basic Information:
```
df.head()
df.info()
df.info()
```
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex06/assets/121166075/0ea2cef7-0b0c-4282-990c-128459f08f6d)

### Before Transformation:
```
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex06/assets/121166075/477e38b9-4643-4ea4-82ee-f53f1e279d55)

```
sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.title("Highly Negative Skew")
plt.show()
```
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex06/assets/121166075/a12077a6-76c8-41da-b381-b24cb617cd6b)

```
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex06/assets/121166075/abdc5546-277b-4fcf-900d-1162e66efbd1)

```
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex06/assets/121166075/46e8b7b7-a2c7-4baf-8adb-d8fb2b5873f5)

### Log Transformation:
```
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex06/assets/121166075/96f75196-aaf3-43ad-ad74-d122173fc3e3)

```
df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex06/assets/121166075/59bdac01-d250-4884-beee-6500bf51534e)

### Reciprocal Transformation:
```
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex06/assets/121166075/047b3d02-3505-45d2-972e-1cf6ccda7e19)

### SquareRoot Transformation:
```
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex06/assets/121166075/709f5d13-19c3-44f7-8c0e-77feaa091059)

### Power Transformation:
```
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex06/assets/121166075/e5ab41de-1a43-4e1a-afc7-633a0ac81154)

```
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex06/assets/121166075/dc2255a3-8559-4a55-a510-1d1ec920ef60)

### Quantile Transformation:
```
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate  Negative Skew")
plt.show()
```
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex06/assets/121166075/d261912c-eee6-4fbc-8d05-a81917a45a5a)

## Result:
Thus feature transformation is done for the given dataset.




