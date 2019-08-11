import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

df= pd.read_csv('cars.csv')
df.head()
df.corr(method ='pearson').head(1)
print(df.shape)
df.info()
col=['Price','Mileage','Cylinder','Liter','Doors','Cruise','Sound','Leather']
df.isnull().sum()
df.Trim.value_counts()
df.Type.value_counts()
df.Make.value_counts()
df.Model.value_counts()
df.Mileage.plot(kind='box')
print(df.Cylinder.value_counts())
df.Cylinder.plot(kind='box')
print(df.Liter.value_counts())
df.Liter.plot(kind='box')
df.Doors.value_counts()
df.Cruise.value_counts()
df.Sound.value_counts()
df.Leather.value_counts()


outliers=[]
def detect_outlier(data_1):
    
    threshold=2.6
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers
  
detect_outlier(df.Mileage)
df['Mileage']=df.Mileage.replace((41829, 42691, 41566, 50387, 48991),(df.Mileage.mean(),df.Mileage.mean(),df.Mileage.mean(),df.Mileage.mean(),df.Mileage.mean()))
sns.boxplot(df.Mileage)
z = np.abs(stats.zscore(df.Mileage))
print(np.where(z > (2.6)))
sns.boxplot(df.Price)
maker=pd.get_dummies(df['Make'],prefix="Maker")
tp=pd.get_dummies(df.Type,prefix='Type')
trim=pd.get_dummies(df.Trim,prefix='Trim')
model=pd.get_dummies(df.Model,prefix='Model')
df=df.join(maker)
df=df.join(tp)
df=df.join(trim)
df=df.join(model)
df=df.drop(['Make','Type','Trim','Model'],axis=1)
df.head()
dfX = pd.DataFrame(df['Price'])
col = dfX['Price'].values.reshape(-1, 1)

scalers = [
    #('Unscaled data', X),
    ('standard scaling', StandardScaler()),
    ('min-max scaling', MinMaxScaler()),
    ('max-abs scaling', MaxAbsScaler()),
    ('robust scaling', RobustScaler(quantile_range=(25, 75))),
    ('quantile transformation (uniform pdf)', QuantileTransformer(output_distribution='uniform')),
    ('quantile transformation (gaussian pdf)', QuantileTransformer(output_distribution='normal')),
    ('sample-wise L2 normalizing', Normalizer())
]

for scaler in scalers:
    dfX[scaler[0]] = scaler[1].fit_transform(col)
    
dfX.head()
scaler =RobustScaler()
robust_df = scaler.fit_transform(df)
rdf = pd.DataFrame(robust_df, columns=df.columns)
rdf.head()
X=rdf.drop("Price",axis=1)
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.03)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_test, y_test)],
             verbose=False)
print("Training accuracy",my_model.score(X_train,y_train))
print("Testing accuracy",my_model.score(X_test,y_test))
