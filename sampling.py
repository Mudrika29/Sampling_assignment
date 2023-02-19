from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
import math
import random
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv('Creditcard_data.csv')
x=data.loc[:,data.columns!='Class']
y=data['Class']
# print((y==0).sum())
# print((y==1).sum())

#Balancing the data
from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_smote, y_smote = smote.fit_resample(x, y)
# print((y_smote==0).sum())
# print((y_smote==1).sum())

#Final balanced data
data=pd.concat([x_smote,y_smote],axis=1)
# print(data)
data.to_csv("Balanaced.csv",index=False)
data=pd.read_csv("Balanaced.csv")
# print(data)

#simple random sampling
z=1.96
p=0.5
E=0.05
n1=(pow(z,2)*p*(1-p))/pow(E,2)
random_sample = data.sample(n=int(n1))
# print(random_sample)

# #Stratified
# S=2
# n2=(pow(z,2)*p*(1-p))/pow((E/S),2)
strat_sample=data.groupby('Class', group_keys=False).apply(lambda x: x.sample(190))
# print(n2)

#Systematic
N=1526
n3 = N/(1+(N-1)/(N*pow(E,2)))
n3=int(n3)
sys_sample = data.iloc[::5]
# print(sys_sample)

#Clustering
def cluster_sampling(df, number_of_clusters):
    try:
        df['cluster_id'] = np.repeat(
            [range(1, number_of_clusters+1)], len(df)/number_of_clusters)

        indexes = []
        for i in range(0, len(df)):
            if df['cluster_id'].iloc[i] % 3 == 0:
                indexes.append(i)
        cluster_sample = df.iloc[indexes]
        cluster_sample.drop(['cluster_id'], axis=1, inplace=True)
        return (cluster_sample)

    except:
        print("The population cannot be divided into clusters of equal size!")
cluster_sample = cluster_sampling(data,7)
# print(cluster_sample)

#Convenience sampling
con_sample=data.head(350)

total_samples=[]
total_samples.append(random_sample)
total_samples.append(sys_sample)
total_samples.append(cluster_sample)
total_samples.append(strat_sample)
total_samples.append(con_sample)

accuracy=pd.DataFrame()
names=["Random sampling",'Systematic sampling',"Cluster sampling","Stratified sampling","Convenience sampling"]

for i in range(0,5):
    x=total_samples[i].drop(total_samples[i].columns[-1],axis=1)
    y=total_samples[i]['Class']
    x_train,x_test,y_train,y_test=train_test_split(x,y ,random_state=104,test_size=0.25, shuffle=True)
    acc=[]
    rf=RandomForestClassifier(n_estimators=100,random_state=42,max_depth=10, min_samples_leaf=2)
    et=ExtraTreesClassifier(n_estimators=50,random_state=40,max_depth=10, min_samples_leaf=2)
    svm=SVC(kernel = 'linear', C = 1)
    dt=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=10, min_samples_leaf=2)
    logistic=LogisticRegression(max_iter=1000,random_state=30)

    rf.fit(x_train,y_train)
    y_pred1 = rf.predict(x_test)
    score1=accuracy_score(y_test,y_pred1)
    acc.append(score1*100)

    et.fit(x_train,y_train)
    y_pred2=et.predict(x_test)
    score2=accuracy_score(y_test,y_pred2)
    acc.append(score2*100)

    svm.fit(x_train,y_train)
    y_pred3=svm.predict(x_test)
    score3=accuracy_score(y_test,y_pred3)
    acc.append(score3*100)

    dt.fit(x_train,y_train)
    y_pred4=dt.predict(x_test)
    score4=accuracy_score(y_test,y_pred4)
    acc.append(score4*100)

    logistic.fit(x_train,y_train)
    y_pred5=logistic.predict(x_test)
    score5=accuracy_score(y_test,y_pred5)
    acc.append(score5*100)

    accuracy[f"{names[i]}"]=acc
accuracy.index=["Random forest","Extra tree classifier","SVM","Decision tree","Logistic"]
print(accuracy)

    

