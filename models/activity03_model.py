import pandas as pd
import numpy as np
from get_data import get_df
from sklearn import preprocessing

result=get_df()
activities=['activity_01','activity_04','activity_05','activity_07']
act3=result[result['activities']=='activity_03']

act_arr=[]
act_arr=np.array(act_arr)

count=0
for i in act3.iloc:
    if i[3]=='80yo+' and i[4]=='middle' and i[2]=='no' and count%2==0:
        act_arr=np.append(act_arr,activities[0])
    elif i[3]=='80yo+' and i[4]=='middle' and i[2]=='no' and count%2!=0:
        act_arr=np.append(act_arr,activities[2])
    elif i[3]=='0-19yo' and i[4]=='low' and i[2]=='yes':
        act_arr=np.append(act_arr,activities[3])
    else:
        act_arr=np.append(act_arr,activities[1])
    count+=1

act3['target']=act_arr
act3=act3.drop(columns=['trace_id','activities'])

# veri ön işleme
le=preprocessing.LabelEncoder()
target=act3.iloc[:,3:4].values
target[:,0]=le.fit_transform(act3.iloc[:,3:4])
df_target=pd.DataFrame(target,columns=['target'],dtype='int32')

age=act3.iloc[:,1:2].values
ohe=preprocessing.OneHotEncoder()
age=ohe.fit_transform(age).toarray()
df_age=pd.DataFrame(age,columns=['0-19yo','20-39yo','40-59yo','60-79yo','80yo+'],dtype='int32')

income=act3.iloc[:,2:3].values
ohe=preprocessing.OneHotEncoder()
income=ohe.fit_transform(income).toarray()
df_income=pd.DataFrame(income,columns=['high','low','middle'],dtype='int32')

employed=act3.iloc[:,0:1].values
employed[:,0]=le.fit_transform(act3.iloc[:,0])
df_employed=pd.DataFrame(employed,columns=['employed'],dtype='int32')

frames=[df_employed,df_age,df_income,df_target]
result_act3= pd.concat(frames,axis=1)
result_act3.to_csv('3s2.csv',index=False)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
X=result_act3.iloc[:,0:9]
Y=result_act3.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=15,random_state=1)
mtf = MultiOutputClassifier(rfc, n_jobs=-1)
mtf.fit(x_train, y_train.values.reshape(-1,1))
y_pred=mtf.predict(x_test)
print("Model3 - Random Forest Accuracy:{0}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix \n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))