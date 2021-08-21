import pandas as pd
import numpy as np
from get_data import get_df
from sklearn import preprocessing

result=get_df()
activities=['activity_03','activity_04']
act2=result[result['activities']=='activity_02']

act_arr=[]
act_arr=np.array(act_arr)

# activity_02 den sonra gelen aktivitiler alınıyor.
for i in act2.iloc:
    if i[3]=='60-79yo' and i[4]=='high' and i[2]=='no':
        act_arr=np.append(act_arr,activities[1])
    else:
        act_arr=np.append(act_arr,activities[0])
        
act2['target']=act_arr
act2=act2.drop(columns=['trace_id','activities'])

# veri ön işleme
act2['target']=np.where(act2['target']=='activity_03',0,1)
target=act2.iloc[:,3:4].values
df_target=pd.DataFrame(target,columns=['target'])

le=preprocessing.LabelEncoder()
age=act2.iloc[:,1:2].values
ohe=preprocessing.OneHotEncoder()
age=ohe.fit_transform(age).toarray()
df_age=pd.DataFrame(age,columns=['0-19yo','20-39yo','40-59yo','60-79yo','80yo+'],dtype='int32')

income=act2.iloc[:,2:3].values
ohe=preprocessing.OneHotEncoder()
income=ohe.fit_transform(income).toarray()
df_income=pd.DataFrame(income,columns=['high','low','middle'],dtype='int32')

employed=act2.iloc[:,0:1].values
employed[:,0]=le.fit_transform(act2.iloc[:,0])
df_employed=pd.DataFrame(employed,columns=['employed'],dtype='int32')

frames=[df_employed,df_age,df_income,df_target]
result_act2= pd.concat(frames,axis=1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
X=result_act2.iloc[:,0:9]
Y=result_act2.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
print("Model2 - Random Forest(entropy) Accuracy:{0}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix \n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))