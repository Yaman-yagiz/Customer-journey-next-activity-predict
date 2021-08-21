import pandas as pd
import numpy as np
from get_data import get_df
from sklearn import preprocessing

# Activitiy_01 den sonra gelebilecek aktiviteler.
result=get_df()
activities=['activity_02','activity_03']
act1=result[result['activities']=='activity_01']

act_arr=[]
act_arr=np.array(act_arr)

# activity_01 den sonra gelen aktivitiler alınıyor.
for i in act1.iloc:
    if i[3]=='80yo+' and i[4]=='middle' and i[2]=='no':
        act_arr=np.append(act_arr,activities[1])
    else:
        act_arr=np.append(act_arr,activities[0])
        
act1['target']=act_arr
act1=act1.drop(columns=['trace_id','activities'])

# veri ön işleme
act1['target']=np.where(act1['target']=='activity_02',0,1)
target=act1.iloc[:,3:4].values
df_target=pd.DataFrame(target,columns=['target'])
le=preprocessing.LabelEncoder()

age=act1.iloc[:,1:2].values
ohe=preprocessing.OneHotEncoder()
age=ohe.fit_transform(age).toarray()
df_age=pd.DataFrame(age,columns=['0-19yo','20-39yo','40-59yo','60-79yo','80yo+'],dtype='int32')

income=act1.iloc[:,2:3].values
ohe=preprocessing.OneHotEncoder()
income=ohe.fit_transform(income).toarray()
df_income=pd.DataFrame(income,columns=['high','low','middle'],dtype='int32')

employed=act1.iloc[:,0:1].values
employed[:,0]=le.fit_transform(act1.iloc[:,0])
df_employed=pd.DataFrame(employed,columns=['employed'],dtype='int32')

frames=[df_employed,df_age,df_income,df_target]
act1_result = pd.concat(frames,axis=1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
X=act1_result.iloc[:,0:9]
Y=act1_result.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
print("Model1 - Random Forest(entropy) Accuracy:{0}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix \n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))