import pandas as pd
import numpy as np
from get_data import get_df
from sklearn import preprocessing

result=get_df()
# aktivite 5'ten sonra sadece activity_02 ' e gidiyor. Ama activity_05 ün son yolculuk olmasıda hesaba alınıp
# bu yüzden kendi üzerine tekrar dönmesi burada son yolculuk olduğu anlamına gelmelidir.
activities5=['activity_02','activity_05']
act5=result[result['activities']=='activity_05']

act_arr=[]
act_arr=np.array(act_arr)

# activity_05 den sonra gelen aktivitiler alınıyor. kendisi de dahil.
for i in act5.iloc:
    if i[3]=='60-79yo' and i[4]=='high' and i[2]=='no':
        act_arr=np.append(act_arr,activities5[0])
    else:
        act_arr=np.append(act_arr,activities5[1])
        
act5['target']=act_arr
act5 = act5.sample(frac=1).reset_index(drop=True)
act5.drop(columns=['trace_id','activities'],inplace=True)

# veri ön işleme
act5['target']=np.where(act5['target']=='activity_05',0,1)
target=act5.iloc[:,3:4].values
df_target=pd.DataFrame(target,columns=['target'])

le=preprocessing.LabelEncoder()
age=act5.iloc[:,1:2].values
ohe=preprocessing.OneHotEncoder()
age=ohe.fit_transform(age).toarray()
df_age=pd.DataFrame(age,columns=['0-19yo','20-39yo','40-59yo','60-79yo','80yo+'],dtype='int32')

income=act5.iloc[:,2:3].values
ohe=preprocessing.OneHotEncoder()
income=ohe.fit_transform(income).toarray()
df_income=pd.DataFrame(income,columns=['high','middle','low'],dtype='int32')

employed=act5.iloc[:,0:1].values
employed[:,0]=le.fit_transform(act5.iloc[:,0])
df_employed=pd.DataFrame(employed,columns=['employed'],dtype='int32')

frames=[df_employed,df_age,df_income,df_target]
result_act5= pd.concat(frames,axis=1)
result_act5.to_csv('5s.csv',index=False)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
X=result_act5.iloc[:,0:9]
Y=result_act5.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)

print("Model5 - Random Forest(entropy) Accuracy:{0}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix \n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))