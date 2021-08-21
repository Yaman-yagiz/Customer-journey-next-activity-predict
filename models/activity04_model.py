import pandas as pd
import numpy as np
from get_data import get_df
from sklearn import preprocessing

result=get_df()
# aktivite 4'ten sonra sadece activity_01 ' e gidiyor. Ama activity_04 ün son yolculuk olmasıda hesaba alınıp
# bu yüzden kendi üzerine tekrar dönmesi burada son yolculuk olduğu anlamına gelmelidir.
activities=['activity_01','activity_04']
act4=result[result['activities']=='activity_04']

act_arr=[]
act_arr=np.array(act_arr)

# activity_04 den sonra gelen aktivitiler alınıyor. kendisi de dahil.
count=0
for i in act4.iloc:
    if i[3]=='40-59yo' and i[4]=='middle' and i[2]=='yes' and count%2==0:
        act_arr=np.append(act_arr,activities[0])
    elif i[3]=='40-59yo' and i[4]=='middle' and i[2]=='yes' and count%2!=0:
        act_arr=np.append(act_arr,activities[1])
    elif i[3]=='60-79yo' and i[4]=='high' and i[2]=='no':
        act_arr=np.append(act_arr,activities[0])
    else:
        act_arr=np.append(act_arr,activities[0])
    count+=1
    
act4['target']=act_arr
act4 = act4.sample(frac=1).reset_index(drop=True)
act4.drop(columns=['trace_id','activities'],inplace=True)

# veri ön işleme
act4['target']=np.where(act4['target']=='activity_04',0,1)
target=act4.iloc[:,3:4].values
df_target=pd.DataFrame(target,columns=['target'])

le=preprocessing.LabelEncoder()
age=act4.iloc[:,1:2].values
ohe=preprocessing.OneHotEncoder()
age=ohe.fit_transform(age).toarray()
df_age=pd.DataFrame(age,columns=['0-19yo','20-39yo','40-59yo','60-79yo','80yo+'],dtype='int32')

income=act4.iloc[:,2:3].values
ohe=preprocessing.OneHotEncoder()
income=ohe.fit_transform(income).toarray()
df_income=pd.DataFrame(income,columns=['high','middle','low'],dtype='int32')

employed=act4.iloc[:,0:1].values
employed[:,0]=le.fit_transform(act4.iloc[:,0])
df_employed=pd.DataFrame(employed,columns=['employed'],dtype='int32')

frames=[df_employed,df_age,df_income,df_target]
result_act4= pd.concat(frames,axis=1)
result_act4.to_csv('4s.csv',index=False)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
X=result_act4.iloc[:,0:9]
Y=result_act4.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)

print("Model4 - Random Forest(entropy) Accuracy:{0}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix \n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))