import probability as prob
import pandas as pd
import numpy as np
from get_data import get_df
    
class prob_matrix:
    # dosya okuma
    df=pd.read_csv('1.csv')
    df2=pd.read_csv('2.csv')
    df3=pd.read_csv('3.csv')
    df4=pd.read_csv('4.csv')
    df5=pd.read_csv('5.csv')

    df=prob.make_list_target(df)
    df2=prob.make_list_target(df2)
    df3=prob.make_list_target(df3)
    df4=prob.make_list_target(df4)
    df5=prob.make_list_target(df5)
    
    result=get_df()
    result.to_csv('result.csv',index=False)
    unique_activities=result['activities'].unique()
    unique_activities=np.sort(unique_activities)
     
    x1=prob.calculate_next('activity_01',df)*5
    x2=prob.calculate_next('activity_02',df)*5
    x3=prob.calculate_next('activity_03',df)*5
    x4=prob.calculate_next('activity_04',df)*5
    x5=prob.calculate_next('activity_05',df)*5
    x7=prob.calculate_next('activity_07',df)*5
    print(len(x1),len(x2),len(x3),len(x4),len(x5),len(x7))

    # activity_01 den sonra activity_02 veya activity_03 gelme olasılığı
    x1_2,x1_3=prob.calculate_prob('activity_02',x1)
    print('activity_01 den sonra activity_02 gelme olasılığı: ',x1_2)
    print('activity_01 den sonra activity_03 gelme olasılığı: ',x1_3)

    # activity_02 den sonra activity_03 veya activity_04 gelme olasılığı
    x2_3,x2_4=prob.calculate_prob('activity_03',x2)
    print('activity_02 den sonra activity_03 gelme olasılığı: ',x2_3)
    print('activity_02 den sonra activity_04 gelme olasılığı: ',x2_4)

    # activity_03 den sonra activity_01,activity_04,activity_05,activity_07 gelme olasılığı
    x3_1,x3_4,x3_5,x3_7=prob.calculate_prob_activity3('activity_03', x3)

    # activity_04 den sonra activity_01 gelme olasılığı
    x4_1=prob.calculate_prob_activity4_5('activity_04')

    # activity_05 den sonra activity_02 gelme olasılığı
    x5_2=prob.calculate_prob_activity4_5('activity_05')

    index_and_col=['Activity_01','Activity_02','Activity_03','Activity_04','Activity_05','Activity_07']
    prob_df=pd.DataFrame([(0,x1_2,x1_3,0,0,0),(0,0,x2_3,x2_4,0,0),\
                      (x3_1,0,0,x3_4,x3_5,x3_7),\
                      (x4_1,0,0,0,0,0),(0,x5_2,0,0,0,0),(0,0,0,0,0,0)],
                      index=index_and_col,columns=index_and_col)
    print(prob_df)