import probability as prob
import pandas as pd

def get_df():
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

    data_frames=[df,df2,df3,df4,df5]
    result=pd.concat(data_frames)
    return result