def calculate_next(activity,df):
    x=[]
    for i in range(len(df['trace_id'])):
        for j in range(len(df[df['trace_id']==i])):
            if (df[df['trace_id']==i].iloc[j][1] == activity) & (len(df[df['trace_id']==i])==4) & (j!=3):
                x.append(df[df['trace_id']==i].iloc[j+1][1])
            elif (df[df['trace_id']==i].iloc[j][1] == activity) & (len(df[df['trace_id']==i])==5) & (j!=4):
                x.append(df[df['trace_id']==i].iloc[j+1][1])            
    return x

def calculate_prob(activity,x):
    x1_act=[]
    for i in x:
        if i == activity:
            x1_act.append(i)
    x1_prob_act2=len(x1_act)/len(x)
    x1_prob_act3=1-x1_prob_act2
    return x1_prob_act2,x1_prob_act3

# activity_03 den sonra activity_07,activity_04,activity_01,activity_05 gelme olasılığı
def calculate_prob_activity3(activity,x3):
    x3_act7=[]
    x3_act4=[]
    x3_act1=[]
    x3_act5=[]
    for i in x3:
        if i == 'activity_07':
            x3_act7.append(i)
        elif i == 'activity_04':
            x3_act4.append(i)
        elif i == 'activity_01':
            x3_act1.append(i)
        elif i == 'activity_05':
            x3_act5.append(i)
    x3_prob_act7=len(x3_act7)/len(x3)
    x3_prob_act4=len(x3_act4)/len(x3)
    x3_prob_act1=len(x3_act1)/len(x3)
    x3_prob_act5=len(x3_act5)/len(x3)
    return x3_prob_act1,x3_prob_act4,x3_prob_act5,x3_prob_act7
    
def calculate_prob_activity4_5(activity):
    return 1.0

# clean data
def explode(df):
    df=df.explode('g')
    df['activities']=df['g']
    df.drop(columns=['g'],axis=1,inplace=True)
    return df

# unique trace_id
def duplicates(df):
    df = df.drop_duplicates(subset = ["trace_id"]).reset_index(drop=True)
    return df

# make a list target column
def make_list_target(df):
    df['g']=df['g'].apply(lambda x: x.split(","))
    df=duplicates(df)
    df=explode(df)
    return df
