import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

#read in data, add label, cut out from year 1977,
df=pd.read_excel('NBA_Basic.xlsx')
allstar=pd.read_excel('Allstar.xlsx')
TM=pd.read_excel('TM.xlsx')
allstar2=allstar.copy()
allstar2['Season Start']=allstar2['Season Start']+1
allstar2.columns=['Season Start','Player Name','Is_Allstar_Lastyear']#add a feature that describes whether a player was in all star the previous year

allstar2['Season Start']=allstar2['Season Start'].astype(float)
df['Season Start']=df['Season Start'].astype(float)
allstar['Season Start']=allstar['Season Start'].astype(float)
df2=df.merge(allstar, how='left', on=['Season Start','Player Name'])#add label
df2=df2.merge(allstar2, how='left', on=['Season Start','Player Name'])
df2.Label=df2.Label.fillna(0)
df2=df2.drop(['#','blanl','blank2','Player Salary in $'],axis=1)#drop the blank/nan columns
df2=df2.loc[(df2['Season Start'] >= 1978.0) ]#cut from year 1978
df2=df2.fillna(0)
df2['RB']=df2['TRB']/df2['G']#replace total score with average score
df2['APTS']=df2['PTS']/df2['G']#replace total rebound with average rebound
df2 = df2.drop(['Tm','PTS','TRB','Player Name','Pos','WS/48'],axis=1)#drop categorical term, drop redundant term
data=df2

#separate out test and train data: year 1978-year 2016 will be the train data, year 2017 will be the test data.
start = 1978
i=39
data_now=data.loc[(data['Season Start'] >= start)&(data['Season Start']<=start+i-1 )]
x_train = data_now.drop(['Season Start','Label'],axis=1).as_matrix()
y_train = data_now.Label.as_matrix()
data_test=df2.loc[(df2['Season Start']==start+i)]
x_test = data_test.drop(['Season Start','Label'],axis=1).as_matrix()
y_test = data_test.Label.as_matrix()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
start = 50#starting tree number
result_list_1 = []
result_list_2 = []
for i in range(0,50):
    RF = RandomForestClassifier(n_estimators=50+10*i, random_state=0)#search over all possible tree numbers, with an increment of 10
    RF.fit(x_train,y_train)
    y_pred_RF=RF.predict(x_test)
    y_pred_RF2=RF.predict(x_train)
    m = confusion_matrix(y_test,y_pred_RF)
    if m[0][1]+m[1][1] == 0:
        result_list_1.append(0)
        result_list_2.append(0)
    else:
        result_list_1.append(m[1][1]/(m[0][1]+m[1][1]))
        result_list_2.append(m[1][1]/(m[1][0]+m[1][1]))
f1 = ['0']*50
for i in range(0,50):
    f1[i] = 2*result_list_1[i]*result_list_2[i]/(result_list_1[i]+result_list_2[i]+0.00001)
plt.plot(f1)
plt.show()