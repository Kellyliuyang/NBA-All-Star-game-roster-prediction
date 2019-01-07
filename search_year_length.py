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

final_score_pre = [0]*37
final_score_rec = [0]*37
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
#loop over all possible selection of years
for i in range(2,39):
    start = 1978.0
    totalround = 39-i+1
    score_pre = 0
    score_rec = 0
    for j in range(1,totalround+1):
        data_now=data.loc[(data['Season Start'] >= start)&(data['Season Start']<=start+i-1 )]
        x = data_now.drop(['Season Start','Label'],axis=1).as_matrix()
        y = data_now.Label.as_matrix()
        data_test=df2.loc[(df2['Season Start']==start+i)]
        x_test_t = data_test.drop(['Season Start','Label'],axis=1).as_matrix()
        y_test_t = data_test.Label.as_matrix()
        classifier = LogisticRegression(C=1)
        classifier.fit(x,y)
        y_pred=classifier.predict(x_test_t)
        k=confusion_matrix(y_test_t,y_pred)
        score_pre = score_pre+k[1][1]/(k[0][1]+k[1][1])
        score_rec = score_rec+k[1][1]/(k[1][0]+k[1][1])
        start = start+1
    final_score_pre[i-2] = score_pre/totalround#Average Precision
    final_score_rec[i-2] = score_rec/totalround#Average Recall
print (final_score_pre)
print (final_score_rec)
f1_log = [0]*37
for i in range(0,37):
    f1_log[i] = 2*final_score_pre[i]*final_score_rec[i]/(final_score_pre[i]+final_score_rec[i]+0.000001) #Calculate F1
plt.xlabel('Year Length')
plt.ylabel('F1 Score')
plt.title('Search Over the Best Year Length')
plt.plot(f1_log)
plt.show()
print (f1_log)