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

start = 1978
i=39
data_now=data.loc[(data['Season Start'] >= start)&(data['Season Start']<=start+i-1 )]
x_train = data_now.drop(['Season Start','Label'],axis=1).as_matrix()
y_train = data_now.Label.as_matrix()
data_test=df2.loc[(df2['Season Start']==start+i)]
x_test = data_test.drop(['Season Start','Label'],axis=1).as_matrix()
y_test = data_test.Label.as_matrix()

from sklearn.svm import SVC
i=-8
k=np.linspace(10**(-8), 10**(-6), num=20)#define the range and step length of the search over gamma value. This range is hand-tested to be where SVM won't crash.
result_list_1 = []
result_list_2 = []
from sklearn.metrics import classification_report, confusion_matrix
for i in range(20):
    j=k[i]
    classifier = SVC(kernel = 'poly',C=1,gamma = j)
    classifier.fit(x_train,y_train)
    y_pred=classifier.predict(x_test)
    m = confusion_matrix(y_test,y_pred)
    if m[0][1]+m[1][1] == 0:
        result_list_1.append(0)
        result_list_2.append(0)
    else:
        result_list_1.append(m[1][1]/(m[0][1]+m[1][1]))
        result_list_2.append(m[1][1]/(m[1][0]+m[1][1]))
print(classification_report(y_test,y_pred))
f1 = ['0']*20
for i in range(0,19):
    f1[i+1] = 2*result_list_1[i+1]*result_list_2[i+1]/(result_list_1[i+1]+result_list_2[i+1]+0.00001)
print (f1)
print (k)
f1[0] = f1[1]
plt.plot(k,f1)
plt.show()