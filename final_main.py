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

#import logistic regression and run it
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression(C=1)
logis.fit(x_train,y_train)
y_logis=logis.predict(x_test)
y_logis2=logis.predict(x_train)
print ('Logistic Regression Results')
print ('Test Results')
print(classification_report(y_test,y_logis))
print ('Train Results')
print(classification_report(y_train,y_logis2))

#Import Support Vector Machine and run it
from sklearn.svm import SVC
svm = SVC(kernel = 'poly',C=1,gamma = 4*10**(-7))#kernel and gamma values are the optimized ones
svm.fit(x_train,y_train)
y_svm=svm.predict(x_test)
y_svm2 = svm.predict(x_train)
print ('SVM Results')
print ('Test Results')
print(classification_report(y_test,y_svm))
print ('Train Results')
print(classification_report(y_train,y_svm2))

#Define a plot function thah returns the feature importance plot as output
def plot_feature_importance(classifier):
    cols    = [i for i in df2.columns if i not in ['Tm','PTS','TRB','Player Name','Season Start','Pos','Tm','Label']]
    cols_df=pd.DataFrame(cols)
    feature_i=pd.DataFrame(classifier.feature_importances_)
    feature_importance=pd.concat([cols_df,feature_i],axis=1)
    feature_importance.columns=['feature','feature importance']
    feature_importance=feature_importance.sort_values(by = 'feature importance',ascending = False)

    y_pos=np.arange(len(feature_importance))
    y_value=feature_importance['feature importance'].values.tolist()
    x_value=feature_importance['feature'].values.tolist()
    plt.figure(figsize=(10, 8))
    plt.bar(y_pos,y_value, align='center', alpha=0.5,color="blue")
    plt.xticks(y_pos, x_value,rotation=90,fontsize=8)
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance')
    plt.show()

#Import Decision Tree and run it
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(random_state=0)
DT.fit(x_train,y_train)
y_pred_DT=DT.predict(x_test)
y_pred_DT2=DT.predict(x_train)
from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test,y_pred_DT))
print ('Decision Tree Results')
print ('Testing Results')
print(classification_report(y_test,y_pred_DT))
print ('Training Results')
print(classification_report(y_train,y_pred_DT2))
plot_feature_importance(DT)

#Important Gradient Boosting and run it
from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(x_train,y_train)
y_pred_GBDT=GBDT.predict(x_test)
y_pred_GBDT2=GBDT.predict(x_train)
from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test,y_pred_GBDT))
print ('Gradient Boosting Results')
print ('Testing Results')
print(classification_report(y_test,y_pred_GBDT))
print ('Training Results')
print(classification_report(y_train,y_pred_GBDT2))
plot_feature_importance(GBDT)

#Import Random Forest and run it
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=300, random_state=0)
RF.fit(x_train,y_train)
y_pred_RF=RF.predict(x_test)
y_pred_RF2=RF.predict(x_train)
from sklearn.metrics import classification_report, confusion_matrix
#(confusion_matrix(y_test,y_pred_RF))
print ('Random Forest Results')
print ('Testing Results')
print(classification_report(y_test,y_pred_RF))
print ('Training Results')
print(classification_report(y_train,y_pred_RF2))
plot_feature_importance(RF)



