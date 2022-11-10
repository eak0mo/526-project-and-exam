#%%

#import libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.decomposition import PCA


#%%

#data preprocessing, cleaning and a little sorting
health_data_raw=pd.read_stata('Data Sets_Other/sect3_health.dta')
health_data=pd.read_stata('Data Sets_Other/sect3_health.dta',convert_categoricals=False,preserve_dtypes=False,convert_missing=False)
health_data=health_data.fillna(0)
health_data
#health_data.describe()
#health_data_raw
#health_data_raw.describe()
#sea.pairplot(health_data)

#%%

health_data.columns

#%%

#data for the south west for past 30 days
southwest_total_data=health_data.loc[health_data['zone']==6,['state','sector','s03q03','s03q04_1','s03q04_2',
       's03q05','s03q06_1','s03q06_2','s03q07a','s03q08','s03q09', 's03q10_1',
       's03q10_2', 's03q11__1', 's03q11__2', 's03q11__3',
       's03q11__4', 's03q11__5', 's03q12', 's03q13',
       's03q14', 's03q15', 's03q16a', 's03q16b', 's03q17', 's03q18', 's03q18b']]
#state_zone=health_data.loc[health_data['zone']<=6,['state','zone']]
#state_zone=state_zone.loc[state_zone['state']==24,:]
#sea.pairplot(southwest_total_data)
southwest_total_data

#%%

#data plotting
sw_plot_1=southwest_total_data.loc[:,['state','sector','s03q03','s03q04_1','s03q04_2']]
sea.pairplot(sw_plot_1)


#%%

#data plotting
sw_plot_2=southwest_total_data.loc[:,['s03q05','s03q06_1','s03q06_2','s03q07a','s03q08']]
sea.pairplot(sw_plot_2)

#%%

#data plotting
sw_plot_3=southwest_total_data.loc[:,['s03q09', 's03q10_1','s03q10_2', 's03q11__1', 's03q11__2']]
sea.pairplot(sw_plot_3)

#%%

#data plotting
sw_plot_4=southwest_total_data.loc[:,['s03q11__3','s03q11__4', 's03q11__5', 's03q12', 's03q13']]
sea.pairplot(sw_plot_4)

#%%

#data plotting
sw_plot_5=southwest_total_data.loc[:,['s03q14', 's03q15', 's03q16a', 's03q16b', 's03q17']]
sea.pairplot(sw_plot_5)

#%%

#data plotting
sw_plot_6=southwest_total_data.loc[:,[ 's03q18', 's03q18b']]
plot6=sea.pairplot(sw_plot_6)


#%%

#test for colinearity if needed (vif)
vifdatax=health_data.loc[health_data['zone']==6,['zone','state','sector','s03q04_1','s03q04_2',
       's03q05','s03q06_1','s03q06_2','s03q07a','s03q08','s03q09', 's03q10_1',
       's03q10_2', 's03q11__1', 's03q11__2', 's03q11__3',
       's03q11__4', 's03q11__5', 's03q12', 's03q13',
       's03q14', 's03q15', 's03q16a', 's03q16b', 's03q17', 's03q18', 's03q18b']]


vif_data=pd.DataFrame()
vif_data['feature']=vifdatax.columns

vif_data['VIF']=[vif(vifdatax.values,i) for i in range(len(vifdatax.columns))]
print(vif_data)

#%%

#dropping zone
vifdatax2=health_data.loc[health_data['zone']==6,['state','sector','s03q04_1','s03q04_2',
       's03q05','s03q06_1','s03q06_2','s03q07a','s03q08','s03q09', 's03q10_1',
       's03q10_2', 's03q11__1', 's03q11__2', 's03q11__3',
       's03q11__4', 's03q11__5', 's03q12', 's03q13',
       's03q14', 's03q15', 's03q16a', 's03q16b', 's03q17', 's03q18', 's03q18b']]

vif2=pd.DataFrame()
vif2['features']=vifdatax2.columns

vif2['vif without zone']=[vif(vifdatax2.values,i) for i in range(len(vifdatax2.columns))]
print(vif2)

#%%

#dropping s03q16b
vifdatax3=health_data.loc[health_data['zone']==6,['state','sector','s03q04_1','s03q04_2',
       's03q05','s03q06_1','s03q06_2','s03q07a','s03q08','s03q09', 's03q10_1',
       's03q10_2', 's03q11__1', 's03q11__2', 's03q11__3',
       's03q11__4', 's03q11__5', 's03q12', 's03q13',
       's03q14', 's03q15', 's03q16a', 's03q17', 's03q18', 's03q18b']]

vif3=pd.DataFrame()
vif3['features']=vifdatax3.columns

vif3['vif without zone']=[vif(vifdatax3.values,i) for i in range(len(vifdatax3.columns))]
print(vif3)

#%%

#dropping s03q05
vifdatax4=health_data.loc[health_data['zone']==6,['state','sector','s03q04_1','s03q04_2',
       's03q06_1','s03q06_2','s03q07a','s03q08','s03q09', 's03q10_1',
       's03q10_2', 's03q11__1', 's03q11__2', 's03q11__3',
       's03q11__4', 's03q11__5', 's03q12', 's03q13',
       's03q14', 's03q15', 's03q16a', 's03q17', 's03q18', 's03q18b']]

vif4=pd.DataFrame()
vif4['features']=vifdatax4.columns

vif4['vif without zone']=[vif(vifdatax4.values,i) for i in range(len(vifdatax4.columns))]
print(vif4)

#%%

#dropping s03q12
vifdatax5=health_data.loc[health_data['zone']==6,['state','sector','s03q04_1','s03q04_2',
       's03q06_1','s03q06_2','s03q07a','s03q08','s03q09', 's03q10_1',
       's03q10_2', 's03q11__1', 's03q11__2', 's03q11__3',
       's03q11__4', 's03q11__5', 's03q13',
       's03q14', 's03q15', 's03q16a', 's03q17', 's03q18', 's03q18b']]

vif5=pd.DataFrame()
vif5['features']=vifdatax5.columns

vif5['vif without zone']=[vif(vifdatax5.values,i) for i in range(len(vifdatax5.columns))]
print(vif5)

#%%

#changing the values of 1 and 2 to yes and no respectively.
val2str=southwest_total_data.loc[:,'s03q03']

for i in southwest_total_data.index:
    if val2str[i]==1.0:
        southwest_total_data['s03q03'][i]='YES'
    else:
        southwest_total_data['s03q03'][i]='NO'



southwest_total_data.head(30)

#%%

#split data using 80/20 method and train and test
southwest_train, southwest_test= split(southwest_total_data,test_size=0.2)
southwest_train
southwest_test

#%%

#logisitc regression
'''
visit_doc=health_data.loc[health_data['zone']==6,['state','sector','s03q03','s03q04_1','s03q04_2']]
val2str=visit_doc.loc[:,'s03q03']
for i in visit_doc.index:
    if val2str[i]==1.0:
        visit_doc['s03q03'][i]='YES'
    else:
        visit_doc['s03q03'][i]='NO'
        '''
visit_doc_model = LogisticRegression()
swtrainx= southwest_train.loc[:, ['sector', 's03q04_1', 's03q04_2',
       's03q05','s03q06_1','s03q06_2','s03q07a','s03q08','s03q09', 's03q10_1',
       's03q10_2', 's03q11__1', 's03q11__2', 's03q11__3',
       's03q11__4', 's03q11__5', 's03q12', 's03q13',
       's03q14', 's03q15', 's03q16a', 's03q16b', 's03q17', 's03q18', 's03q18b']]
swtrainy= southwest_train.loc[:, ['s03q03']]
visit_doc_model.fit(swtrainx, swtrainy)
print(visit_doc_model.coef_)
#visit_doc_model.n_features_in_
#visit_doc_model


#%%

#predition and summary
swtesttrue= southwest_test.loc[:, ['s03q03']]
swtestx= southwest_test.loc[:, ['sector', 's03q04_1', 's03q04_2',
       's03q05','s03q06_1','s03q06_2','s03q07a','s03q08','s03q09', 's03q10_1',
       's03q10_2', 's03q11__1', 's03q11__2', 's03q11__3',
       's03q11__4', 's03q11__5', 's03q12', 's03q13',
       's03q14', 's03q15', 's03q16a', 's03q16b', 's03q17', 's03q18', 's03q18b']]
logmodel_pred=visit_doc_model.predict(swtestx)
print(metrics.classification_report(swtesttrue, logmodel_pred))
metrics.confusion_matrix(swtesttrue, logmodel_pred)
#metrics.f1_score(swtesttrue,logmodel_pred)

#%%

#lda
visitdoc_ldamodel=LinearDiscriminantAnalysis()
visitdoc_ldamodel.fit(swtrainx, swtrainy)
visitdoc_ldamodel

#%%

#prediction and summary
ldamodel_pred=visitdoc_ldamodel.predict(swtestx)
print(metrics.classification_report(swtesttrue, ldamodel_pred))
metrics.confusion_matrix(swtesttrue, ldamodel_pred)

#%%

#qda
visitdoc_qdamodel=QuadraticDiscriminantAnalysis()
visitdoc_qdamodel.fit(swtrainx, swtrainy)
visitdoc_qdamodel

#%%

#prediction and summary
qdamodel_pred=visitdoc_qdamodel.predict(swtestx)
print(metrics.classification_report(swtesttrue, qdamodel_pred))
metrics.confusion_matrix(swtesttrue, qdamodel_pred)

#%%

#sorting data x and y
kcvx=southwest_total_data.loc[:,['sector','s03q04_1','s03q04_2',
       's03q05','s03q06_1','s03q06_2','s03q07a','s03q08','s03q09', 's03q10_1',
       's03q10_2', 's03q11__1', 's03q11__2', 's03q11__3',
       's03q11__4', 's03q11__5', 's03q12', 's03q13',
       's03q14', 's03q15', 's03q16a', 's03q16b', 's03q17', 's03q18', 's03q18b']]

kcvy=southwest_total_data.loc[:,['s03q03']]
kcvymae=health_data.loc[:,['s03q03']]
#%%

#cross validation with 7 folds for log regression
k=7
folds=KFold(n_splits=k)
accuracy=[]
metrep=[]

for train_index,test_index in folds.split(kcvx):
    #x_train=[],x_test=[],y_train=[],y_test=[]
    x_train,x_test= kcvx.iloc[train_index,:],kcvx.iloc[test_index,:]
    y_train,y_test= kcvy.iloc[train_index,:],kcvy.iloc[test_index,]

    #print(train_index,test_index)
    visit_doc_model.fit(x_train,y_train)
    y_pred=visit_doc_model.predict(x_test)
    metrep.append(metrics.classification_report(y_test,y_pred))
    accuracy.append(metrics.accuracy_score(y_test,y_pred))
    #mae.append(metrics.mean_absolute_error(y_test,y_pred))

#print(accuracy)
#%%
#to collect mae data

#cross validation with 7 folds for log regression
k=7
folds=KFold(n_splits=k)


mae=[]
for train_index,test_index in folds.split(kcvx):
    #x_train=[],x_test=[],y_train=[],y_test=[]
    x_train,x_test= kcvx.iloc[train_index,:],kcvx.iloc[test_index,:]
    y_train,y_test= kcvymae.iloc[train_index,:],kcvymae.iloc[test_index,]

    #print(train_index,test_index)
    visit_doc_model.fit(x_train,y_train)
    y_pred=visit_doc_model.predict(x_test)
    #metrep.append(metrics.classification_report(y_test,y_pred))
    #accuracy.append(metrics.accuracy_score(y_test,y_pred))
    mae.append(metrics.mean_absolute_error(y_test,y_pred))

#%%
print(metrep[0])
print(mae[0])
#%%

plt.plot(accuracy)
plt.plot(mae)


#%%

#pca method on single 80/20 data used originally using 85% variance retained
pca=PCA(0.85)
pca.fit(swtrainx)
#pca.n_components_
pcatrainx=pca.transform(swtrainx)
pcatestx=pca.transform(swtestx)
pcatrainx
#pcatestx
pca.explained_variance_ratio_


#%%

#pca method on single 80/20 data used originally using 95% variance retained
pca=PCA(0.90)
pca.fit(swtrainx)
#pca.n_components_
pcatrainx=pca.transform(swtrainx)
pcatestx=pca.transform(swtestx)
pcatrainx
#pcatestx
pca.explained_variance_ratio_

#%%

#pca method on single 80/20 data used originally using 95% variance retained
pca=PCA(0.95)
pca.fit(swtrainx)
#pca.n_components_
pcatrainx=pca.transform(swtrainx)
pcatestx=pca.transform(swtestx)
pcatrainx
#pcatestx
pca.explained_variance_ratio_
#%%

#pca method on single 80/20 data used originally using 95% variance retained
pca=PCA(0.97)
pca.fit(swtrainx)
#pca.n_components_
pcatrainx=pca.transform(swtrainx)
pcatestx=pca.transform(swtestx)
pcatrainx
#pcatestx
pca.explained_variance_ratio_

#%%

#pca method on single 80/20 data used originally using 95% variance retained
pca=PCA(0.99)
pca.fit(swtrainx)
#pca.n_components_
pcatrainx=pca.transform(swtrainx)
pcatestx=pca.transform(swtestx)
pcatrainx
pca.n_components_
pca.explained_variance_ratio_
#pcatestx

#%%

#pca method on single 80/20 data used originally using 95% variance retained
pca=PCA(n_components=5)
pca.fit(swtrainx)
#pca.n_components_
pcatrainx=pca.transform(swtrainx)
pcatestx=pca.transform(swtestx)
pcatrainx
pca.n_components_
pca.explained_variance_ratio_
#pcatestx

#%%

#pca method on single 80/20 data used originally using 95% variance retained
pca=PCA(n_components=10)
pca.fit(swtrainx)
#pca.n_components_
pcatrainx=pca.fit_transform(swtrainx)
pcatestx=pca.transform(swtestx)
pca.n_components_
pca.explained_variance_ratio_
#pcatestx
