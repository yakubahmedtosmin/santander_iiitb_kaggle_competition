#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing, ensemble
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier

###################################################################
feat_col = ["fecha_dato","ind_empleado","pais_residencia","sexo","age", "ind_nuevo", "antiguedad", "nomprov", 
            "cod_prov", "renta","indrel","segmento","ncodpers"]
target_col = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',
              'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
                'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
              'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
              'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
              'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
              'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1',
              'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
##################################################################
train_1 = pd.read_csv("../input/santander-pr/train.csv", usecols=feat_col)
test = pd.read_csv("../input/santander-pr/test.csv", usecols=feat_col)

##################################################################
import gc
gc.collect()
##################################################################
train.fillna(0,inplace=True)
train["renta"]   = pd.to_numeric(train["renta"], errors="coerce")
test["renta"]   = pd.to_numeric(test["renta"], errors="coerce")
test.fillna(0,inplace=True)
##################################################################
train["age"]   = pd.to_numeric(train["age"], errors="coerce")
test["age"]   = pd.to_numeric(test["age"], errors="coerce")
train['antiguedad'].replace('     NA',np.nan,inplace=True)
train['antiguedad'].replace('-999999',np.nan,inplace=True)
train['antiguedad'].replace(-999999,np.nan,inplace=True)
train.antiguedad = pd.to_numeric(train.antiguedad, errors='coerce')
train.antiguedad = train.antiguedad.astype('float16')

test['antiguedad'].replace('     NA',np.nan,inplace=True)
test['antiguedad'].replace('-999999',np.nan,inplace=True)
test['antiguedad'].replace(-999999,np.nan,inplace=True)
test.antiguedad = pd.to_numeric(test.antiguedad, errors='coerce')
test.antiguedad = test.antiguedad.astype('float16')
train.cod_prov = pd.to_numeric(train.cod_prov, errors='coerce')
train.cod_prov = train.cod_prov.astype('float16')

test.cod_prov = pd.to_numeric(test.cod_prov, errors='coerce')
test.cod_prov = test.cod_prov.astype('float16')
train.fillna(0,inplace=True)
test.fillna(0,inplace=True)

##############################################################

lisobj = []
for ind, col in enumerate(train.columns):
    if train[col].dtype == "object":
        lisobj.append(col)
##############################################################

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import make_column_transformer
le = LabelEncoder()
for i in lisobj:
    train[i]=train[i].astype('str')
    train[i]=le.fit_transform(train[i])

for i in lisobj:
    test[i]=test[i].astype('str')
    test[i]=le.fit_transform(test[i])
    
train_Y = pd.read_csv("../input/santander-pr/train.csv", usecols=['ncodpers','fecha_dato']+target_col)

#############################################################

train_Y=train_Y[train_Y['fecha_dato'] == '2015-05-28']
train_Y.drop(columns='fecha_dato',inplace=True)
#############################################################

tar=[  'ind_ahor_fin_ult1', 'ind_aval_fin_ult1',
       'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
       'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
       'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1',
       'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',
       'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1',
       'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
       'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1',
       'ind_recibo_ult1']
#############################################################
                ############ MODEL ##########
prob=[]
for i in tar:
    clf = xgb.XGBClassifier(max_depth=12, learning_rate = 0.05, 
                 subsample = 0.7, colsample_bytree = 0.7, n_estimators=70,
                 colsample_bylevel= 0.7, min_child_weight = 5,
               nthread=4)
    clf.fit(train,train_Y[i])
    y=clf.predict(test)
    prob.append(clf.predict_proba(test)[:,1])
    clf=0
    undersampled_data = 0
    result = 0
    gc.collect()
preds = np.array(prob)[:,:].T

#############################################################

train_Y = pd.read_csv("../input/santander-pr/train.csv", usecols=['ncodpers','fecha_dato']+target_col)
train_Y=train_Y[train_Y['fecha_dato'] == '2016-04-28']
train_Y.drop(columns='fecha_dato',inplace=True)

#############################################################

                ##########PREDICTION###########

last_instance_df =  train_Y
cust_dict = {}
cust_undict= {}
target_cols = np.array(target_col)
for ind, row in last_instance_df.iterrows():
    cust = row['ncodpers']
    used_products = set(target_cols[np.array(row[1:])==1])
    unused_products = set(target_cols[np.array(row[1:])==0])
    cust_dict[cust] = used_products
    cust_undict[cust] = unused_products
del (last_instance_df)

final_1 = np.argsort(preds, axis=1)
final_2 = np.fliplr(final_1)
test_id = np.array(pd.read_csv("../input/santander-pr/test.csv", usecols=['ncodpers'])['ncodpers'])
final_preds = []
for ind, pred in enumerate(final_2):
    temp_dict={}
    #print("--------------------------------------")
    cust = test_id[ind]
    top_products = target_cols[pred]
    drop_products = target_cols[pred[::-1]]
    used_products = cust_dict.get(cust,[])
    unused_products = cust_undict.get(cust,[])
    new_top_products = []
    new_drop_products = []
    product_col = []
    pred_col = []
    use_unuse = []
    diffrence = []
    rank = []
    # customers after index 928274 are new and 
    #thus do not have previous month data to compare the probablities
    # to compare to the target variables of the previous month
    # thus we process their probabilities differently
    if ind >= 928274:                   
        counter = 0
        for product in drop_products: 
            if product not in unused_products:
                diffrence.append(1 - preds[ind][pred[::-1][counter]])
                new_top_products.append(product)
            counter += 1
        df = pd.DataFrame()
        df['col_1'] = new_top_products
        df['col_2'] = diffrence
        new_top_products = df.sort_values('col_2', ascending = False).col_1.head().to_list()
    #
    else:
        #we separate the target features into two groups 
        #based on their last month data being ‘0’ or ‘1’ 
        counter = 0
        for product in drop_products:
            if product not in unused_products:
                #for all the target features segmented into the ‘1’ category the resultant 
                #probability (of being ‘1’) of that feature given by 
                #model is subtracted to 1 (giving probability of being ‘0’ i.e. dropping product). 
                diffrence.append(1 - preds[ind][pred[::-1][counter]])
                new_top_products.append(product)
            counter += 1
        counter = 0
        for product in top_products: 
            if product not in used_products:
                diffrence.append(preds[ind][pred[counter]])
                new_top_products.append(product)
            counter += 1
        #all the probabilities are compared and the top 5 features with highest probability
        #(of changing) are given as output prediction for that customer. 

        df = pd.DataFrame()
        df['col_1'] = new_top_products  
        df['col_2'] = diffrence
        new_top_products = df.sort_values('col_2', ascending = False).col_1.head().to_list()    
    final_preds.append(" ".join(new_top_products))
out_df = pd.DataFrame({'ncodpers':test_id, 'changed':final_preds})
out_df.to_csv('sub_prev_m4.csv', index=False)



