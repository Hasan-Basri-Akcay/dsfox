"""
Feature
"""

# Author: Hasan Basri Akçay <https://www.linkedin.com/in/hasan-basri-akcay/>

import pandas as pd
import numpy as np
import scipy.stats as ss

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.base import clone

from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor

import gc

def feature_importance(data=None, num_features=[], cat_features=[], target='target', 
                       nfold=10, score='roc_auc', sample=False, sample_num=500000, 
                       features_group=None, random_state=0, n_repeats=10, task='clf', 
                       method='all'):
    def create_corr_df(data, features, target='target'):
        corr = data[features].corrwith(data[target])
        corr_df = pd.DataFrame(corr, columns=['Corr'], index=features)
        return corr_df

    def create_corr_object_df(data, features, target='target'):
        def cramers_corrected_stat(confusion_matrix):
            """ calculate Cramers V statistic for categorial-categorial association.
                uses correction from Bergsma and Wicher, 
                Journal of the Korean Statistical Society 42 (2013): 323-328
            """
            chi2 = ss.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum()
            phi2 = chi2/n
            r,k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
        
            cat_corr_dict = {}
        
            for cat_col in features:
                confusion_matrix = pd.crosstab(data[cat_col], data[target]).to_numpy()
                cr = cramers_corrected_stat(confusion_matrix)
                cat_corr_dict[cat_col] = [cr]
            
            cat_corr_df = pd.DataFrame.from_dict(cat_corr_dict).T
            cat_corr_df.columns = ['chi2']
            cat_corr_df.sort_values('chi2', ascending=False, inplace=True)
            return cat_corr_df
     
    def create_mutual_info_df(data, features, target, sample=False, sample_num=500000, 
                              random_state=1):
        if sample:
            temp_data = data.sample(n=sample_num, random_state=random_state)
            temp_data.reset_index(inplace=True, drop=True)
            mi_scores = mutual_info_classif(temp_data[features], temp_data[target])
            del temp_data
            gc.collect()
        else:
            mi_scores = mutual_info_classif(data[features], data[target])
        
        mi_scores_df = pd.DataFrame(mi_scores, columns=["MI Scores"], index=features)
        mi_scores_df = mi_scores_df.sort_values('MI Scores', ascending=False)
        return mi_scores_df
    
    def create_permutation_importance_df(data, features, target, nfold=10, model_ori=None, 
                                         score='roc_auc', sample=False, sample_num=500000, 
                                         random_state=0, n_repeats=30):
        if sample:
            temp_data = data.sample(n=sample_num, random_state=random_state)
            temp_data.reset_index(inplace=True, drop=True)
            X = temp_data[features]
            y = temp_data[[target]]
            del temp_data
            gc.collect()
        else:
            X = data[features]
            y = data[target]
        
        skf = StratifiedKFold(n_splits=nfold)
        zeros = np.zeros((len(features), 2))
        permutation_importance_df = pd.DataFrame(zeros, index=features, columns=['PI mean','PI std'])
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
            y_train, y_test = y.loc[train_index,:], y.loc[test_index,:]
            
            model = clone(model_ori)
            model.fit(X_train, y_train)
            r = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, 
                                       scoring=score)
            permutation_importance_df['PI mean'] += r.importances_mean / nfold
            permutation_importance_df['PI std'] += r.importances_std / nfold
            
            del X_train
            del X_test
            del y_train
            del y_test
            del model
            gc.collect()
        permutation_importance_df.sort_values('PI mean', inplace=True)
        return permutation_importance_df
    
    def feature_importances_multiple(data=None, num_features=[], cat_features=[], target='target', nfold=10, 
                                     score='roc_auc', sample=False, sample_num=500000, 
                        model=None, features_group=None, random_state=0, n_repeats=10, task='clf'):
        if model is None and task == 'clf':
            model = LGBMClassifier(class_weight='balanced')
        elif model is None and task == 'reg':
            model = LGBMRegressor()
        
        enc = OrdinalEncoder()
        data[cat_features] = enc.fit_transform(data[cat_features])
        
        float_corr_df = create_corr_df(data, num_features, target=target)
        cat_corr_df = create_corr_object_df(data, cat_features, target=target)
        mi_scores_df = create_mutual_info_df(data, num_features+cat_features, target, sample=sample, 
                                             sample_num=sample_num, random_state=random_state)
        pi_scores_df = create_permutation_importance_df(data, num_features+cat_features, target, nfold=nfold,
                                                        model_ori=model, score=score, sample=sample, 
                                                        sample_num=sample_num, random_state=random_state, 
                                                        n_repeats=n_repeats)
        
        df_fi_mutiple = float_corr_df.merge(cat_corr_df, left_index=True, right_index=True, how='outer')
        df_fi_mutiple = df_fi_mutiple.merge(mi_scores_df, left_index=True, right_index=True, how='outer')
        df_fi_mutiple = df_fi_mutiple.merge(pi_scores_df, left_index=True, right_index=True, how='outer')
        
        if features_group is not None:
            for key in features_group.keys():
                for index, row in df_fi_mutiple.iterrows():
                    if index in features_group[key]:
                        df_fi_mutiple.loc[index, 'group'] = key
        
        return df_fi_mutiple
    
    if method == 'all':
        return feature_importances_multiple(data=data, num_features=num_features, cat_features=cat_features, 
                                            target=target, nfold=nfold, score=score, sample=sample, 
                                            sample_num=sample_num, model=None, features_group=features_group, 
                                            random_state=random_state, n_repeats=n_repeats, task=task)

def feature_extraction():
    pass

def feature_distribution():
    pass

def feature_statistic():
    pass