#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:19:59 2018

@author: sowjanya
"""

import pandas as pd
import numpy as np
import glob

path ='data' # use your path
allFiles = glob.glob(path + "/*.csv")
loans_df = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,skiprows=1,low_memory=False)
    list_.append(df)
loans_df = pd.concat(list_)

#loans_df = pd.read_csv('data/LendingClub_2007_2011.csv', skiprows=1,low_memory=False)

loans_df.head(5)

loans_df.info()

print(loans_df.isnull().sum().sort_values(ascending=False))


threshold_count = len(loans_df)*0.7 
# Drop any column with more than 30% missing values
filtered_loans_df = loans_df.dropna(thresh=threshold_count,axis=1)

#filtered_loans_df = loans_df.dropna(thresh=threshold_count,axis=1) 
# These columns are not useful for our purposes
#filtered_loans_df = filtered_loans_df.drop(['id' ,'url','desc'],axis=1)
filtered_loans_df.info()
print(filtered_loans_df.isnull().sum().sort_values(ascending=False))


#filtered_loans_df.to_csv('filtered.csv')

#print(filtered_loans_df['pymnt_plan'].value_counts())

#print(filtered_loans_df['hardship_flag'].value_counts())

#print(filtered_loans_df['debt_settlement_flag'].value_counts())

#print(filtered_loans_df['disbursement_method'].value_counts())

#print(filtered_loans_df['application_type'].value_counts())

filtered_loans_df = filtered_loans_df.drop(['id' ,'url','pymnt_plan','hardship_flag','debt_settlement_flag','disbursement_method' ,'application_type'],axis=1)

#filtered_loans_df.info()

#filtered_loans_df.iloc[:, :20].apply(lambda x: x.fillna(x.mean()) if np.issubdtype(x, np.number),axis=0)
#filtered_loans_mean_df = filtered_loans_df.loc[:, (filtered_loans_df.dtypes==np.float64)]
#filtered_loans_mean_df = filtered_loans_df.loc[:, (filtered_loans_df.dtypes==np.float64)].apply(lambda x: x.fillna(x.mean()),axis=0)

#filtered_categorical_df = filtered_loans_df.loc[:, (filtered_loans_df.dtypes==np.object)]

#filtered_loans_df = pd.concat([filtered_loans_mean_df,filtered_loans_df] , axis = 1 ,join='inner' ,join_axes=[filtered_loans_mean_df.index])

#filtered_loans_mean_df.info()

#filtered_categorical_df.info()

#print(filtered_categorical_df.isnull().sum().sort_values(ascending=False))


filtered_loans_df = filtered_loans_df.drop(['title' ,'zip_code','last_pymnt_d','last_credit_pull_d','earliest_cr_line' ,'sub_grade','emp_title'],axis=1)

#filtered_categorical_df = filtered_categorical_df.drop(['title' ,'zip_code','last_pymnt_d','issue_d','last_credit_pull_d','earliest_cr_line' ,'sub_grade','emp_title'],axis=1)

#filtered_categorical_df.info()
#print(filtered_categorical_df.isnull().sum().sort_values(ascending=False))
#print(filtered_loans_df['emp_title'].value_counts())
#print(filtered_loans_df['emp_length'].value_counts())

filtered_loans_df = filtered_loans_df[(filtered_loans_df["loan_status"] == "Fully Paid") |
                            (filtered_loans_df["loan_status"] == "Charged Off")]

mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0

    },
    "grade":{
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7
    }, 
    "term": {
        " 36 months": 36,
        " 60 months": 60
    },
    "addr_state": {
            "AL": 1,
            "AK": 2,
            "AZ": 3,
            "AR": 4,
            "CA": 5,
            "CO": 6,
            "CT": 7,
            "DE": 8,
            "FL": 9,
            "GA": 10,
            "HI": 11,
            "ID": 12,
            "IL": 13,
            "IN": 14,
            "IA": 15,
            "KS": 16,
            "KY": 17,
            "LA": 18,
            "ME": 19,
            "MD": 20,
            "MA": 21,
            "MI": 22,
            "MN": 23,
            "MS": 24,
            "MO": 25,
            "MT": 26,
            "NE": 27,
            "NV": 28,
            "NH": 29,
            "NJ": 30,
            "NM": 31,
            "NY": 32,
            "NC": 33,
            "ND": 34,
            "OH": 35,
            "OK": 36,
            "OR": 37,
            "PA": 38,
            "RI": 39,
            "SC": 40,
            "SD": 41,
            "TN": 42,
            "TX": 43,
            "UT": 44,
            "VT": 45,
            "VA": 46,
            "DC": 46,
            "WA": 47,
            "WV": 48,
            "WI": 49,
            "WY": 50,
 
            },
            
        "loan_status": {
            "Fully Paid": 1, 
            "Charged Off": 0
            }
    
}
    
#filtered_categorical_df = filtered_categorical_df.replace(mapping_dict1)
filtered_loans_df = filtered_loans_df.replace(mapping_dict)
#filtered_categorical_df[['emp_length','grade']].head()

#filtered_categorical_df['emp_title'].fillna('other', inplace=True)

  
#filtered_categorical_df.apply(lambda x: x.fillna(x.mode()),axis=0)

#print(filtered_categorical_df.isnull().sum().sort_values(ascending=False))


#filtered_loans_float_df = filtered_loans_df.loc[:, (filtered_loans_df.dtypes==np.float64)]

#print(filtered_loans_float_df.isnull().sum().sort_values(ascending=False))
filtered_loans_df['int_rate'] = filtered_loans_df['int_rate'].replace('%','',regex=True).astype('float')/100

filtered_loans_df['revol_util'] = filtered_loans_df['revol_util'].replace('%','',regex=True).astype('float')/100

filtered_loans_df.info()



filtered_numeric_loans_df = filtered_loans_df.select_dtypes(['float64','int64']).apply(lambda x: x.fillna(x.median()),axis=0)

print(filtered_numeric_loans_df.isnull().sum().sort_values(ascending=False))

filtered_numeric_loans_df['fico_average'] = (filtered_numeric_loans_df['fico_range_high'] + filtered_numeric_loans_df['fico_range_low']) / 2






filtered_numeric_loans_df = filtered_numeric_loans_df.drop(['fico_range_high','fico_range_low','issue_d'])

filtered_categorical_loans_df = filtered_loans_df.select_dtypes([object]).apply(lambda x: x.fillna(x.mode()),axis=0)

print(filtered_categorical_loans_df.isnull().sum().sort_values(ascending=False))

filtered_categorical_loans_df = pd.get_dummies(filtered_categorical_loans_df)

issue_d = filtered_categorical_loans_df['issue_d'].str.split('-')

filtered_loans_df['issue_month'] = issue_d[0]

filtered_loans_df['issue_year'] = issue_d[1]


final_filtered_df = pd.concat([filtered_numeric_loans_df, filtered_categorical_loans_df], axis=1)

print(filtered_numeric_loans_df.isnull().sum().sort_values(ascending=False))

final_filtered_df.to_csv('preprocessed_loans.csv')