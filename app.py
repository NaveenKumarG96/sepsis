from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any
from fastapi import FastAPI
import sys,os
import pickle
import pandas as pd
import json
from xgboost import XGBClassifier
import numpy as np
from functions import max_diff, min_diff, mean_diff, median_diff

app = FastAPI() 

model = pickle.load(open('finalized_model.sav', 'rb'))

class QueryRequest(BaseModel):
    features: Any

def predict_features(data):

    for vital in data:
        
        vital_data = json.loads(data[vital])
        if vital == 'Heart rate':
          HeartRate_df = pd.DataFrame(vital_data)
        if vital == 'Respiration rate': 
          Respiratoryrate_df = pd.DataFrame(vital_data)
        if vital == 'SystolicBP':
          SystolicBP_df = pd.DataFrame(vital_data)
        if vital == 'MAP':
          MAP_df = pd.DataFrame(vital_data)
        if vital == 'Temperature':
          Temperature_df = pd.DataFrame(vital_data)
        if vital == 'O2sat':
          OS_df = pd.DataFrame(vital_data)
        
         
    HeartRate_df.dropna(subset=['Heart rate'], inplace=True)
    HeartRate_df['Heart rate trend'] = HeartRate_df.groupby('icustay_id')['Heart rate'].transform(lambda x: (x > 100).sum() >= 3).astype(int) 


    HeartRate_agg_df = HeartRate_df.groupby('icustay_id').agg({'Heart rate': [max_diff, min_diff,mean_diff,median_diff]}).reset_index()
    HeartRate_agg_df.columns = ['icustay_id'] + [f'{col}_{agg}' for col, agg in HeartRate_agg_df.columns[1:]]

    HeartRate_final_df = pd.merge(HeartRate_agg_df , HeartRate_df[['icustay_id','Heart rate trend']], on=['icustay_id'], how='left')
    HeartRate_final_df.drop_duplicates(inplace=True)
    HeartRate_final_df = pd.concat([HeartRate_final_df[~HeartRate_final_df.duplicated(subset=['icustay_id'], keep=False)], HeartRate_final_df[HeartRate_final_df.duplicated(subset=['icustay_id'], keep=False) & (HeartRate_final_df['Heart rate trend'] == 1)]], axis=0,ignore_index=True)


    Respiratoryrate_df.dropna(subset=['Respiration rate'], inplace=True)
    Respiratoryrate_df['Respiration rate trend'] = Respiratoryrate_df.groupby('icustay_id')['Respiration rate'].transform(lambda x: (x > 20).sum() >= 3).astype(int)

    Respiratoryrateagg_df = Respiratoryrate_df.groupby('icustay_id').agg({'Respiration rate': [max_diff, min_diff,mean_diff,median_diff]}).reset_index()
    Respiratoryrateagg_df.columns = ['icustay_id'] + [f'{col}_{agg}' for col, agg in Respiratoryrateagg_df.columns[1:]]

    Respiratoryrate_final_df = pd.merge(Respiratoryrateagg_df , Respiratoryrate_df[['icustay_id','Respiration rate trend']], on=['icustay_id'], how='left')
    Respiratoryrate_final_df.drop_duplicates(inplace=True)
    Respiratoryrate_final_df = pd.concat([Respiratoryrate_final_df[~Respiratoryrate_final_df.duplicated(subset=['icustay_id'], keep=False)], Respiratoryrate_final_df[Respiratoryrate_final_df.duplicated(subset=['icustay_id'], keep=False) & (Respiratoryrate_final_df['Respiration rate trend'] == 1)]], axis=0,ignore_index=True)


    SystolicBP_df.dropna(subset=['SystolicBP'], inplace=True)
    SystolicBP_df['SystolicBP trend'] = SystolicBP_df.groupby('icustay_id')['SystolicBP'].transform(lambda x: (x < 100).sum() >= 2).astype(int)


    SystolicBPagg_df = SystolicBP_df.groupby('icustay_id').agg({'SystolicBP': [max_diff, min_diff,mean_diff,median_diff]}).reset_index()
    SystolicBPagg_df.columns = ['icustay_id'] + [f'{col}_{agg}' for col, agg in SystolicBPagg_df.columns[1:]]

    SystolicBP_final_df = pd.merge(SystolicBPagg_df , SystolicBP_df[['icustay_id','SystolicBP trend']], on=['icustay_id'], how='left')
    SystolicBP_final_df.drop_duplicates(inplace=True)
    SystolicBP_final_df = pd.concat([SystolicBP_final_df[~SystolicBP_final_df.duplicated(subset=['icustay_id'], keep=False)], SystolicBP_final_df[SystolicBP_final_df.duplicated(subset=['icustay_id'], keep=False) & (SystolicBP_final_df['SystolicBP trend'] == 1)]], axis=0,ignore_index=True)

    MAP_df.dropna(subset=['MAP'], inplace=True)
    #MAP_df = identify_trend(MAP_df, 'MAP',increasing = 0,threshold=10, window_size=3)
    MAP_df['MAP trend'] = MAP_df.groupby('icustay_id')['MAP'].transform(lambda x: (x < 70).sum() >= 2).astype(int)

    MAP_agg_df = MAP_df.groupby('icustay_id').agg({'MAP': [max_diff, min_diff,mean_diff,median_diff]}).reset_index()
    MAP_agg_df.columns = ['icustay_id'] + [f'{col}_{agg}' for col, agg in MAP_agg_df.columns[1:]]

    MAP_final_df = pd.merge(MAP_agg_df , MAP_df[['icustay_id','MAP trend']], on=['icustay_id'], how='left')
    MAP_final_df.drop_duplicates(inplace=True)
    MAP_final_df = pd.concat([MAP_final_df[~MAP_final_df.duplicated(subset=['icustay_id'], keep=False)], MAP_final_df[MAP_final_df.duplicated(subset=['icustay_id'], keep=False) & (MAP_final_df['MAP trend'] == 1)]], axis=0,ignore_index=True)


    Temperature_df.dropna(subset=['Temperature'], inplace=True)
    Temperature_df['Temperature trend'] = Temperature_df.groupby('icustay_id')['Temperature'].transform(lambda x: ((x > 98.96) & (x < 99.68)).sum() >= 3).astype(int)

    Temperature_agg_df = Temperature_df.groupby('icustay_id').agg({'Temperature': [max_diff, min_diff,mean_diff,median_diff]}).reset_index()
    Temperature_agg_df.columns = ['icustay_id'] + [f'{col}_{agg}' for col, agg in Temperature_agg_df.columns[1:]]

    Temperature_final_df = pd.merge(Temperature_agg_df , Temperature_df[['icustay_id','Temperature trend']], on=['icustay_id'], how='left')
    Temperature_final_df.drop_duplicates(inplace=True)
    Temperature_final_df = pd.concat([Temperature_final_df[~Temperature_final_df.duplicated(subset=['icustay_id'], keep=False)], Temperature_final_df[Temperature_final_df.duplicated(subset=['icustay_id'], keep=False) & (Temperature_final_df['Temperature trend'] == 1)]], axis=0,ignore_index=True)


    OS_df.dropna(subset=['O2sat'], inplace=True)        
    OS_df['O2sat trend'] = OS_df.groupby('icustay_id')['O2sat'].transform(lambda x: (x < 95).sum() >= 3).astype(int)

    OS_agg_df = OS_df.groupby('icustay_id').agg({'O2sat': [max_diff, min_diff,mean_diff,median_diff]}).reset_index()
    OS_agg_df.columns = ['icustay_id'] + [f'{col}_{agg}' for col, agg in OS_agg_df.columns[1:]]

    OS_final_df = pd.merge(OS_agg_df , OS_df[['icustay_id','O2sat trend']], on=['icustay_id'], how='left')
    OS_final_df.drop_duplicates(inplace=True)

    OS_final_df.at[0, 'icustay_id'] = 200166.00
    OS_final_df


    Patients_sepsis_yes_df = pd.merge(HeartRate_final_df , Respiratoryrate_final_df, on=['icustay_id'], how='left')
    Patients_sepsis_yes_df = pd.merge(Patients_sepsis_yes_df , SystolicBP_final_df, on=['icustay_id'], how='left')
    Patients_sepsis_yes_df = pd.merge(Patients_sepsis_yes_df , MAP_final_df, on=['icustay_id'], how='left')
    Patients_sepsis_yes_df = pd.merge(Patients_sepsis_yes_df , Temperature_final_df, on=['icustay_id'], how='left')
    Patients_sepsis_yes_df = pd.merge(Patients_sepsis_yes_df , OS_final_df, on=['icustay_id'], how='left')

    demog = pd.read_csv(r"demog.csv",sep='|')


    Patients_sepsis_yes_df.drop(columns=['MAP_mean_diff','MAP_min_diff','MAP_median_diff','MAP_max_diff','MAP trend','O2sat_median_diff','O2sat_mean_diff','O2sat_min_diff','O2sat_max_diff','O2sat trend'],inplace =True)
    Patients_sepsis_yes_df

    Patients_sepsis_yes_df.drop(columns=['icustay_id'],inplace =True)
   
    prediction = model.predict(Patients_sepsis_yes_df)
   
    print(prediction)
    sepsis_status = np.where(prediction == 1, "Sepsis", "No sepsis") 
    return sepsis_status

@app.get("/")
async def get_index():
    return "HI"

# Route to do classifier post call
@app.post("/predict")
async def classify(features_data: QueryRequest):
    input_features = features_data.features
    predicted_label = predict_features(input_features)
   
    predicted_label = predicted_label.tolist()
    response_data = {"sepsis": predicted_label}
    return response_data

