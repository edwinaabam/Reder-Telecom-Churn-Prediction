# API is a means of communication between the backend and front end. 
# # basically  exchange of data between the fron and back

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI
import pickle
import json
import pandas as pd
from pydantic import BaseModel, Field
from DataCleaning.clean import clean_data
from typing import List, Dict, Any
import os
import uvicorn
import sys
from pathlib import Path


app = FastAPI(title= "Reder Prediction API", version="1.0")


################################
###  Get the schema for request
#################################

class PredictionRequest(BaseModel): 
    records: List[Dict[str, Any]] = Field( 
        ...,
        example=[{
        'CustomerID':1001, 
        'Name': 'Mary Barrett' , 
        'Age': 31, 
        'Gender': "Female", 
        'Location': 'Andrewfort', 
        'Email': 'mary32@example.net', 
        'Phone': '3192528777',
        'Address': '61234 Shelley Heights Suite 467 Cohentown, GU 05435',
        'Segment': 'Segment B', 
        'NPS': 3, 
        'Timestamp':"2020-01-27 01:36:49", 
        'Plan':"Express",
        'Start_Date':"2020-06-08", 
        'End_Date':"2022-10-27", 
        'PageViews':49 , 
        'TimeSpent(minutes)':15,
        'Logins':19,
        'Frequency':"weekly", 
        'Rating':1, 
        'Comment':"", 
        'TotalPurchaseFrequency':38,
        'TotalPurchaseValue':3994.23, 
        'ProductList':"Frozen Cocktail Mixes|Guacamole|Hockey Stick Care|Invitations|Mortisers|Printer, Copier & Fax Machine Accessories|Rulers", 
        'AvgLatePayment':13.34,
        'PaymentTypes':'Bank Transfer,Credit Card,PayPal',
        'NumPaymentMethod':3, 
        'TotalInteractionType':"Call,Chat,Email", 
        'num_calls':1, 
        'num_emails':1,
        'num_chats':2, 
        'FirstInteractionDate':"2019-09-26", 
        'LastInteractionDate':"2021-07-25",
        'InteractionDuration_days':"667", 
        'FirstInteractionType':"Call",
        'LastInteractionType':"Email", 
        'Action_count':24, 
        'FirstActionTime': "2020-01-15 03:14:20",
        'LastActionTime': "2022-11-05 04:45:10", 
        'AvgTimeBetweenActions_secs': 345600, 
        'TotalDaysActive': 694,
        'MostCommonAction':"Page Visit", 
        'LeastFrequentAction':"Click", 
        'ActivityDuration_days': 690,
        'ActionsPerDay': 1.0433, 
        'most_recent_action_date':"2022-11-07 02:24:31", 
        'TotalPageVisits':24,
        'unique_pages': 13, 
        'FirstActionType':"Page Visit", 
        'LastActionType':"Page Visit", 
        'FirstPageVisited':"main",
        'LastPageVisited': "author", 
        'Recency_days': 101, 
        'InactivityFlag': 0,
        'ActionsLast30Days': 5, 
        'ActiveInLastWeek': 1, 
        'TotalEmailsSent': 1,
        'TotalEmailsOpened': 1, 
        'TotalEmailsClicked':2, 
        'LastEmailSentDate':"2022-10-28 06:15:00",
        'LastEmailOpenedDate':"2022-10-30 08:20:00", 
        'LastEmailClickedDate':"2022-11-01 09:30:00", 
        'AVGOpenDays': 818.0,
        'AVGClickDays': 319.0, 
        'AvgOpenDelay_days':2.0, 
        'AvgClickDelay_days':4.0, 
        'OpenRate':1.0,
        'ClickRate':1.0, 
        'ClickToOpenRate':1.0, 
        'EverOpened':1, 
        'EverClicked':1,
        'RecencyLastOpen_days':100, 
        'RecencyLastClick_days':97, 
        'customer_segments': "loyal_customers"

    }]
    )


###################################
## Uitlity functions
##################################

#def load_assets():
  #  model_path = os.path.join('model', 'model.pkl')
   # with open(model_path, "rb") as f:
    #    model = pickle.load(f)
    
   # schema_path = os.path.join('model', 'schema.json')
   # with open(schema_path, "r") as f:
    #    schema = json.load(f)
     #   feature_schema = schema['model_schema']  

      #  return model, feature_schema


def load_assets():
    model_path = BASE_DIR / "model" / "model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    schema_path = BASE_DIR / "model" / "schema.json"
    with open(schema_path, "r") as f:
        schema = json.load(f)
        feature_schema = schema["model_schema"]

    return model, feature_schema



@app.post("/churn-predict")
def predict(req: PredictionRequest):
    data = pd.DataFrame(req.records)

    
    EXPECTED_COLS = [
        "Frequency", "Recency_days", "ActionsLast30Days",
        "ActiveInLastWeek", "num_calls", "num_emails",
        "num_chats", "Logins", "PageViews", "NPS", "Rating"
    ]

    for col in EXPECTED_COLS:
        if col not in data.columns:
            data[col] = 0

    data = clean_data(data)

    model, feature_schema = load_assets()

    data = data.reindex(columns=feature_schema, fill_value=0)

    # Prediction
    prediction = int(model.predict(data)[0])

    # Probabilities
    proba = model.predict_proba(data)[0]
    non_churn = float(proba[0])
    churn = float(proba[1])
    

    return {
        "prediction_label": prediction,                 # 0 or 1
        "prediction_class": "No Churn" if prediction == 0 else "Churn",
        "churn_probability": churn,
        "non_churn_probability": non_churn
    }

    #return {"churn_probability": probs.tolist()}
    
    # return {"predictions": predictions.tolist()}
    

##############################
## Run the app
##############################
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port = 8000)