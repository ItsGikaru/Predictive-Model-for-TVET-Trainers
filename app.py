# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from tvettrainers import TvetTrainersBase
import numpy as np
import pickle
import pandas as pd
import warnings
from fastapi.middleware.cors import CORSMiddleware
# 2. Create the app object
app = FastAPI()
pickle_in = open("gb_model.pkl","rb")
gb_pipe=pickle.load(pickle_in)

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
# 3. Index route, opens automatically on http://127.0.0.1:8000

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def index():
    return {'Welcome To Predictive Model for TVET Trainers Demand in Kenya'}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_trainer_demand(data: TvetTrainersBase):
    # Convert Pydantic model to dictionary, then to DataFrame
    input_data = pd.DataFrame([{
        'KNQA_Level_course': data.KNQA_Level_course,
        'Course_Name': data.CourseName,
        'ExamBody': data.ExamBody,
        'Institution_Type': data.Institution_Type
    }])
    
    try:
        # Predict using your Gradient Boosting model
        predicted_demand = gb_pipe.predict(input_data)
        
        return {
            'predicted_trainer_demand': int(predicted_demand[0])
        }
    except Exception as e:
        return {
            'error': f'Prediction failed: {str(e)}',
            'predicted_trainer_demand': None
        }

# Alternative approach using dictionary unpacking (more concise)
@app.post('/predict-alt')
def predict_trainer_demand_alt(data: TvetTrainersBase):
    # Convert to DataFrame directly from model dict
    input_data = pd.DataFrame([data.model_dump])
    
    try:
        predicted_demand = gb_pipe.predict(input_data)
        
        return {
            'predicted_trainer_demand': round(predicted_demand[0]),
            'input_data': data.model_dump
        }
    except Exception as e:
        return {
            'error': f'Prediction failed: {str(e)}',
            'predicted_trainer_demand': None
        }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
# if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
