from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app = FastAPI()


class model_input(BaseModel):

    age: float
    weight: float
    bmi: float
    gender: float
    d1_lactate_max: float
    d1_lactate_min: float
    apache_4a_hospital_death_prob: float
    apache_4a_icu_death_prob: float


model = pickle.load(open(
    r'C:\Users\Lenovo\OneDrive\Desktop\varun\WEBDEV\EMID\API\survival-predicton\Model.pkl', 'rb'))


@app.post('/predict')
def survivalprediction(input_parameters: model_input):

    input_data = input_parameters.json()
    input_dict = json.loads(input_data)

    age = input_dict['age']
    weight = input_dict['weight']
    bmi = input_dict['bmi']
    gender = input_dict['gender']
    d1_max = input_dict['d1_lactate_max']
    d1_min = input_dict['d1_lactate_min']
    h_death_prob = input_dict['apache_4a_hospital_death_prob']
    icu_death_prob = input_dict['apache_4a_icu_death_prob']

    input_list = [age, weight, bmi, gender, d1_max,
                  d1_min, h_death_prob, icu_death_prob]

    prediction_proba = model.predict_proba([input_list])
    prediction = model.predict([input_list])

    if prediction[0] == 1:
        return f" {str(prediction_proba[0][1])}"
    else:
        return f" {str(prediction_proba[0][1])}"
