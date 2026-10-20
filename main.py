from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(title="Fretting Dissipated Energy Prediction in BoF Contacts")

Feature_order = ["P", "D", 'µ', 'µP', 'µPD', 'a', 'D_over_a']
lr = joblib.load(r"D:\PyTorch\models\linear_regression_energy.joblib")
svr = joblib.load(r"D:\PyTorch\models\svr_energy.joblib")
rf = joblib.load(r"D:\PyTorch\models\random_forest_energy.joblib")
xgb = joblib.load(r"D:\PyTorch\models\xgboost_energy.joblib")
final_model = joblib.load(r"D:\PyTorch\models\final_dissipated_energy_model.joblib")

class FrettingInput(BaseModel):
                
                P : float = Field(..., gt=0, description="Applied normal load in N", examples=[20.0])
                µ : float = Field(..., gt=0, description="Coefficient of friction", examples=[0.10])
                D : float = Field(..., gt=0, description="Displacement amplitude in mm", examples=[7.5])
                a : float = Field(..., gt=0, description="Contact radius", examples=[0.10])
   

class PredictionResponse(BaseModel):

                predicted_DE: float
                energy_unit: str
                input_features: dict[str, float]
                derived_features: dict[str, float]

@app.get("/")
def home():
        return {
                "message": "Fretting API for DE prediction is running",
                "documentation": "/docs",
                "healthcheck": "/health"
        }

@app.get("/health")
def healthcheck():
        return {
                "Status": "Healthy",
                "model_loaded": "True",
                "model_name": "LR"
        }

@app.post('/predict_Linear_regression')
def predict_lr(data: FrettingInput):
        try:
                µP = data.µ * data.P
                µPD = data.µ * data.P * data.D
                D_over_a = data.D/data.a
                input_dataframe = pd.DataFrame({
                        "P": [data.P],
                        "µ": [data.µ],
                        "D": [data.D],
                        "µP": [µP],
                        "µPD": [µPD],
                        "a": [data.a],
                        "D_over_a": [D_over_a]
                })
                input_dataframe = input_dataframe[Feature_order]
                prediction_lr = lr.predict(input_dataframe)
                predicted_Energy_lr = float(prediction_lr[0])


                return PredictionResponse(
                        predicted_DE = predicted_Energy_lr,
                        energy_unit='mJ',
                        input_features = {
                                "P": data.P,
                                "µ": data.µ,
                                "D": data.D,
                                "a": data.a                                
                        },
                        derived_features ={
                        "µP": µP,
                        "µPD": µPD,
                        "D_over_a": D_over_a}
                )
        except Exception as error:
                raise HTTPException(
                        status_code=500, detail=f"Prediction failed: {str(error)}"
                )

@app.post('/predict_Support_Vector_regression')
def predict_svr(data: FrettingInput):
        try:
                µP = data.µ * data.P
                µPD = data.µ * data.P * data.D
                D_over_a = data.D/data.a
                input_dataframe = pd.DataFrame({
                        "P": [data.P],
                        "µ": [data.µ],
                        "D": [data.D],
                        "µP": [µP],
                        "µPD": [µPD],
                        "a": [data.a],
                        "D_over_a": [D_over_a]
                })
                input_dataframe = input_dataframe[Feature_order]
                prediction_svr = svr.predict(input_dataframe)
                predicted_Energy_svr = float(prediction_svr[0])

                return PredictionResponse(
                        predicted_DE = predicted_Energy_svr,
                        energy_unit='mJ',
                        input_features = {
                                "P": data.P,
                                "µ": data.µ,
                                "D": data.D,
                                "a": data.a                                
                        },
                        derived_features ={
                        "µP": µP,
                        "µPD": µPD,
                        "D_over_a": D_over_a}
                )
        except Exception as error:
                raise HTTPException(
                        status_code=500, detail=f"Prediction failed: {str(error)}"
                )

@app.post('/predict_Random_Forest_regression')
def predict_rf(data: FrettingInput):
        try:
                µP = data.µ * data.P
                µPD = data.µ * data.P * data.D
                D_over_a = data.D/data.a
                input_dataframe = pd.DataFrame({
                        "P": [data.P],
                        "µ": [data.µ],
                        "D": [data.D],
                        "µP": [µP],
                        "µPD": [µPD],
                        "a": [data.a],
                        "D_over_a": [D_over_a]
                })
                input_dataframe = input_dataframe[Feature_order]
                prediction_rf = rf.predict(input_dataframe)
                predicted_Energy_rf = float(prediction_rf[0])

                return PredictionResponse(
                        predicted_DE = predicted_Energy_rf,
                        energy_unit='mJ',
                        input_features = {
                                "P": data.P,
                                "µ": data.µ,
                                "D": data.D,
                                "a": data.a                                
                        },
                        derived_features ={
                        "µP": µP,
                        "µPD": µPD,
                        "D_over_a": D_over_a}
                )
        except Exception as error:
                raise HTTPException(
                        status_code=500, detail=f"Prediction failed: {str(error)}"
                )

@app.post('/predict_XGBoost_regression')
def predict_xgb(data: FrettingInput):
        try:
                µP = data.µ * data.P
                µPD = data.µ * data.P * data.D
                D_over_a = data.D/data.a
                input_dataframe = pd.DataFrame({
                        "P": [data.P],
                        "µ": [data.µ],
                        "D": [data.D],
                        "µP": [µP],
                        "µPD": [µPD],
                        "a": [data.a],
                        "D_over_a": [D_over_a]
                })
                input_dataframe = input_dataframe[Feature_order]
                prediction_xgb = xgb.predict(input_dataframe)
                predicted_Energy_xgb = float(prediction_xgb[0])
        
                return PredictionResponse(
                        predicted_DE = predicted_Energy_xgb,
                        energy_unit='mJ',
                        input_features = {
                                "P": data.P,
                                "µ": data.µ,
                                "D": data.D,
                                "a": data.a                                
                        },
                        derived_features ={
                        "µP": µP,
                        "µPD": µPD,
                        "D_over_a": D_over_a}
                )
        except Exception as error:
                raise HTTPException(
                        status_code=500, detail=f"Prediction failed: {str(error)}"
                )

@app.post('/predict_Best_Model_regression')
def predict_best(data: FrettingInput):
        try:
                µP = data.µ * data.P
                µPD = data.µ * data.P * data.D
                D_over_a = data.D/data.a
                input_dataframe = pd.DataFrame({
                        "P": [data.P],
                        "µ": [data.µ],
                        "D": [data.D],
                        "µP": [µP],
                        "µPD": [µPD],
                        "a": [data.a],
                        "D_over_a": [D_over_a]
                })
                input_dataframe = input_dataframe[Feature_order]
                prediction_final_model = final_model.predict(input_dataframe)
                predicted_Energy_final_model = float(prediction_final_model[0])

                return PredictionResponse(
                        predicted_DE = predicted_Energy_final_model,
                        energy_unit='mJ',
                        input_features = {
                                "P": data.P,
                                "µ": data.µ,
                                "D": data.D,
                                "a": data.a                                
                        },
                        derived_features ={
                        "µP": µP,
                        "µPD": µPD,
                        "D_over_a": D_over_a}
                )
        except Exception as error:
                raise HTTPException(
                        status_code=500, detail=f"Prediction failed: {str(error)}"
                )