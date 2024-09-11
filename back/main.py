from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('rf_model.joblib')
scaler = joblib.load('scaler.joblib')

# Load columns used in training
columns_to_use = joblib.load('columns_to_use.joblib')

app = FastAPI()

origins = [
    "https://Final_BC_Predict-frontend.onrender.com",  # ใส่ URL ของ frontend app
    "http://localhost:3000"  # สำหรับการทดสอบ local
]


# Allow CORS for frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ควรระบุโดเมน frontend ของคุณแทน '*' เพื่อความปลอดภัย
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    BRCA: list[str]
    BMI_GROUP: dict[str, float]
    AGE_GROUP: int
    PROVINCE_GROUP: list[str]
    GENDER_N: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())

        # Mapping BRCA
        brca_mapping = {'negative': '1:N', 'positive': '2:P'}
        df['BRCA'] = [brca_mapping.get(val, '0:Unknown') for val in data.BRCA]

        # Calculate BMI Group
        weight = data.BMI_GROUP.get('weight', 0)
        height_cm = data.BMI_GROUP.get('height', 0)
        height_m = height_cm / 100
        bmi = weight / (height_m ** 2) if height_m > 0 else 0
        
        bmi_group = '1:<19' if bmi <= 18.5 else '2:<25' if bmi <= 24.9 else '3:<30' if bmi <= 29.9 else '4:<99' if bmi < 99 else '0:No'
        df['BMI_GROUP'] = bmi_group

        # Determine Age Group
        age = data.AGE_GROUP
        age_group = '1:<30' if age < 30 else '2:<40' if age <= 39 else '3:<50' if age <= 49 else '4:<99' if age < 99 else '0:Unknown'
        df['AGE_GROUP'] = age_group

        # Map Province Group
        province_mapping = {
            'ยะลา': 1, 'ปัตตานี': 1, 'นราธิวาส': 1,
            'สงขลา': 2, 'สตูล': 2,
            'พังงา': 3, 'พัทลุง': 3,
            'อื่นๆ': 4
        }
        province_group = province_mapping.get(data.PROVINCE_GROUP[0], 4)
        df['PROVINCE_GROUP'] = province_group

        # Map Gender
        gender_mapping = {'Male': 0, 'Female': 1}
        df['GENDER_N'] = gender_mapping.get(data.GENDER_N, -1)

        # Prepare DataFrame for prediction
        df = pd.get_dummies(df, drop_first=True)
        df = df.reindex(columns=columns_to_use, fill_value=0)
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)

        result = "At risk of cancer" if prediction[0] == 1 else "Healthy"
        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")

