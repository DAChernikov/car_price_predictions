import pickle
import pandas as pd
import os

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

# Загрузка обученной модели и его фичей
with open("ridge_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# Сервисный код
app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


# Обработчик поступаемых данных
def preprocess_data(df: pd.DataFrame,
                    feature_columns: List[str]) -> pd.DataFrame:
    '''
    Обработчик датасета cars с зафиксированным составом колонок.
    На вход подается Pandas Dataframe, с которым происходит обработка данных, такая же,
    какая была в ipynb файле (DS-части ДЗ1 ML)

    params:
        df: pd.DataFrame, - подаваемый датасет формата Pandas Dataframe
        feature_columns: List[str], - список фичей, на которых обучалась модель (по ним ведется предсказание)

    return:
        df_ohe: pd.DataFrame, - готовый обработанный датасет, на котором можно строить предсказания selling_price
    '''

    medians = df.median(numeric_only=True)
    df['seats'] = df['seats'].fillna(medians['seats'])
    df['mileage'] = df['mileage'].str.replace(' kmpl', '').str.replace(
        ' km/kg', '').astype(float)
    df['engine'] = df['engine'].str.replace(' CC', '').astype(float)
    df['max_power'] = df['max_power'].str.replace(' bhp', '')
    df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
    df['name'] = df['name'].apply(lambda x: ' '.join(x.split()[:1]))

    cats = list(df.select_dtypes(include=['object']).columns) + ['seats']
    df_ohe = pd.get_dummies(df, columns=cats, drop_first=True)

    for col in feature_columns:
        if col not in df_ohe:
            df_ohe[col] = 0

    df_ohe = df_ohe[feature_columns]

    return df_ohe


# Основной функционал - обработчики POST запросов по endpoint'ам
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    '''
    POST метод получения предсказания цены автомобиля на примере одного подаваемого объекта.
    На вход поступает объект Item (класс, заданный выше, который задает список необходимых подаваемых данных для POST запроса)
    На выходе выдается результат запроса - предсказание цены автомобиля для поданного объекта
    '''
    df = pd.DataFrame([item.dict()])
    feature_columns = features
    df_ohe = preprocess_data(df, feature_columns)
    prediction = model.predict(df_ohe)

    return int(prediction[0])


@app.post("/predict_items")
async def predict_items(file: UploadFile):
    '''
    POST метод получения предсказания цены автомобиля на примере подаваемого датасета формата csv.
    На вход поступает объект file: UploadFile (текстовый файл, получаемый в ходе Post-запроса)
    На выходе выдается результат запроса - предсказание цен автомобилей, содержащихся в поданном датасете.
    Также в результате запроса происходит генерация файла predictions.csv, который содержит в себе тот же датасет,
    за исключением наличия дополнительной колонки с результатом предсказания цены автомобиля моделью.
    '''
    feature_columns = features

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400,
                            detail="Uploaded file must be a CSV.")
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error reading the CSV file: {str(e)}")
    if df.empty:
        raise HTTPException(status_code=400,
                            detail="The provided CSV file is empty.")

    try:
        df_processed = preprocess_data(df, feature_columns)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error during preprocessing: {str(e)}")

    predictions = model.predict(df_processed)
    df["predictions"] = predictions.astype(int)
    output_path = os.path.join('./data/', "predictions.csv")
    df.to_csv(output_path, index=False)
    return FileResponse(output_path, filename="predictions.csv",
                        media_type="text/csv")
