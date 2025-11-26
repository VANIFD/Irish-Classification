from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load trained model
model_path = os.path.join("model", "model.pkl")
model = joblib.load(model_path)

# Load dataset
data = pd.read_csv("data/iris.csv")
species_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
data["species_num"] = data["species"].map(species_map)

species_reverse_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# Prediction page
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": ""})

@app.post("/", response_class=HTMLResponse)
def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    # Predict species
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    species_name = species_reverse_map[int(prediction)]
    return templates.TemplateResponse("index.html", {"request": request, "prediction": species_name})

# Data visualization page
@app.get("/visualization", response_class=HTMLResponse)
def visualization(request: Request):
    # Petal scatter plot
    fig_petal = px.scatter(
        data, x="petal_length", y="petal_width",
        color="species",
        title="Petal Length vs Petal Width",
        color_discrete_map={"setosa": "#FF5733", "versicolor": "#33FF57", "virginica": "#3357FF"}
    )

    # Sepal scatter plot
    fig_sepal = px.scatter(
        data, x="sepal_length", y="sepal_width",
        color="species",
        title="Sepal Length vs Sepal Width",
        color_discrete_map={"setosa": "#FF5733", "versicolor": "#33FF57", "virginica": "#3357FF"}
    )

    # Convert plots to HTML
    plot_petal_html = fig_petal.to_html(full_html=False)
    plot_sepal_html = fig_sepal.to_html(full_html=False)

    return templates.TemplateResponse(
        "visualization.html",
        {"request": request, "plot_petal": plot_petal_html, "plot_sepal": plot_sepal_html}
    )
