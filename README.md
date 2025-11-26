# Irish-Classification
# Iris Flower Classification Web App

A Machine Learning web application that predicts the species of an Iris flower based on its measurements. The app also provides interactive data visualizations of the Iris dataset.

Features:
- Predicts Iris species: Setosa, Versicolor, Virginica.
- Input features: Sepal Length, Sepal Width, Petal Length, Petal Width.
- Displays interactive charts:
    - Petal Length vs Petal Width
    - Sepal Length vs Sepal Width
- Colorful and responsive web interface using Bootstrap.
- Uses FastAPI for serving the ML model.
- Model trained using Random Forest Classifier and saved with joblib.

Project Structure:
iris_classification/
│
├─ data/iris.csv                 # Dataset
├─ model/model.pkl               # Trained ML model
├─ templates/
│    ├─ index.html               # Prediction page
│    └─ visualization.html       # Visualization page
├─ static/style.css              # Custom CSS
├─ main.py                       # FastAPI app
├─ README.md                     # Project documentation
└─ venv/                         # Python virtual environment

Installation:

1. Clone the repository:
git clone <repository_url>
cd iris_classification

2. Create and activate a virtual environment:

# Windows
D:/iris_classification/venv/Scripts/activate.bat

# macOS/Linux
source venv/bin/activate

3. Install dependencies:
pip install fastapi uvicorn pandas scikit-learn joblib plotly jinja2

Usage:

1. Start the FastAPI server:
uvicorn main:app --reload

2. Open your browser and go to:
- Prediction page: http://127.0.0.1:8000/
- Data Visualization page: http://127.0.0.1:8000/visualization

3. Enter flower measurements in the prediction form to get the species.
4. Visit the Data Visualization page to explore the Iris dataset interactively.

Example Inputs:

Species      | Sepal Length | Sepal Width | Petal Length | Petal Width
------------ | ------------ | ----------- | ------------ | -----------
Setosa       | 5.1          | 3.5         | 1.4          | 0.2
Versicolor   | 6.0          | 2.7         | 4.5          | 1.5
Virginica    | 6.5          | 3.0         | 5.5          | 2.0

Technologies Used:
- Python 3.x
- FastAPI – Backend API
- Scikit-learn – Machine Learning
- Pandas – Data handling
- Joblib – Model saving/loading
- Plotly – Interactive charts
- HTML, CSS, Bootstrap – Frontend

Future Improvements:
- Highlight user input on both Petal and Sepal graphs.
- Show prediction probabilities for each species.
- Add feature importance chart for model explainability.
- Deploy to cloud (Heroku, Render, or AWS).
- Allow selection of different ML models for comparison.

Author:
Vani
Email: vanifdodamani07@gmail.com
