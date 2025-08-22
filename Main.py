# backend/main.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
from LinearRegression import linearregression
from PolynomialRegression import Polyregression
from RidgeRegrssion import RidgeRegressionModel
from LassoRegression import LasooRegressionModel
from Logisticregression import LogisticRegressionModel
from KnnClassifier import KNNModel
from SVM_Classifier import SVMModel
from DecisionTreeClassifier import DecisionTreeClassifierModel
from KMeans_cluster import KMeansClusteringModel
from pipeline import Preprocessing

app = FastAPI()

# Enable CORS (for Lovable or any frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_objects = {
    "linear": LinearRegressionModel,
    "poly": Polyregression,
    "ridge": RidgeRegressionModel,
    "lasso": LassoRegressionModel,
    "logistic": LogisticRegressionModel,
    "knn": KNNClassifier,
    "svm": SVMClassifier,
    "decision_tree": DecisionTreeClassifierModel,
    "kmeans": KMeansClustering
}

@app.post("/train")
async def train_model(
    file: UploadFile,
    target_column: str = Form(...),
    model_name: str = Form(...),
    poly_degree: int = Form(2),
    k_neighbors: int = Form(3),
    n_clusters: int = Form(3)
):
    df = pd.read_csv(io.BytesIO(await file.read()))
    preprocessor = Preprocessing(df)
    preprocessor.duplicate()
    preprocessor.missing()
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Dynamic Model Instantiation
    if model_name == "poly":
        model = Polyregression(poly=poly_degree)
    elif model_name == "knn":
        model = KNNClassifier(n_neighbors=k_neighbors)
    elif model_name == "kmeans":
        model = KMeansClustering(n_clusters=n_clusters)
    else:
        model = model_objects[model_name]()

    model.train(X, y)
    score = model.evaluate(X, y) if hasattr(model, 'evaluate') else None
    coeffs = model.get_coefficients() if hasattr(model, 'get_coefficients') else None

    return {
        "message": f"{model_name} model trained successfully.",
        "score": score,
        "coefficients": coeffs
    }
