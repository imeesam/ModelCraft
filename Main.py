# backend/main.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
from models.linear_regression import LinearRegressionModel
from models.poly_regression import Polyregression
from models.ridge_regression import RidgeRegressionModel
from models.lasso_regression import LassoRegressionModel
from models.logistic_regression import LogisticRegressionModel
from models.knn_classifier import KNNClassifier
from models.svm_classifier import SVMClassifier
from models.decision_tree_classifier import DecisionTreeClassifierModel
from models.kmeans import KMeansClustering
from pipeline.preprocessing import Preprocessing

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
