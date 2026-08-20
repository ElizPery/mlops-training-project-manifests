import os
from dotenv import load_dotenv

import shutil

import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# Load environment variables
load_dotenv()

# Configuration via Environment Variables or Defaults
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PUSHGATEWAY_URI = os.getenv("PUSHGATEWAY_URI", "localhost:9091")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadminpassword")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

# Setup S3/MinIO environment for MLflow Artifacts
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL

# Set MLflow Tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "iris-classification-grid-search"
mlflow.set_experiment(EXPERIMENT_NAME)

# Load Dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter Grid
grid = [
    {"C": 0.01, "max_iter": 100},
    {"C": 0.1, "max_iter": 200},
    {"C": 1.0, "max_iter": 300},
    {"C": 10.0, "max_iter": 500},
]

best_accuracy = -1.0
best_run_id = None
best_model_artifact_path = None

print(f"Starting MLflow experiment: {EXPERIMENT_NAME}")
print(f"MLflow Tracking Server: {MLFLOW_TRACKING_URI}")
print(f"Prometheus PushGateway: {PUSHGATEWAY_URI}")

for params in grid:
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Train Model
        clf = LogisticRegression(
            C=params["C"], 
            max_iter=params["max_iter"], 
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        # Predict & Evaluate
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        
        acc = float(accuracy_score(y_test, y_pred))
        loss = float(log_loss(y_test, y_proba))
        
        # 1. Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("loss", loss)
        mlflow.sklearn.log_model(clf, artifact_path="model")
        
        print(f"Run {run_id} | Params: {params} | Accuracy: {acc:.4f} | Loss: {loss:.4f}")
        
        # 2. Push Metrics to Prometheus PushGateway
        registry = CollectorRegistry()
        g_acc = Gauge(
            "mlflow_accuracy", 
            "Model Accuracy Score from MLflow Experiment", 
            ["run_id", "c_param", "max_iter"], 
            registry=registry
        )
        g_loss = Gauge(
            "mlflow_loss", 
            "Model Log Loss from MLflow Experiment", 
            ["run_id", "c_param", "max_iter"], 
            registry=registry
        )
        
        g_acc.labels(
            run_id=run_id, 
            c_param=str(params["C"]), 
            max_iter=str(params["max_iter"])
        ).set(acc)
        
        g_loss.labels(
            run_id=run_id, 
            c_param=str(params["C"]), 
            max_iter=str(params["max_iter"])
        ).set(loss)
        
        try:
            push_to_gateway(
                PUSHGATEWAY_URI, 
                job="mlflow_model_training", 
                registry=registry
            )
            print(f"  └─ Successfully pushed metrics to PushGateway for run {run_id}")
        except Exception as e:
            print(f"  └─ ⚠️ Failed to push to PushGateway: {e}")
            
        # Track Best Model
        if acc > best_accuracy:
            best_accuracy = acc
            best_run_id = run_id

# 3. Download Best Model Artifact to Local Directory
if best_run_id:
    print("\n" + "="*50)
    print(f"Best Model Run ID: {best_run_id} with Accuracy: {best_accuracy:.4f}")
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "best_model"))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    local_path = mlflow.artifacts.download_artifacts(
        run_id=best_run_id, 
        artifact_path="model", 
        dst_path=output_dir
    )
    print(f"✅ Best model artifact downloaded to: {local_path}")