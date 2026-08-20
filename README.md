# MLOps Training Project — Experiment Tracking & Monitoring

This repository contains GitOps deployment manifests for Argo CD and a Python model training pipeline. The project implements end-to-end experiment tracking using **MLflow**, artifact storage with **MinIO**, metadata persistence in **PostgreSQL**, and metric export to **Grafana** via **Prometheus PushGateway**.

---

## Repository Structure

The Argo CD `ApplicationSet` is configured to track the `namespace/*` directory. Each subdirectory corresponds to an isolated Kubernetes Namespace in the cluster.

```text
mlops-training-project-manifests/
├── namespace/
│   ├── application/
│   │   ├── ns.yaml          # Creates the 'application' namespace
│   │   ├── minio.yaml
│   │   ├── mlflow-postgress.yaml
│   │   ├── mlflow.yaml
│   │   └── pushgateaway.yaml 
│   ├── infra-tools/
│   │   ├── kube-prometheus-stack.yaml 
│   │   ├── loki.yaml 
│   │   └── ns.yaml          # Namespace for system/infrastructure tools
└── README.md
```

---

## Infrastructure Architecture

* **MLflow Tracking Server:** Deployed in the `application` namespace (ClusterIP, Port `5000`).
* **MinIO:** S3-compatible artifact storage in the `application` namespace (`mlflow-artifacts` bucket).
* **PostgreSQL:** Relational database for MLflow metadata in the `application` namespace.
* **Prometheus PushGateway:** Ephemeral metrics collection service in the `monitoring` namespace (Port `9091`).
* **Prometheus & Grafana (kube-prometheus-stack):** Monitoring and visualization suite in the `infra-tools` namespace.

---

## Step 1: Verify Argo CD Deployment

Ensure all applications are successfully deployed and in a `Synced / Healthy` state:

```bash
# Check Argo CD Application statuses
kubectl get applications -n infra-tools

# Check pod statuses across namespaces
kubectl get pods -n application
kubectl get pods -n monitoring
kubectl get pods -n infra-tools
```

## Step 2: Configure Port-Forwarding

To allow the local Python script to communicate with cluster services, open separate terminal windows and run:

```bash
# MLflow Tracking Server (namespace: application)
kubectl port-forward svc/mlflow -n application 5000:5000

# MinIO S3 API (namespace: application)
kubectl port-forward svc/minio -n application 9000:9000

# Prometheus PushGateway (namespace: monitoring)
kubectl port-forward svc/prometheus-pushgateway -n monitoring 9091:9091

# Grafana UI (namespace: infra-tools)
kubectl port-forward svc/prometheus-operator-grafana -n infra-tools 3000:80
```

## Step 3: Environment Setup (.env)

In the `experiments/` directory, create a `.env` file matching the credentials defined in your Helm manifests:

```bash
# MLflow Tracking Server
MLFLOW_TRACKING_URI=http://localhost:5000

# Prometheus PushGateway
PUSHGATEWAY_URI=localhost:9091

# MinIO S3 Credentials (from minio.yaml)
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```

## Step 4: Execute Training Pipeline (train_and_push.py)

1. Install required dependencies:

```bash
cd experiments
pip install -r requirements.txt
```

2. Run the model training and metric push script:

```bash
python train_and_push.py
```

3. Pipeline Logic:

- Loads the Iris dataset and trains multiple `LogisticRegression` models with hyperparameter tuning (C, max_iter).
- Logs parameters, metrics (`accuracy`, `loss`), and model artifacts to MLflow.
- Pushes `mlflow_accuracy` and `mlflow_loss` metrics labeled with run_id to Prometheus PushGateway.
- Identifies the best-performing model based on `accuracy` and exports it to the local `best_model/` directory.

## Step 5: Verify Results in Web Dashboards

In the `experiments/` directory, create a `.env` file matching the credentials defined in your Helm manifests:

1. **MLflow UI** (http://localhost:5000)

- Open `http://localhost:5000` in your browser.
- Select the `iris-classification-grid-search` experiment.
- Confirm the presence of 4 runs, hyperparameter values, metrics, and model artifacts stored in MinIO.

2. **PushGateway UI** (http://localhost:9091)

- Open `http://localhost:9091`.
- Verify that metrics are listed under `job="mlflow_model_training"`.

3. **Grafana Explore** (http://localhost:3000)

- Navigate to Grafana → Explore (Username: `admin`, Password: `prom-operator`).
- Select the Prometheus datasource.
- Run PromQL queries to view experiment metrics:

```bash
# View accuracy across all runs
mlflow_accuracy

# View log loss across all runs
mlflow_loss
```

---

## Resource Teardown

Go to repo where infrastracture was created: `https://github.com/ElizPery/mlops-training-project.git`

To avoid incurring unnecessary cloud expenses, destroy the resources when testing is complete. **Order matters**: you must destroy the EKS cluster prior to destroying the VPC.

```bash
# 1. Destroy Argo CD and ApplicationSet
cd argocd
terraform destroy 

# 2. Destroy EKS Cluster
cd ../eks
terraform destroy 

# 3. Destroy VPC Network
cd ../vpc
terraform destroy 
```