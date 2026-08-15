# MLOps Training Project — GitOps Manifests

This repository contains Kubernetes manifests for automated application deployment to an AWS EKS cluster managed by **Argo CD ApplicationSet**.

---

## Repository Structure

The Argo CD `ApplicationSet` is configured to track the `namespace/*` directory. Each subdirectory corresponds to an isolated Kubernetes Namespace in the cluster.

```text
mlops-training-project-manifests/
├── namespace/
│   ├── application/
│   │   ├── ns.yaml          # Creates the 'application' namespace
│   │   └── demo-nginx.yaml  # Deployment manifest for the demo Nginx app
│   └── infra-tools/
│       └── ns.yaml          # Namespace for system/infrastructure tools
└── README.md
```