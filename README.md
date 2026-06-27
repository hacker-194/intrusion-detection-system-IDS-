# 🛡️ CyberLens IDS

A real-time **AI-powered Intrusion Detection System (IDS)** that detects malicious network traffic using a hybrid machine learning approach. CyberLens combines deep learning, gradient boosting, and ensemble learning to provide accurate, explainable, and scalable intrusion detection with a modern web dashboard.

## ✨ Features

* Hybrid IDS using **Autoencoder**, **LightGBM**, and **Stacking Meta-Model**
* Real-time network traffic analysis
* SHAP-based prediction explainability
* Background model retraining with concept drift detection
* FastAPI REST API with WebSocket support
* React dashboard for live monitoring
* Prometheus metrics and structured logging
* Docker support for easy deployment

## 🛠️ Tech Stack

| Category         | Technologies                          |
| ---------------- | ------------------------------------- |
| Backend          | FastAPI, Python                       |
| Machine Learning | PyTorch, LightGBM, Scikit-learn, SHAP |
| Frontend         | React, Vite                           |
| Monitoring       | Prometheus                            |
| Database         | SQLite                                |
| Deployment       | Docker, Docker Compose                |
| Network Capture  | NFStream                              |

## 📁 Project Structure

```text
.
├── main.py
├── model.py
├── capture_engine.py
├── frontend/
├── models/
├── tests.py
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/your-username/CyberLens.git
cd CyberLens
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the backend

```bash
uvicorn main:app --reload
```

### Run the frontend

```bash
cd frontend
npm install
npm run dev
```

## 🧠 Model Architecture

```
Network Traffic
        │
        ▼
 Feature Extraction
        │
 ┌──────────────┐
 │ Autoencoder  │
 └──────────────┘
        │
 ┌──────────────┐
 │ LightGBM     │
 └──────────────┘
        │
        ▼
  Meta-Model
        │
        ▼
 Final Prediction
```

## 📡 API Endpoints

| Method | Endpoint              | Description             |
| ------ | --------------------- | ----------------------- |
| GET    | `/api/health`         | Health check            |
| POST   | `/api/predict`        | Predict network traffic |
| POST   | `/api/explain`        | Explain prediction      |
| POST   | `/api/retrain`        | Retrain the model       |
| GET    | `/api/alerts/history` | Alert history           |

## 🧪 Training

Train the model using the default dataset:

```bash
python model.py
```

Train using a custom dataset:

```bash
python model.py --data path/to/dataset.csv
```

## 📊 Dashboard

The React dashboard provides:

* Real-time intrusion alerts
* Prediction interface
* System metrics
* Model status
* Live traffic monitoring

> Add screenshots or GIFs of your dashboard here.

## 🤝 Contributing

Contributions are welcome. Feel free to open an issue or submit a pull request.

## 📄 License

This project was developed as a Final Year Project for academic purposes.
