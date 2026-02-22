# AI Finance Assistant 🤖💰

An AI-powered personal finance assistant that categorizes expenses, detects anomalies, forecasts future spending, and provides budgeting recommendations using Machine Learning.

---

## 🚀 Features

- 📤 **Upload & Manage Transactions** - CSV import with automatic processing
- 🏷️ **Smart Categorization** - ML-powered expense categorization (99% accuracy)
- 🔍 **Anomaly Detection** - Identifies unusual spending patterns
- 📈 **Spending Forecasts** - Predicts future expenses for 30+ days
- 📊 **Interactive Dashboard** - Beautiful Streamlit UI
- 🔐 **User Authentication** - Secure login system
- 🐳 **Docker Ready** - One-command deployment
- 📱 **REST API** - Full FastAPI backend

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit    │    │    FastAPI     │    │   PostgreSQL    │
│   Dashboard    │◄──►│    Backend     │◄──►│   Database     │
│   (UI)         │    │   (API)        │    │   (Storage)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Forecasting   │    │  Anomaly       │    │  Transaction   │
│  Models        │    │  Detection     │    │  Classifier   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 📂 Project Structure

```
ai-finance-assistant/
├── backend/                 # FastAPI application
│   └── app/
│       ├── main.py          # FastAPI app entry point
│       ├── models.py       # SQLAlchemy models
│       ├── database.py     # Database configuration
│       ├── auth.py         # Authentication utilities
│       ├── init_db.py      # Database initialization
│       └── routes/        # API endpoints
│           ├── users.py    # User management
│           ├── transactions.py # Transaction CRUD
│           └── forecast.py # Forecasting endpoints
├── streamlit_app/         # Streamlit dashboard
│   ├── app.py           # Main dashboard
│   ├── api_client.py     # API integration
│   └── pages/           # Multi-page components
│       ├── 1_Manage_Data.py
│       ├── 1_Upload_Data.py
│       ├── 2_expense_categorization.py
│       ├── 3_Anomaly_Detection.py
│       ├── 4_Forecasting.py
│       ├── 4_Forecasting_By_Category.py
│       └── 5_Forecast_Dashboard.py
├── src/                  # ML training & inference
│   ├── training/         # Model training scripts
│   │   ├── train_forecasting.py
│   │   ├── train_forecasting_by_category.py
│   │   ├── train_anomaly.py
│   │   └── train_classifier.py
│   ├── inference/        # Model inference scripts
│   │   ├── run_forecasting.py
│   │   ├── run_forecast_by_category.py
│   │   ├── run_anomaly_detection.py
│   │   ├── run_categorization.py
│   │   ├── detect_anomaly.py
│   │   ├── forecast_expenses.py
│   │   ├── forecast_by_category.py
│   │   ├── predict_category.py
│   │   └── quick_predict.py
│   ├── features/         # Feature engineering
│   │   ├── build_features.py
│   │   ├── extract_features.py
│   │   └── forecast_features.py
│   ├── data/            # Data processing
│   │   ├── clean_data.py
│   │   └── create_sample_data.py
│   └── reports/         # Report generation
│       ├── professional_report.py
│       ├── generate_pdf_report.py
│       ├── run_report_generation.py
│       └── run_professional_report.py
├── models/               # Trained ML models
├── data/                 # Dataset storage
│   ├── raw/             # Original data
│   └── processed/       # Processed data
├── reports/              # Generated reports
├── requirements.txt       # Python dependencies
├── docker-compose.yml    # Docker configuration
└── README.md           # This file
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- PostgreSQL (or use Docker)

### 1️⃣ Clone & Setup
```bash
git clone <repository-url>
cd ai-finance-assistant
```

### 2️⃣ Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Database Setup
```bash
# Option 1: Docker (Recommended)
docker-compose up postgres -d

# Option 2: Local PostgreSQL
psql -U postgres
CREATE USER admin WITH PASSWORD 'password';
CREATE DATABASE finance OWNER admin;
GRANT ALL PRIVILEGES ON DATABASE finance TO admin;
```

### 5️⃣ Initialize Database
```bash
cd backend
python -m app.init_db
```

---

## 🚀 Quick Start

### Option A: Docker (Recommended)
```bash
docker-compose up -d
```
Access:
- Dashboard: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option B: Local Development

**Terminal 1 - API:**
```bash
& d:/ai-finance-assistant/venv/Scripts/Activate.ps1
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Dashboard:**
```bash
& d:/ai-finance-assistant/venv/Scripts/Activate.ps1
cd streamlit_app
streamlit run app.py --server.port=8501
```

---

## 🧠 ML Models

### Transaction Categorization
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 99%
- **Features**: 33 engineered features
- **Categories**: 9 expense types

### Anomaly Detection
- **Algorithm**: Autoencoder (Neural Network)
- **Detection Rate**: 5% anomalies flagged
- **Features**: 10 financial metrics

### Spending Forecast
- **Algorithm**: Time Series (Prophet-style)
- **Horizon**: 30+ days
- **Accuracy**: Historical trend analysis

---

## 📊 Usage

### 1. Create Users & Add Transactions
- Navigate to "Manage Data" in the dashboard
- Create user accounts
- Add transactions manually or upload CSV files
- Automatic categorization

### 2. View Analytics
- Spending trends by category
- Anomaly detection results
- Future spending forecasts

### 3. API Integration
```python
from streamlit_app.api_client import FinanceAPIClient

client = FinanceAPIClient("http://localhost:8000")
transactions = client.get_transactions()
forecast = client.get_forecast(user_id=1, days=30)
```

---

## 🔧 Development

### Model Training
```bash
# Train individual models
python src/training/train_forecasting.py
python src/training/train_anomaly.py
python src/training/train_classifier.py

# Train forecasting by category
python src/training/train_forecasting_by_category.py
```

### Model Inference
```bash
# Run individual inference
python src/inference/run_forecasting.py
python src/inference/run_anomaly_detection.py
python src/inference/run_categorization.py
python src/inference/run_forecast_by_category.py
```

### Data Processing
```bash
# Create sample data
python src/data/create_sample_data.py

# Clean and process data
python src/data/clean_data.py
python src/features/build_features.py
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `backend/app/main.py` | FastAPI application |
| `streamlit_app/app.py` | Main dashboard |
| `streamlit_app/api_client.py` | API integration |
| `backend/app/init_db.py` | Database initialization |
| `backend/app/routes/users.py` | User management |
| `backend/app/routes/transactions.py` | Transaction CRUD |
| `backend/app/routes/forecast.py` | Forecasting endpoints |

---

## 🐳 Docker Services

| Service | Port | Description |
|---------|-------|-------------|
| `postgres` | 5432 | PostgreSQL database |
| `api` | 8000 | FastAPI backend |
| `dashboard` | 8501 | Streamlit UI |

---

## 🔍 API Endpoints

### Users
- `POST /users/` - Create user
- `GET /users/` - List users

### Transactions
- `POST /transactions/` - Create transaction
- `GET /transactions/` - List transactions (optional `user_id` filter)
- `GET /transactions/{id}` - Get specific transaction
- `PUT /transactions/{id}` - Update transaction
- `DELETE /transactions/{id}` - Delete transaction

### Forecast
- `GET /forecast/` - Get financial forecast (optional `user_id`, `days` params)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make your changes
4. Add tests if applicable
5. Submit pull request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 👨‍💻 Author

Built by [Geetechcell](https://github.com/Geetechcell)
AI / Machine Learning Developer

⭐ **Star this repo if it helps you!**
# finance-ai
