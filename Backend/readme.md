# Fake Job Prediction Backend

This is the backend service for the Fake Job Prediction system. It provides API endpoints for job prediction using various machine learning models (BERT, XGBoost, Logistic Regression).

## Setup Instructions

Follow these steps to set up and run the backend server:

### 1. Navigate to the Backend Directory
Open your terminal and navigate to the backend folder:
```bash
cd 26_P2_fake_job_prediction_backend
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:
```bash
python -m venv myenv
```

### 3. Activate the Virtual Environment
Activate the created environment:

**On Linux/macOS:**
```bash
source myenv/bin/activate
```

**On Windows:**
```bash
myenv\Scripts\activate
```

### 4. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 5. Run the Application
Start the Flask/FastAPI server:
```bash
python app.py
```

The server should now be running and ready to handle requests from the frontend.

## Project Structure
- `app.py`: Main entry point of the application.
- `models/`: Contains pre-trained models.
- `requirements.txt`: List of dependencies.
- `users.json`: User data storage (for demo purposes).
