# MLAPI
ML model trained on penguin dataset with fast api application 

Step 1: Install Required Libraries
Ensure you have FastAPI and Uvicorn installed. You might also need libraries for your machine learning model (e.g., TensorFlow, PyTorch, scikit-learn). Install them using pip:

bash
pip install fastapi uvicorn
pip install <your-model-library>
Step 2: Load Your Trained Model
Load the model you want to deploy in your Python code. For example:

python
from sklearn.externals import joblib

# Load pre-trained model
model = joblib.load("model.pkl")  # Replace with your model file
Step 3: Create the FastAPI App
Set up the FastAPI application and define an endpoint for making predictions:

python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define request schema
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.post("/predict/")
def predict(data: InputData):
    input_features = [[data.feature1, data.feature2, data.feature3]]
    prediction = model.predict(input_features)
    return {"prediction": prediction.tolist()}
Step 4: Start the API Server
Run your FastAPI application using Uvicorn:

bash
uvicorn main:app --reload
Here:

main refers to your Python file name (e.g., main.py).

app is the FastAPI instance.

Step 5: Test the API
You can test your API using tools like Postman or Curl. For example:

bash
curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d '{"feature1": 1.2, "feature2": 3.4, "feature3": 5.6}'
Step 6: Deploy to Production
To deploy:

Use Docker for containerization and ensure portability.

Or deploy on cloud platforms like AWS, Azure, or Google Cloud for scalability and reliability.

A Docker example:

Create a Dockerfile:

dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install fastapi uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
Build and run the container:

bash
docker build -t fastapi-app .
docker run -p 8000:80 fastapi-app
