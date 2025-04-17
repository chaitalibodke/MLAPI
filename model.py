import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.pipeline import Pipeline

#load and prepare data 
def load_data():
    data = pd.read_csv('penguins.csv').dropna()
    X = data[["bill_length_mm","flipper_length_mm"]]
    y = LabelEncoder().fit_transform(data["species"])
    return train_test_split(X,y,test_size=0.2,random_state=42)

#train the model 
def train_model():
    X_train,X_test,y_train,y_test = load_data()
    model=Pipeline([
        ("scaler",StandardScaler()),
        ("knn",KNeighborsClassifier(n_neighbors=3))
    ])
    model.fit(X_train,y_train)
    return model

#save the model for later use 
model = train_model()