import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns=["quality"])

logged_model = "runs:/4c9b1fdf9e5f42e0946f37ff7db831de/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(x)
print(y)
