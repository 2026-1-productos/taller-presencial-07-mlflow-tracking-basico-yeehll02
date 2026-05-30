import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns=["quality"])

## Debe verificarse el run_id del modelo que se quiere cargar
## Se puede obtener el run_id desde la interfaz de MLflow
logged_model = "runs:/2716ff0c619a4f19b11beb376016a7a8/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(x)

print(y)
