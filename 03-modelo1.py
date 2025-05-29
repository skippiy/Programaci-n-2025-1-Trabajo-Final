import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Cargar los datos preprocesados
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

# Entrenar modelo Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Evaluación del modelo Random Forest:")
print(f"Error cuadrático medio (MSE): {mse:.3f}")
print(f"Error absoluto medio (MAE): {mae:.3f}")
print(f"Coeficiente de determinación (R²): {r2:.3f}")

# Curva de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='r2',
    n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label="R² entrenamiento")
plt.plot(train_sizes, test_scores_mean, label="R² validación")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Puntaje R²")
plt.title("Curva de aprendizaje – Random Forest")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
