import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Cargar datos
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

# Inicializar modelos
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_knn = KNeighborsRegressor(n_neighbors=5)

# Entrenar y predecir
model_rf.fit(X_train, y_train)
model_knn.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
y_pred_knn = model_knn.predict(X_test)

# Evaluación cruzada (R² promedio)
r2_rf_cv = cross_val_score(model_rf, X_train, y_train, cv=5, scoring='r2').mean()
r2_knn_cv = cross_val_score(model_knn, X_train, y_train, cv=5, scoring='r2').mean()

# Métricas de evaluación
def get_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

metrics_rf = get_metrics(y_test, y_pred_rf)
metrics_knn = get_metrics(y_test, y_pred_knn)

# Mostrar resultados
print("Comparación de modelos:\n")
print("Random Forest:")
print(f"  MSE: {metrics_rf['MSE']:.3f}, MAE: {metrics_rf['MAE']:.3f}, R²: {metrics_rf['R2']:.3f}, R² CV: {r2_rf_cv:.3f}")
print("KNN:")
print(f"  MSE: {metrics_knn['MSE']:.3f}, MAE: {metrics_knn['MAE']:.3f}, R²: {metrics_knn['R2']:.3f}, R² CV: {r2_knn_cv:.3f}")

# Visualización de métricas
labels = ['MSE', 'MAE', 'R²']
rf_values = [metrics_rf['MSE'], metrics_rf['MAE'], metrics_rf['R2']]
knn_values = [metrics_knn['MSE'], metrics_knn['MAE'], metrics_knn['R2']]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width/2, rf_values, width, label='Random Forest')
bars2 = ax.bar(x + width/2, knn_values, width, label='KNN')

ax.set_ylabel('Valores')
ax.set_title('Comparación de métricas entre modelos')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Concordancia entre predicciones
correlacion, _ = pearsonr(y_pred_rf, y_pred_knn)
print(f"\nCoeficiente de correlación entre predicciones (Inter-model reliability): {correlacion:.3f}")
