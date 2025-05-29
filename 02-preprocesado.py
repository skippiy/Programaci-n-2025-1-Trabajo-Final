import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Cargar el dataset
df = pd.read_csv("AirQualityUCI_5faltantes.csv", sep=';', decimal=',')

# Eliminar columnas completamente vacías
df.dropna(axis=1, how='all', inplace=True)

# Eliminar filas completamente vacías
df.dropna(axis=0, how='all', inplace=True)

# Reemplazar valores -200 (indicadores de datos faltantes) por NaN
df.replace(-200, np.nan, inplace=True)

# Opcional: eliminar columnas no numéricas irrelevantes (como fecha y hora si están presentes)
columnas_no_usar = ['Date', 'Time'] if 'Date' in df.columns and 'Time' in df.columns else []
df.drop(columns=columnas_no_usar, inplace=True, errors='ignore')

# Separar variables predictoras (X) y variable objetivo (y)
# Puedes cambiar la columna objetivo según tu caso. Usaremos 'C6H6(GT)' como ejemplo.
target_column = 'C6H6(GT)'
X = df.drop(columns=[target_column])
y = df[target_column]

# Imputación de valores faltantes con la media
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y_imputed = y.fillna(y.mean())

# Normalización (escalado estándar)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_imputed, test_size=0.2, random_state=42
)

# Mostrar resultados del preprocesamiento
print("Dimensiones del conjunto de entrenamiento:", X_train.shape)
print("Dimensiones del conjunto de prueba:", X_test.shape)
print("Primeras filas de X_train:")
print(X_train.head())

# Guardar conjuntos para modelos posteriores (opcional)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Archivos preprocesados guardados como: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
