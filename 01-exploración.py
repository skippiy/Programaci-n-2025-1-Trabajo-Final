import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Cargar el dataset
df = pd.read_csv("AirQualityUCI_5faltantes.csv", sep=';', decimal=',')

# Eliminar columnas completamente vacías
df.dropna(axis=1, how='all', inplace=True)

# Eliminar filas completamente vacías
df.dropna(axis=0, how='all', inplace=True)

# Mostrar primeras filas
print("Primeras 5 filas del dataset:")
print(df.head())

# Información general del dataset
print("\nInformación del dataset:")
print(df.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())

# Revisión de datos faltantes
print("\nCantidad de datos faltantes por columna:")
print(df.isnull().sum())

# Revisión de valores -200 (marcadores de datos faltantes)
print("\nValores -200 por columna:")
print((df == -200).sum())

# Reemplazar -200 por NaN
df.replace(-200, np.nan, inplace=True)

# Histograma de las variables numéricas
df.hist(bins=30, figsize=(16, 12), edgecolor='black')
plt.suptitle("Distribuciones de las variables numéricas", fontsize=16)
plt.tight_layout()
plt.show()

# Mapa de calor de correlación
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Mapa de correlación de variables numéricas")
plt.show()

# Diagrama de pares para algunas variables
variables = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
df_pairs = df[variables].dropna()

print("\nGenerando diagrama de pares para variables seleccionadas...")
sns.pairplot(df_pairs)
plt.show()
