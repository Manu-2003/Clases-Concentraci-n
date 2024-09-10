"""
Importar Librerias
"""

import pandas as pd # Lectura y Transformación de Datos
import matplotlib.pyplot as plt # Visualización de Datos
import seaborn as sns
import numpy as np # Manejo de arreglos y operaciones de vectores

from sklearn.model_selection import cross_val_score, train_test_split, KFold, learning_curve, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.ensemble import GradientBoostingRegressor

"""
Lectura de la base de datos
"""

# Establecemos la ruta de la base
ruta = r'C:\Users\Manuel Montufar\Documents\ProyectosManu\Escuela\Concentración\Clase Uresti\Implementación\Vehiculos.csv'
# Definimos el data frame y leemos la base
df = pd.read_csv(f'{ruta}')
#Imprimimos el data frame y el tamaño de la base
print(df)


""" 
Transformación 
de Datos
"""

# Conteo de valores nulos por columna
nulos = df.isnull().sum()
# Imprime el resultado
print(f"El conteo de valores nulos es: {nulos}")

# Encuentra la moda de la columna 'interior'
moda_interior = df['interior'].mode()[0]
print(f"La moda de la columna interior es: {moda_interior}")

# Rellena los valores nulos en la columna 'interior' con la moda
df.update(df[['interior']].fillna(moda_interior))

# Verifica que ya no haya valores nulos
nulos_post = df.isnull().sum()
print(f"Los valores nulos después de aplicar la moda son: {nulos_post}")

# Obtiene los valores únicos por columna
valores_unicos = {col: df[col].unique().tolist() for col in df.columns}

# Imprime los valores únicos
for col, valores in valores_unicos.items():
    print(f"Columna '{col}' tiene {len(valores)} valores únicos: {valores[:10]} {'...' if len(valores) > 10 else ''}")

# Vemos estadísticas descriptivas
print(df.describe())

""" 
Definir el target
"""
# Variables predictoras (X) y objetivo (y)
X = df[['year','mmr']]
y = df[['sellingprice']]

# Transformarlo a un array unidimensional
y = y.values.ravel()

""" 
Validación Cruzada
"""

# Definir el modelo
gbr = GradientBoostingRegressor(random_state=42)

# Definir las métricas para la validación cruzada
scoring = {
    'MSE': make_scorer(mean_squared_error, greater_is_better=False),
    'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
    'R2': make_scorer(r2_score),
    'RMSE': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False),
    'Explained Variance': make_scorer(explained_variance_score)
}

# Validación cruzada
cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for metric_name, metric in scoring.items():
    scores = cross_val_score(gbr, X, y, cv=cv, scoring=metric)
    results[metric_name] = scores.mean(), scores.std()

# Mostrar resultados de validación cruzada
results_df = pd.DataFrame(results, index=['Mean', 'Std'])
print(results_df)

""" 
Entrenamiento del
Modelo
"""

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Asegúrate de que y_train y y_test sean vectores unidimensionales
y_train = y_train.ravel()
y_test = y_test.ravel()


# Entrenar el modelo solo con los datos de entrenamiento
gbr.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = gbr.predict(X_test)

# Calcular métricas utilizando los valores de prueba
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
ev = explained_variance_score(y_test, y_pred)
rmse = np.sqrt(mse)

# Mostrar métricas
metrics = {
    'MSE': mse,
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2,
    'Explained Variance': ev
}

metrics_df = pd.DataFrame(metrics, index=[0])
print(metrics_df)

""" 
Curva de Aprendizaje
"""

# Obtener la curva de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(
    gbr, X_train, y_train, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

# Convertir los puntajes negativos de MSE a positivos
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Visualizar la curva de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Puntuación de Entrenamiento')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Puntuación de Validación')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Error Cuadrático Medio')
plt.title('Curva de Aprendizaje')
plt.legend(loc='best')
plt.grid(True)
plt.show()

""" 
Grafico de Predicción
"""

# Verificamos el tipo de dato
print(type(y_test))
print(type(y_pred))
print(y_test[:5])
print(y_pred[:5])

# Trasnformamos el tipo de dato
y_test_values = y_test.astype(float)

# Asegúrate de que y_test_values y y_pred sean numpy arrays de tipo float
y_test_values = np.array(y_test_values, dtype=float)
y_pred = np.array(y_pred, dtype=float)

# Encontrar el valor mínimo y máximo
min_val = min(np.min(y_test_values), np.min(y_pred))
max_val = max(np.max(y_test_values), np.max(y_pred))

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.plot(y_test_values, y_pred, 'o', label='Predicciones vs Valor Real')
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Línea de Igualdad')
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.title('Predicciones vs Valor Real')
plt.legend()
plt.grid(True)
plt.show()