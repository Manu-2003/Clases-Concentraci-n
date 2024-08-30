""""" 
Importación de librerias
"""

import polars as pl # Lectura y Transformación de Datos
import matplotlib.pyplot as plt # Visualización de Datos
import seaborn as sns
import numpy as np # Manejo de arreglos y operaciones de vectores

""""
Lectura de la base de datos
"""

# Establecemos la ruta de la base
ruta = r'C:\Users\Manuel Montufar\Documents\ProyectosManu\Escuela\Concentración\Clase Uresti\Implementación\Vehiculos.csv'
# Definimos el data frame y leemos la base
df = pl.read_csv(f'{ruta}')
#Imprimimos el data frame y el tamaño de la base
print(df)


"""""
Analisis y Transformación de Datos
"""
# Esta base ya habia sido previamente limpiada para un proyecto anterior, adjuntare el EDA de dicho trabajo

# Conteo de valores nulos por columna
nulos = df.null_count()
# Imprime el resultado
print(f"El conteo de valores nulos es: {nulos}")

# Encuentra la moda de la columna 'interior'
moda_interior = df.select(pl.col("interior").mode()).item()
print(f"La moda de la columna interior es: {moda_interior}")

# Rellena los valores nulos en la columna 'interior' con la moda
df = df.with_columns(
    pl.col("interior").fill_null(moda_interior)
)

# Verifica que ya no haya valores nulos
nulos_post = df.null_count()
print(f"Lo valores nulos despues de aplicar la moda son: {nulos_post}")

# Obtiene los valores únicos por columna
valores_unicos = {col: df[col].unique().to_list() for col in df.columns}

# Imprime los valores únicos
for col, valores in valores_unicos.items():
    print(f"Columna '{col}' tiene {len(valores)} valores únicos: {valores[:10]} {'...' if len(valores) > 10 else ''}")

# Vemos estadisticas descriptivas
print(df.describe())


"""
Identificación de Outliers en las variables a elegir
"""
# Selecciona las columnas relevantes
df = df.select(["mmr", "sellingprice"])

# Asignamos el mismo tipo de Dato
mmr = df['mmr'].to_numpy().astype(float)
sellingprice = df['sellingprice'].to_numpy().astype(float)

# Función para calcular el IQR y detectar outliers
def detectar_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (data < lower_bound) | (data > upper_bound), lower_bound, upper_bound

# Identificar outliers
outliers_mmr, lower_bound_mmr, upper_bound_mmr = detectar_outliers(mmr)
outliers_sellingprice, lower_bound_sp, upper_bound_sp = detectar_outliers(sellingprice)

# Imprimir resultados
print(f"Número de outliers en 'mmr': {np.sum(outliers_mmr)}")
print(f"Límites para outliers en 'mmr': {lower_bound_mmr} (inferior) y {upper_bound_mmr} (superior)")

print(f"Número de outliers en 'sellingprice': {np.sum(outliers_sellingprice)}")
print(f"Límites para outliers en 'sellingprice': {lower_bound_sp} (inferior) y {upper_bound_sp} (superior)")

# Filtrar valores negativos en 'mmr'
valores_negativos_mmr = df.filter(pl.col("mmr") < 0)
print(f"Valores negativos en 'mmr': {valores_negativos_mmr.shape[0]}")
print(valores_negativos_mmr.head())

# Filtrar valores negativos en 'sellingprice'
valores_negativos_sellingprice = df.filter(pl.col("sellingprice") < 0)
print(f"Valores negativos en 'sellingprice': {valores_negativos_sellingprice.shape[0]}")
print(valores_negativos_sellingprice.head())

# Eliminar outliers basados en los límites
df_limpio = df.filter(
    (pl.col("mmr") >= 0) & (pl.col("mmr") <= 35100) &
    (pl.col("sellingprice") >= 0) & (pl.col("sellingprice") <= 35150)
)

print(f"Número de filas después de eliminar outliers: {df_limpio.shape[0]}")


"""
Construcción de un modelo de regresión lineal sin framework
"""

# Convertir a numpy arrays
X = df["mmr"].to_numpy().reshape(-1, 1)  # Característica
y = df["sellingprice"].to_numpy()       # Variable objetivo


# Calcular medias
x_mean = X.mean()
y_mean = y.mean()

# Restar las medias
X_centered = X - x_mean
y_centered = y - y_mean

# Calcular la pendiente (beta1) y la intersección (beta0)
numerador = (X_centered.flatten() * y_centered).sum()
denominador = (X_centered.flatten() ** 2).sum()
beta1 = numerador / denominador
beta0 = y_mean - beta1 * x_mean

# Realizar predicciones
y_pred = beta0 + beta1 * X

# Evaluar el modelo
mse = ((y - y_pred.flatten()) ** 2).mean()
ss_total = ((y - y_mean) ** 2).sum()
ss_residual = ((y - y_pred.flatten()) ** 2).sum()
r2 = 1 - (ss_residual / ss_total)

# Imprimir resultados
print(f"Error cuadrático medio (MSE): {mse}")
print(f"Coeficiente de determinación (R^2): {r2}")

""""
Visualización de los datos
"""

# Configuración de gráficos
plt.figure(figsize=(10, 6))

# Graficar los puntos de datos reales
plt.scatter(X, y, color='blue', label='Datos reales')

# Graficar la línea de regresión
plt.plot(X, y_pred, color='red', linewidth=2, label='Línea de regresión')

# Etiquetas y título
plt.xlabel('Precio de Mercado (mmr)')
plt.ylabel('Precio de Venta (sellingprice)')
plt.title('Regresión Lineal: Precio de Mercado vs. Precio de Venta')
plt.legend()

# Mostrar el gráfico
plt.show()


"""
Predicción
"""

# Definir un nuevo valor de mmr para la predicción
nuevo_mmr = float(input('Ingresa el valor de mmr que consideres: '))  # Inputa este valor con el valor que deseas predecir

# Calcular el precio de venta predicho usando la fórmula de la regresión
precio_predicho = beta0 + beta1 * nuevo_mmr

# Imprimir el resultado
print(f"El precio de venta predicho para un mmr de {nuevo_mmr} es {precio_predicho:.2f}")
