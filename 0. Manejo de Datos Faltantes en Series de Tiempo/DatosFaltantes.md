# Tratamiento de valores faltantes en series de tiempo de ventas

Al trabajar con **series de tiempo de ventas por SKU**, es común encontrar días en los que no existen registros. Esto puede deberse a dos motivos principales:

1. **No hubo ventas ese día** → la demanda fue **0**.
2. **Datos faltantes** → errores en la captura o almacenamiento de la información.

La decisión sobre cómo manejar estos casos es clave para la calidad del pronóstico.

---

## Opciones de imputación

### 🔹 1. Llenar con Ceros
- **Recomendado para datos de ventas.**
- Refleja la realidad de que ese día hubo demanda nula.
- Evita inventar valores que no ocurrieron.
- Permite a los modelos de forecasting (ARIMA, Prophet, LSTM, etc.) capturar la intermitencia de la demanda.

df = df.asfreq("D", fill_value=0)


### 2. Imputación Univariada (con pandas)

Media/Mediana
Sustituye valores faltantes por el promedio histórico.
❌ Suaviza en exceso y puede distorsionar patrones temporales.

Forward Fill (ffill)
Rellena hacia adelante copiando el último valor conocido.
❌ Muy sensible a valores atípicos (un pico puede extenderse artificialmente).

Backward Fill (bfill)
Rellena hacia atrás copiando el siguiente valor conocido.
❌ También sensible a valores extremos.

df['sales'] = df['sales'].fillna(method='ffill')


### 🔹 3. Imputación Multivariada

Utiliza otras variables (precio, almacén, familia de producto, día de la semana, etc.) para predecir los valores faltantes mediante modelos como IterativeImputer con regresión lineal o bayesiana.

✅ Considera relaciones entre features.

❌ Más costosa computacionalmente y puede sobreajustar.

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

imp = IterativeImputer(estimator=BayesianRidge())
df_imputed = imp.fit_transform(df[['sales', 'price', 'warehouse', 'family']])


🔹 4. Imputación por Interpolación

Rellena valores intermedios usando métodos numéricos como lineal, polinómico o splines.

✅ Útil para variables continuas como precios o inventarios.

❌ En ventas discretas puede inventar fracciones irreales de unidades.

df['sales'] = df['sales'].interpolate(method='polynomial', order=2)

### ✅ Conclusión

Para el caso de pronóstico de demanda de ventas por SKU, la opción más robusta y recomendada es llenar los días sin ventas con cero.
Esto refleja fielmente la naturaleza del negocio:

Si no hubo transacción → la demanda fue nula.

Los modelos de series de tiempo necesitan estos ceros para aprender patrones de intermitencia, estacionalidad y picos de demanda.

Los métodos de imputación (univariada, multivariada o por interpolación) se reservan únicamente para situaciones donde realmente exista evidencia de datos faltantes y no de demanda nula.