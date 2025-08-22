# Sentiment-Analysis-Product-Reviews
Análisis de sentimientos en reseñas de productos utilizando machine learning y NLP con Python.
<br>

Análisis de Sentimientos de Reseñas de Productos
Este proyecto es un ejemplo práctico de Análisis de Sentimientos aplicado a reseñas de productos. Utiliza técnicas de procesamiento de lenguaje natural (NLP) y aprendizaje automático (Machine Learning) para clasificar automáticamente el tono de una reseña como positiva o negativa.

Características Principales
Preprocesamiento de Texto: Limpieza de datos textuales, incluyendo la eliminación de puntuación y stop words (palabras comunes que no aportan significado).

Análisis Exploratorio de Datos (EDA): Visualización de la distribución de sentimientos y las palabras más comunes, tanto en el conjunto de datos completo como por cada categoría.

Modelado Predictivo: Construcción de un modelo de Machine Learning con un clasificador de Naive Bayes Multinomial, ideal para tareas de clasificación de texto.

Vectorización de Texto: Conversión del texto limpio a un formato numérico que el modelo puede entender, usando el método TF-IDF (Term Frequency-Inverse Document Frequency) para dar mayor peso a las palabras más relevantes.

Evaluación del Modelo: Medición del rendimiento del modelo con métricas estándar como la Matriz de Confusión y el Informe de Clasificación, que muestran la precisión, recall y F1-score del modelo.

Funcionalidad de Predicción: Una función que permite predecir el sentimiento de una nueva reseña de forma individual, demostrando la aplicación práctica del modelo entrenado.

Tecnologías Utilizadas
Python: El lenguaje de programación principal.

Pandas: Para la manipulación y análisis de datos en tablas.

NumPy: Para operaciones numéricas eficientes.

Scikit-learn: La biblioteca de Machine Learning para la vectorización de texto, el entrenamiento del modelo y la evaluación.

Matplotlib y Seaborn: Para la creación de visualizaciones y gráficos (como la matriz de confusión y la distribución de sentimientos).

WordCloud: Para generar nubes de palabras que ofrecen una representación visual de la frecuencia de las palabras.

Re (Regex): Para operaciones de búsqueda y manipulación de texto con expresiones regulares.
