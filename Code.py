import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import sys

# --- Paso 2: Adquisición y Carga de Datos ---
try:
    df = pd.read_csv('D:/Proyectos Programacion/reviews.csv')
except FileNotFoundError:
    print("Error: Asegúrate de que el archivo del dataset esté en la misma carpeta que el script.")
    print("El script no puede continuar sin el archivo de datos. Por favor, corrige la ruta o el nombre del archivo.")
    sys.exit(1)

print("Primeras 5 filas del dataset:")
print(df.head())
print("\nInformación del dataset:")
print(df.info())
print("\nColumnas disponibles:", df.columns.tolist())

review_column = 'Text'
rating_column = 'Score'

if review_column not in df.columns or rating_column not in df.columns:
    print(f"\nError: Las columnas '{review_column}' o '{rating_column}' no se encontraron en el dataset.")
    print("Por favor, revisa los nombres de las columnas en tu archivo CSV y ajusta 'review_column' y 'rating_column' en el script.")
    sys.exit(1)

df['sentiment'] = df[rating_column].apply(lambda score: 'positive' if score >= 4 else ('negative' if score <= 2 else 'neutral'))
df = df[df['sentiment'] != 'neutral'].copy()
print("\nDistribución inicial de sentimientos (después de filtrar neutrales):")
print(df['sentiment'].value_counts())

# --- Paso 3: Preprocesamiento de Datos (MODIFICADO SIN NLTK) ---

df[review_column] = df[review_column].astype(str).fillna('')

# Lista de stopwords en inglés (definida manualmente o cargada desde un archivo)
# Para este ejemplo, la definimos aquí.
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now'
])

def clean_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar puntuación y caracteres especiales, y tokenizar
    # Usamos re.findall para encontrar todas las palabras y así 'tokenizar'
    words = re.findall(r'[a-z]+', text)
    # Eliminar stopwords
    words = [word for word in words if word not in stop_words]
    # Unir las palabras limpias de nuevo en una cadena
    return ' '.join(words)

# Aplicar la función de limpieza a la columna de reseñas
df['cleaned_review'] = df[review_column].apply(clean_text)

print("\nPrimeras 5 reseñas originales y limpias:")
print(df[[review_column, 'cleaned_review']].head())

# --- El resto del código es idéntico a tu script original ---
# (EDA, vectorización, entrenamiento, predicción y visualización)
# ...

# 1. Distribución de Sentimientos
plt.figure(figsize=(7, 5))
sns.countplot(x='sentiment', data=df, palette='viridis')
plt.title('Distribución de Sentimientos en las Reseñas')
plt.xlabel('Sentimiento')
plt.ylabel('Número de Reseñas')
plt.show()

# 2. Palabras Más Frecuentes en General
all_words = ' '.join(df['cleaned_review']).split()
word_freq = Counter(all_words)
print("\nLas 20 palabras más comunes en todas las reseñas:")
print(word_freq.most_common(20))

# 3. Nube de Palabras (Opcional, pero visualmente impactante)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['cleaned_review']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras de Reseñas')
plt.show()

# 4. Palabras Más Frecuentes por Sentimiento
positive_reviews = ' '.join(df[df['sentiment'] == 'positive']['cleaned_review']).split()
negative_reviews = ' '.join(df[df['sentiment'] == 'negative']['cleaned_review']).split()

positive_word_freq = Counter(positive_reviews)
negative_word_freq = Counter(negative_reviews)

print("\nLas 20 palabras más comunes en reseñas POSITIVAS:")
print(positive_word_freq.most_common(20))

print("\nLas 20 palabras más comunes en reseñas NEGATIVAS:")
print(negative_word_freq.most_common(20))

# --- Paso 5: Análisis de Sentimientos (Implementación con Machine Learning) ---
X = df['cleaned_review']
y = df['sentiment']
y_numeric = y.apply(lambda x: 1 if x == 'positive' else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)

print(f"\nTamaño del conjunto de entrenamiento: {len(X_train)} reseñas")
print(f"Tamaño del conjunto de prueba: {len(X_test)} reseñas")

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\nDimensiones de los datos vectorizados de entrenamiento: {X_train_vec.shape}")
print(f"Dimensiones de los datos vectorizados de prueba: {X_test_vec.shape}")

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print("\n--- Evaluación del Modelo de Machine Learning ---")
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# --- Paso 6: Visualización de Resultados del Modelo ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Matriz de Confusión del Modelo de Sentimientos')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

def predict_sentiment(text, model, vectorizer):
    cleaned_text = clean_text(text)
    vec_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vec_text)
    return 'Positive' if prediction == 1 else 'Negative'

new_review_positive = "This product is absolutely amazing! I love it so much, it exceeded my expectations."
new_review_negative = "Terrible product, completely useless and a waste of money. Very disappointed."

print(f"\nPredicción para reseña positiva: '{new_review_positive}' -> {predict_sentiment(new_review_positive, model, vectorizer)}")
print(f"Predicción para reseña negativa: '{new_review_negative}' -> {predict_sentiment(new_review_negative, model, vectorizer)}")
