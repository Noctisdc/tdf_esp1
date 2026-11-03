import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import SnowballStemmer

st.set_page_config(page_title="Motor de Búsqueda TF-IDF", layout="wide")

st.title("Motor de Búsqueda y Análisis de Textos (TF-IDF)")

stemmer = SnowballStemmer("spanish")

def procesar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)
    tokens = [t for t in texto.split() if len(t) > 1]
    return [stemmer.stem(t) for t in tokens]

st.sidebar.header("Parámetros del Buscador")
umbral_similitud = st.sidebar.slider("Umbral mínimo de similitud", 0.0, 1.0, 0.05, 0.01)
cantidad_resultados = st.sidebar.number_input("Top K resultados", min_value=1, max_value=10, value=3)

docs_base = """Los microcontroladores ESP32 integran wifi y bluetooth para proyectos conectados.
La arquitectura de la nube permite procesar datos telemétricos en tiempo real.
Los sensores DHT22 miden con precisión la temperatura y humedad del entorno.
El desarrollo de dashboards con Streamlit facilita la visualización de datos.
MQTT es un protocolo ligero ideal para la comunicación entre dispositivos físicos.
Las bases de datos de series de tiempo como InfluxDB son óptimas para almacenar métricas."""

col_izq, col_der = st.columns([1.2, 1])

with col_izq:
    entrada_documentos = st.text_area("Base de conocimiento (un documento por línea):", docs_base, height=220)

with col_der:
    consulta_usuario = st.text_input("Escribe tu consulta:", "¿Qué dispositivo mide la temperatura?")
    opciones_rapidas = [
        "Ninguna",
        "¿Qué dispositivo mide la temperatura?",
        "¿Cómo funciona el protocolo MQTT?",
        "¿Para qué sirven las bases de datos?"
    ]
    seleccion_rapida = st.selectbox("O usa una consulta de prueba:", opciones_rapidas)

if seleccion_rapida != "Ninguna":
    consulta_usuario = seleccion_rapida

if st.button("Ejecutar Análisis TF-IDF", type="primary"):
    lista_docs = [d.strip() for d in entrada_documentos.split("\n") if d.strip()]

    if len(lista_docs) > 0 and consulta_usuario.strip():
        vectorizador = TfidfVectorizer(tokenizer=procesar_texto, min_df=1)
        matriz_tfidf = vectorizador.fit_transform(lista_docs)

        vector_consulta = vectorizador.transform([consulta_usuario])
        similitudes = cosine_similarity(vector_consulta, matriz_tfidf).flatten()

        datos_resultados = []
        for i, puntaje in enumerate(similitudes):
            datos_resultados.append({
                "Documento": lista_docs[i], 
                "Similitud": puntaje,
                "ID": f"Doc {i+1}"
            })

        df_res = pd.DataFrame(datos_resultados).sort_values(by="Similitud", ascending=False)
        df_filtrado = df_res[df_res["Similitud"] >= umbral_similitud].head(cantidad_resultados)

        st.markdown("### Resultados de la Búsqueda")
        if not df_filtrado.empty:
            for idx, fila in df_filtrado.iterrows():
                st.write(f"**{fila['ID']}** ({fila['Similitud']:.3f}) - {fila['Documento']}")
            
            st.bar_chart(df_res.set_index("ID")["Similitud"])
        else:
            st.write("No se encontraron documentos que superen el umbral establecido.")

        st.markdown("### Matriz de Pesos")
        df_matriz = pd.DataFrame(
            matriz_tfidf.toarray(),
            columns=vectorizador.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(lista_docs))]
        )
        st.dataframe(df_matriz)
    else:
        st.write("Asegúrate de tener al menos un documento y una consulta escrita.")
