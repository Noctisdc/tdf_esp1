import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import SnowballStemmer

st.set_page_config(page_title="Analizador TF-IDF", layout="wide")

stemmer = SnowballStemmer("spanish")

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)
    tokens = [t for t in texto.split() if len(t) > 1]
    return [stemmer.stem(t) for t in tokens]

st.title("Sistema de Recuperación de Información")

st.sidebar.header("Configuración del Motor")
umbral = st.sidebar.slider("Umbral mínimo de relevancia", 0.0, 1.0, 0.05, 0.01)
max_res = st.sidebar.number_input("Límite de resultados a mostrar", 1, 20, 5)
usar_bigramas = st.sidebar.checkbox("Incluir bigramas (pares de palabras)")

rango_ngramas = (1, 2) if usar_bigramas else (1, 1)

tab_buscar, tab_datos, tab_matriz = st.tabs(["Buscador", "Gestión de Datos", "Análisis del Modelo"])

with tab_datos:
    st.subheader("Base de Conocimiento")
    docs_texto = ""
    archivo = st.file_uploader("Cargar corpus desde archivo (.txt)", type=["txt"])
    
    if archivo is not None:
        docs_texto = archivo.getvalue().decode("utf-8")
    else:
        docs_texto = st.text_area(
            "O ingresa los documentos manualmente:", 
            "El microcontrolador ESP32 permite conexión a redes.\nLa telemetría viaja por el protocolo MQTT hacia el servidor.\nLos sensores DHT22 leen variables físicas del entorno.\nStreamlit sirve para crear dashboards interactivos.\nMQTT es ligero y óptimo para arquitectura IoT.", 
            height=250
        )

with tab_buscar:
    st.subheader("Consulta")
    consulta = st.text_input("Ingresa los términos a buscar:")
    ejecutar = st.button("Buscar en documentos")

if ejecutar or consulta:
    documentos = [d.strip() for d in docs_texto.split("\n") if d.strip()]
    
    if documentos and consulta:
        vectorizador = TfidfVectorizer(tokenizer=limpiar_texto, min_df=1, ngram_range=rango_ngramas)
        matriz = vectorizador.fit_transform(documentos)
        
        vec_consulta = vectorizador.transform([consulta])
        similitudes = cosine_similarity(vec_consulta, matriz).flatten()
        
        resultados = []
        for i, score in enumerate(similitudes):
            if score >= umbral:
                resultados.append({
                    "ID": f"Doc {i+1}", 
                    "Relevancia": score, 
                    "Contenido": documentos[i]
                })
                
        df_resultados = pd.DataFrame(resultados)
        
        with tab_buscar:
            if not df_resultados.empty:
                df_resultados = df_resultados.sort_values(by="Relevancia", ascending=False).head(max_res)
                st.write("---")
                for _, fila in df_resultados.iterrows():
                    st.write(f"**{fila['ID']}** | Relevancia: {fila['Relevancia']:.4f}")
                    st.write(f"{fila['Contenido']}")
                    st.write("---")
            else:
                st.write("No se encontraron coincidencias suficientes para esta consulta.")

        with tab_matriz:
            st.subheader("Métricas Internas")
            col1, col2 = st.columns(2)
            col1.metric("Total Documentos Procesados", len(documentos))
            col2.metric("Tamaño del Vocabulario (Tokens)", len(vectorizador.get_feature_names_out()))
            
            st.write("Matriz de Pesos (TF-IDF)")
            df_matriz = pd.DataFrame(
                matriz.toarray(),
                columns=vectorizador.get_feature_names_out(),
                index=[f"Doc {i+1}" for i in range(len(documentos))]
            )
            st.dataframe(df_matriz, use_container_width=True)
