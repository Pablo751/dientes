import streamlit as st
from openai import OpenAI
import pandas as pd
from typing import Optional, Dict
import os

# -------------------------------
# 1. Cargar de Forma Segura la Clave API de OpenAI
# -------------------------------

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------------
# 2. Configuración de la Aplicación Streamlit
# -------------------------------

st.set_page_config(
    page_title="💬 Chatbot de Productos Dentales",
    page_icon="🦷",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🦷 Chatbot de Productos Dentales")

# -------------------------------
# 3. Cargar y Cachear los Datos de Productos
# -------------------------------

@st.cache_data
def load_product_data(file_path: str) -> pd.DataFrame:
    """
    Cargar y preprocesar los datos de productos desde un archivo CSV.
    """
    df = pd.read_csv(file_path)
    return df

# -------------------------------
# 4. Funciones Auxiliares
# -------------------------------

def get_product_info(product_name: str, data: pd.DataFrame) -> Optional[Dict[str, str]]:
    """
    Recuperar información del producto basado en el nombre del producto.
    """
    # Buscar el producto exacto en la Descripción
    product_row = data[data['Descripción'] == product_name]
    if not product_row.empty:
        return product_row.iloc[0].to_dict()
    return None

@st.cache_data
def cached_generate_chatbot_response(product_info: Dict[str, str], user_question: str) -> str:
    """
    Generar una respuesta del chatbot usando datos cacheados.
    """
    prompt = (
        f"Eres un asistente dental especializado que ayuda a responder preguntas sobre productos dentales. "
        f"Usa únicamente la siguiente información para tu respuesta en español.\n\n"
        f"**Descripción del Producto**: {product_info['Descripción']}\n"
        f"**Instrucciones de Uso**: {product_info['Instrucciones de Uso']}\n"
        f"**Ventajas**: {product_info['Ventajas']}\n"
        f"**Presentación**: {product_info['Presentación']}\n\n"
        f"**Pregunta del Usuario**: {user_question}\n"
        f"**Respuesta**:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "Eres un asistente dental especializado y respondes siempre en español."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Ocurrió un error al procesar tu solicitud: {e}"

# -------------------------------
# 5. Cargar los Datos de Productos
# -------------------------------

uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV de productos dentales:", type=["csv"])

if uploaded_file is not None:
    try:
        product_data = load_product_data(uploaded_file)
        st.sidebar.success("¡Datos de productos dentales cargados exitosamente!")
    except Exception as e:
        st.sidebar.error(f"Error al cargar el archivo: {e}")
        st.stop()
else:
    default_file_path = 'Merged_Dental_Products.csv'
    try:
        product_data = load_product_data(default_file_path)
        st.sidebar.info("Datos de productos dentales por defecto cargados.")
    except FileNotFoundError:
        st.sidebar.error("Archivo de productos dentales por defecto no encontrado. Por favor, sube un archivo CSV.")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"Error al cargar el archivo por defecto: {e}")
        st.stop()

# -------------------------------
# 6. Sección de Interacción del Usuario
# -------------------------------

st.sidebar.header("🦷 Pregunta al Chatbot Dental")

# Extraer nombres de productos directamente de la columna Descripción
product_names = product_data['Descripción'].tolist()

# Agregar un título descriptivo al selector
selected_product = st.sidebar.selectbox(
    "Selecciona un producto dental:",
    product_names,
    format_func=lambda x: x  # Mostrar el nombre completo del producto
)

user_question = st.sidebar.text_input("Ingresa tu pregunta sobre el producto:")

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

if st.sidebar.button("Obtener Respuesta"):
    if not user_question:
        st.sidebar.warning("Por favor, ingresa una pregunta para obtener una respuesta.")
    else:
        with st.spinner("Generando respuesta..."):
            product_info = get_product_info(selected_product, product_data)
            if not product_info:
                st.sidebar.error("Información del producto dental no encontrada. Por favor, selecciona un producto válido.")
            else:
                answer = cached_generate_chatbot_response(product_info, user_question)
                st.session_state['conversation'].append((user_question, answer))
                st.sidebar.success("¡Respuesta generada!")

# Limitar el historial de conversación
MAX_HISTORY = 5
if len(st.session_state['conversation']) > MAX_HISTORY:
    st.session_state['conversation'].pop(0)

# -------------------------------
# 7. Mostrar el Historial de Conversación
# -------------------------------

if st.session_state['conversation']:
    st.header("🗨️ Historial de Conversación")
    for i, (question, answer) in enumerate(st.session_state['conversation'], 1):
        st.markdown(f"**Pregunta {i}:** {question}")
        st.markdown(f"**Respuesta {i}:** {answer}")
        st.markdown("---")
