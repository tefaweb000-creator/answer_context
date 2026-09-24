"""
📔 Mi Diario — Diario emocional con análisis de sentimiento
Streamlit + TextBlob + TF-IDF + WordCloud + Lottie

Instalación:
    pip install -r requirements.txt

Ejecución:
    streamlit run app.py
"""

import os
import re
import io
import json
from datetime import datetime
from collections import Counter

import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import SnowballStemmer

try:
    from streamlit_lottie import st_lottie
    LOTTIE_DISPONIBLE = True
except ImportError:
    LOTTIE_DISPONIBLE = False


# ─────────────────────────────────────────────
# CONFIGURACIÓN — pega aquí tus URLs de LottieFiles (.json)
# Búscalas en lottiefiles.com y copia el link "Lottie JSON"
# ─────────────────────────────────────────────
LOTTIE_ESCRITURA = "https://lottie.host/4f0a4d0e-0000-0000-0000-000000000000/journal.json"   # animación de "escribiendo / diario"
LOTTIE_FELIZ     = "https://lottie.host/00000000-0000-0000-0000-000000000000/sunny.json"      # animación soleada / feliz
LOTTIE_TRISTE    = "https://lottie.host/11111111-0000-0000-0000-000000000000/rain-cozy.json"  # animación lluvia cozy / calma

DATA_PATH = "data/diario_historial.csv"

st.set_page_config(page_title="Mi Diario", page_icon="📔", layout="wide", initial_sidebar_state="collapsed")


# ─────────────────────────────────────────────
# CLASIFICACIÓN DE ÁNIMO Y PALETAS
# ─────────────────────────────────────────────
# Cada rango define: etiqueta, emoji, color de fondo, color de acento, color de tarjeta
RANGOS_ANIMO = [
    (0.5, 1.01,   "Muy positivo",  "☀️", "#fdf6e3", "#e0a721", "#fffaf0"),
    (0.1, 0.5,    "Positivo",      "🌿", "#f1f7ee", "#7fb685", "#f7fbf5"),
    (-0.1, 0.1,   "Neutral",       "🌾", "#f5f1ea", "#a68a64", "#faf7f2"),
    (-0.5, -0.1,  "Bajo de ánimo", "🌧️", "#eef1f6", "#7c93b0", "#f4f6f9"),
    (-1.01, -0.5, "Muy bajo",      "🌙", "#e9ebf2", "#68769b", "#eff1f6"),
]


def clasificar_animo(polaridad):
    for lo, hi, etiqueta, emoji, fondo, acento, tarjeta in RANGOS_ANIMO:
        if lo <= polaridad < hi:
            return {"etiqueta": etiqueta, "emoji": emoji, "fondo": fondo, "acento": acento, "tarjeta": tarjeta}
    return {"etiqueta": "Neutral", "emoji": "🌾", "fondo": "#f5f1ea", "acento": "#a68a64", "tarjeta": "#faf7f2"}


def animo_actual():
    """Determina el tema de color según la última entrada guardada (o neutral si no hay historial)."""
    df = cargar_historial()
    if df.empty:
        return clasificar_animo(0.0)
    return clasificar_animo(float(df.iloc[-1]["polaridad"]))


# ─────────────────────────────────────────────
# PERSISTENCIA
# ─────────────────────────────────────────────
def cargar_historial():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=["fecha", "texto", "polaridad", "subjetividad", "estado"])
    return pd.read_csv(DATA_PATH)


def guardar_entrada(texto, polaridad, subjetividad, estado):
    df = cargar_historial()
    nueva = pd.DataFrame([{
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "texto": texto,
        "polaridad": polaridad,
        "subjetividad": subjetividad,
        "estado": estado,
    }])
    df = pd.concat([df, nueva], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)
    return df


# ─────────────────────────────────────────────
# LOTTIE
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def cargar_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def mostrar_lottie(url, alto=180, key=None):
    if not LOTTIE_DISPONIBLE:
        return
    datos = cargar_lottie(url)
    if datos:
        st_lottie(datos, height=alto, key=key)


# ─────────────────────────────────────────────
# ESTILOS DINÁMICOS SEGÚN ÁNIMO
# ─────────────────────────────────────────────
def aplicar_estilos(tema):
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Literata:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

        .stApp {{
            background-color: {tema['fondo']};
            transition: background-color 0.6s ease;
        }}

        h1, h2, h3 {{
            font-family: 'Literata', serif !important;
            color: #2d2a26 !important;
        }}
        p, li, label, .stMarkdown {{ color: #4a453f !important; }}

        textarea {{
            background-color: {tema['tarjeta']} !important;
            border: 1px solid {tema['acento']}55 !important;
            border-radius: 14px !important;
            color: #2d2a26 !important;
            font-family: 'Literata', serif !important;
            font-size: 1.02rem !important;
        }}
        textarea:focus {{
            border-color: {tema['acento']} !important;
            box-shadow: 0 0 0 3px {tema['acento']}22 !important;
        }}

        .stButton > button, [data-testid="stDownloadButton"] button {{
            background: {tema['acento']} !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 24px !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.6rem !important;
            transition: opacity 0.2s ease !important;
        }}
        .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {{ opacity: 0.85 !important; }}

        .diario-card {{
            background: {tema['tarjeta']};
            border: 1px solid {tema['acento']}33;
            border-radius: 18px;
            padding: 26px 30px;
            margin-bottom: 18px;
            box-shadow: 0 2px 14px rgba(0,0,0,0.05);
        }}

        .mood-badge {{
            display: inline-flex; align-items: center; gap: 8px;
            background: {tema['acento']}22;
            border: 1px solid {tema['acento']}55;
            color: #2d2a26;
            border-radius: 20px;
            padding: 6px 16px;
            font-weight: 600;
            font-size: 0.95rem;
        }}

        .entrada-row {{
            background: {tema['tarjeta']};
            border: 1px solid {tema['acento']}33;
            border-left: 4px solid {tema['acento']};
            border-radius: 10px;
            padding: 12px 18px;
            margin-bottom: 10px;
        }}

        [data-testid="stTabs"] button {{ font-family: 'Literata', serif !important; font-size: 1rem !important; }}
        [data-baseweb="tab-highlight"] {{ background-color: {tema['acento']} !important; }}

        hr {{ border-color: {tema['acento']}44 !important; }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SENTIMIENTO
# ─────────────────────────────────────────────
def analizar_sentimiento(texto):
    try:
        texto_en = GoogleTranslator(source="es", target="en").translate(texto)
    except Exception:
        texto_en = texto
    blob = TextBlob(texto_en)
    return round(blob.sentiment.polarity, 3), round(blob.sentiment.subjectivity, 3)


# ─────────────────────────────────────────────
# BOT TF-IDF (busca la entrada más parecida a la pregunta)
# ─────────────────────────────────────────────
stemmer = SnowballStemmer("spanish")

def tokenizar_stem(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    tokens = [t for t in texto.split() if len(t) > 1]
    return [stemmer.stem(t) for t in tokens]


def responder_pregunta(pregunta, documentos):
    vectorizer = TfidfVectorizer(tokenizer=tokenizar_stem, min_df=1)
    X = vectorizer.fit_transform(documentos)
    q_vec = vectorizer.transform([pregunta])
    similitudes = cosine_similarity(q_vec, X).flatten()
    idx = similitudes.argmax()
    return idx, similitudes[idx]


# ─────────────────────────────────────────────
# NUBE DE PALABRAS
# ─────────────────────────────────────────────
STOPWORDS_ES = {
    "de","la","el","en","y","a","los","del","se","las","un","por","con","no","una","su",
    "para","es","al","lo","como","mas","pero","sus","le","ya","o","este","si","porque",
    "esta","entre","cuando","muy","sin","sobre","tambien","me","hasta","hay","donde",
    "quien","desde","nos","durante","ni","contra","ese","eso","ante","bajo","tras",
    "que","fue","son","han","ha","ser","era","estan","siendo","sido","he","has","hemos",
    "hoy","dia","siento","sentí","estoy","fui","era","ser","tengo","tuve","fue",
}

def generar_nube(texto, color_acento):
    sw = set(STOPWORDS) | STOPWORDS_ES
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    palabras = [p for p in texto.split() if p not in sw and len(p) > 2]
    texto_limpio = " ".join(palabras)
    if not texto_limpio.strip():
        return None, None

    def color_func(*args, **kwargs):
        return color_acento

    wc = WordCloud(width=900, height=420, background_color="white",
                    color_func=color_func, collocations=False,
                    max_words=60, prefer_horizontal=0.8).generate(texto_limpio)
    df_freq = pd.DataFrame(Counter(palabras).most_common(15), columns=["Palabra", "Frecuencia"])
    return wc, df_freq


# ─────────────────────────────────────────────
# TEMA ACTUAL (se recalcula tras cada guardado)
# ─────────────────────────────────────────────
tema = animo_actual()
aplicar_estilos(tema)

# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────
col_txt, col_lottie = st.columns([3, 1])
with col_txt:
    st.markdown(f"""
    <div class="diario-card">
        <h1 style="margin:0; font-size:2rem;">📔 Mi Diario</h1>
        <p style="margin:6px 0 0 0; font-size:1rem;">Un espacio tranquilo para escribir cómo te sientes hoy.</p>
        <span class="mood-badge">{tema.get('emoji','🌾')} Ánimo actual: {tema.get('etiqueta','Neutral')}</span>
    </div>
    """, unsafe_allow_html=True)
with col_lottie:
    mostrar_lottie(LOTTIE_ESCRITURA, alto=140, key="header_lottie")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_escribir, tab_historial, tab_bot, tab_nube = st.tabs(
    ["✍️ Escribir", "📖 Historial", "🤖 Pregúntale a tu diario", "☁️ Palabras más usadas"]
)

# ── TAB: ESCRIBIR ──
with tab_escribir:
    st.markdown('<div class="diario-card">', unsafe_allow_html=True)
    st.subheader("¿Cómo te sientes hoy?")
    texto = st.text_area("Escribe libremente:", height=180,
                          placeholder="Hoy me sentí...", label_visibility="collapsed")
    guardar = st.button("💾 Guardar en el diario")
    st.markdown('</div>', unsafe_allow_html=True)

    if guardar:
        if not texto.strip():
            st.warning("Escribe algo antes de guardar.")
        else:
            with st.spinner("Analizando tu entrada..."):
                polaridad, subjetividad = analizar_sentimiento(texto)
                estado = clasificar_animo(polaridad)
                cargar_historial()  # asegura carpeta/archivo
                guardar_entrada(texto, polaridad, subjetividad, estado["etiqueta"])

            st.markdown(f"""
            <div class="diario-card">
                <span class="mood-badge">{estado['emoji']} {estado['etiqueta']}</span>
                <p style="margin-top:14px;">Polaridad: <b>{polaridad}</b> &nbsp;·&nbsp; Subjetividad: <b>{subjetividad}</b></p>
            </div>
            """, unsafe_allow_html=True)

            lottie_resultado = LOTTIE_FELIZ if polaridad >= 0 else LOTTIE_TRISTE
            mostrar_lottie(lottie_resultado, alto=160, key="resultado_lottie")

            st.success("Tu entrada quedó guardada en el historial. Recarga la página para ver el diario con el nuevo tono de ánimo. 🌿")

# ── TAB: HISTORIAL ──
with tab_historial:
    df_hist = cargar_historial()
    if df_hist.empty:
        st.info("Todavía no tienes entradas. Escribe la primera en la pestaña ✍️ Escribir.")
    else:
        st.subheader("Tu evolución emocional")
        df_plot = df_hist.copy()
        df_plot["fecha"] = pd.to_datetime(df_plot["fecha"])
        st.line_chart(df_plot.set_index("fecha")["polaridad"])

        st.subheader("Entradas anteriores")
        for _, fila in df_hist.iloc[::-1].iterrows():
            estado_fila = clasificar_animo(float(fila["polaridad"]))
            resumen = fila["texto"][:160] + ("..." if len(fila["texto"]) > 160 else "")
            st.markdown(f"""
            <div class="entrada-row">
                <b>{fila['fecha']}</b> &nbsp;·&nbsp; {estado_fila['emoji']} {fila['estado']}
                <p style="margin:6px 0 0 0;">{resumen}</p>
            </div>
            """, unsafe_allow_html=True)

        st.download_button("⬇️ Descargar historial (.csv)",
                            data=df_hist.to_csv(index=False).encode("utf-8"),
                            file_name="diario_historial.csv", mime="text/csv")

        with st.expander("⚠️ Borrar todo el historial"):
            if st.button("Borrar historial definitivamente"):
                os.remove(DATA_PATH)
                st.rerun()

# ── TAB: BOT ──
with tab_bot:
    df_hist = cargar_historial()
    st.subheader("Pregúntale a tu diario")
    st.caption("Busca, entre tus entradas anteriores, la que más se relacione con tu pregunta.")
    if len(df_hist) < 1:
        st.info("Escribe al menos una entrada para poder usar esta sección.")
    else:
        pregunta = st.text_input("Tu pregunta:", placeholder="¿Cuándo me sentí más feliz este mes?")
        if st.button("🔍 Buscar en mi diario") and pregunta.strip():
            idx, score = responder_pregunta(pregunta, df_hist["texto"].tolist())
            fila = df_hist.iloc[idx]
            estado_fila = clasificar_animo(float(fila["polaridad"]))
            st.markdown(f"""
            <div class="diario-card">
                <span class="mood-badge">{estado_fila['emoji']} {fila['fecha']}</span>
                <p style="margin-top:12px;">{fila['texto']}</p>
            </div>
            """, unsafe_allow_html=True)
            if score < 0.05:
                st.caption("Coincidencia baja — quizás no tengas una entrada muy relacionada todavía.")

# ── TAB: NUBE DE PALABRAS ──
with tab_nube:
    df_hist = cargar_historial()
    st.subheader("Palabras que más se repiten en tu diario")
    if df_hist.empty:
        st.info("Escribe algunas entradas para generar tu nube de palabras.")
    else:
        texto_total = " ".join(df_hist["texto"].astype(str).tolist())
        wc, df_freq = generar_nube(texto_total, tema["acento"])
        if wc is None:
            st.warning("No hay suficientes palabras para generar la nube todavía.")
        else:
            fig, ax = plt.subplots(figsize=(9, 4.2))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            st.download_button("⬇️ Descargar nube (.png)", data=buf.getvalue(),
                                file_name="nube_diario.png", mime="image/png")

            st.subheader("Top 15 palabras")
            for _, fila in df_freq.iterrows():
                st.markdown(f"**{fila['Palabra']}** — {fila['Frecuencia']}")
            plt.close(fig)
