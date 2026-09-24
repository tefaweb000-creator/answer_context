import base64
import datetime
import json
import re

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import requests
    from streamlit_lottie import st_lottie
    LOTTIE_OK = True
except ImportError:
    LOTTIE_OK = False

# ───────────────────────── Configuración ─────────────────────────
LOTTIE_URL = ""  # Pega aquí la URL .json de lottiefiles.com (opcional)
VISION_MODEL = "gpt-4o"
TEXT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

st.set_page_config(page_title="Apuntes inteligentes", page_icon="📚", layout="centered")

# ───────────────────────── Estilo ─────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&family=Literata:opsz,wght@7..72,500;7..72,700&family=Inter:wght@400;500;600&display=swap');

:root {
  --desk: #E4E1D8;
  --paper: #FDFCF7;
  --paper-2: #F4F1E7;
  --rule: #D6E2F3;
  --margin: #E39A9A;
  --ink: #1E2A44;
  --ink-soft: #4A5570;
  --pen: #2F4DA8;
  --pen-dark: #233B85;
  --marker: #FFE45C;
  --line: #B9C6DD;
}

/* Escritorio y hoja */
.stApp, [data-testid="stAppViewContainer"] { background: var(--desk); }
[data-testid="stHeader"] { background: transparent; }

.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 780px;
  margin: 2.5rem auto 3rem;
  padding: 3rem 2.5rem 3.5rem 5.25rem !important;
  background-color: var(--paper);
  background-image:
    linear-gradient(90deg, transparent 3.6rem, var(--margin) 3.6rem,
                    var(--margin) calc(3.6rem + 2px), transparent calc(3.6rem + 2px)),
    repeating-linear-gradient(180deg, transparent 0, transparent 31px,
                    var(--rule) 31px, var(--rule) 32px);
  border-radius: 4px 14px 14px 4px;
  box-shadow: 0 1px 0 #D2CDBF, 0 14px 32px rgba(30, 42, 68, 0.13);
}

/* Tipografía y color de texto */
.stApp p, .stApp li, .stApp td, .stApp th { color: var(--ink); }
.stApp, .stMarkdown { font-family: 'Inter', system-ui, sans-serif; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: 'Literata', Georgia, serif !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em;
}
.stMarkdown p, .stMarkdown li { line-height: 1.65; }
.stMarkdown strong {
  background: linear-gradient(transparent 58%, rgba(255, 228, 92, 0.85) 58%);
  padding: 0 0.1em;
  border-radius: 2px;
}
[data-testid="stWidgetLabel"] p { color: var(--ink) !important; font-weight: 600; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
.stCaption { color: var(--ink-soft) !important; }
hr { border-color: var(--line) !important; }

/* Encabezado de la página */
.page-date {
  font-family: 'Caveat', cursive;
  font-size: 1.35rem;
  color: var(--ink-soft);
  text-align: right;
  border-bottom: 1.5px solid var(--ink-soft);
  width: fit-content;
  margin-left: auto;
  padding: 0 0.3rem;
}
.hero-title {
  font-family: 'Caveat', cursive;
  font-weight: 700;
  font-size: clamp(2.8rem, 8vw, 3.8rem);
  line-height: 1;
  color: var(--ink);
  margin: 0.8rem 0 0.8rem;
}
.hero-sub {
  color: var(--ink-soft);
  font-size: 1rem;
  line-height: 1.6;
  max-width: 34rem;
  margin-bottom: 2rem;
}

/* Campos de texto */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="base-input"] {
  background: #FFFFFF !important;
  border-radius: 10px !important;
  border-color: var(--line) !important;
}
.stTextArea textarea, .stTextInput input {
  background: #FFFFFF !important;
  color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important;
  caret-color: var(--pen);
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
  color: #7A849A !important;
  -webkit-text-fill-color: #7A849A !important;
}

/* Subida de archivos */
[data-testid="stFileUploaderDropzone"] {
  background: var(--paper-2) !important;
  border: 1.5px dashed #9FB3D6 !important;
  border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] svg { color: var(--ink-soft) !important; fill: var(--ink-soft); }
[data-testid="stFileUploaderDropzone"] button {
  background: #FFFFFF !important;
  color: var(--ink) !important;
  border: 1.5px solid var(--ink) !important;
  border-radius: 10px !important;
}
[data-testid="stFileUploaderFile"] div,
[data-testid="stFileUploaderFile"] small { color: var(--ink) !important; }

/* Botones */
.stButton > button, .stDownloadButton > button {
  background: #FFFFFF;
  color: var(--ink);
  border: 1.5px solid var(--ink);
  border-radius: 10px;
  font-weight: 600;
  min-height: 2.75rem;
  transition: background .15s ease, transform .15s ease;
}
.stButton > button p, .stDownloadButton > button p { color: inherit !important; }
.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--paper-2);
  color: var(--ink);
  border-color: var(--ink);
  transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
  background: var(--pen);
  border-color: var(--pen);
  color: #FFFFFF;
}
.stButton > button[kind="primary"]:hover {
  background: var(--pen-dark);
  border-color: var(--pen-dark);
  color: #FFFFFF;
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
  outline: 2px solid var(--pen);
  outline-offset: 2px;
}

/* Pestañas como separadores de cuaderno */
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1.5px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: 0.2rem 0.6rem; }
.stTabs [data-baseweb="tab"] p {
  font-family: 'Caveat', cursive;
  font-size: 1.45rem;
  font-weight: 700;
  color: var(--ink-soft) !important;
}
.stTabs [aria-selected="true"] p { color: var(--ink) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--marker) !important; height: 5px; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem; }

/* Expanders, radios, métricas, avisos */
[data-testid="stExpander"] details {
  background: #FFFFFF;
  border: 1px solid var(--line) !important;
  border-radius: 10px;
}
[data-testid="stExpander"] summary p { color: var(--ink) !important; font-weight: 600; }
[data-testid="stRadio"] label p { color: var(--ink) !important; }
[data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] { color: var(--ink) !important; }
[data-testid="stAlert"] p { color: var(--ink) !important; }
[data-testid="stProgress"] p { color: var(--ink-soft) !important; }

/* Barra lateral */
[data-testid="stSidebar"] { background: var(--paper-2); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label { color: var(--ink) !important; }

/* Etiquetas de fuentes */
.source-chip {
  display: inline-block;
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--ink);
  background: var(--paper-2);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.15rem 0.6rem;
  margin: 0 0.35rem 0.35rem 0;
}

/* Móvil */
@media (max-width: 640px) {
  .block-container, [data-testid="stMainBlockContainer"] {
    margin: 0.75rem 0.4rem 2rem;
    padding: 2.25rem 1rem 2.5rem 2.9rem !important;
    background-image:
      linear-gradient(90deg, transparent 2.1rem, var(--margin) 2.1rem,
                      var(--margin) calc(2.1rem + 2px), transparent calc(2.1rem + 2px)),
      repeating-linear-gradient(180deg, transparent 0, transparent 31px,
                      var(--rule) 31px, var(--rule) 32px);
  }
  .page-date { font-size: 1.15rem; }
}

@media (prefers-reduced-motion: reduce) {
  .stButton > button, .stDownloadButton > button { transition: none; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────── Estado ─────────────────────────
DEFAULTS = {
    "sources": [],      # [{"name", "text", "type", "img", "mime"}]
    "index": None,
    "summary": "",
    "quiz": [],
    "quiz_checked": False,
    "last_answer": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_all():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v if not isinstance(v, list) else []
    clear_quiz_keys()


def clear_quiz_keys():
    for k in [k for k in st.session_state.keys() if str(k).startswith("q_")]:
        del st.session_state[k]


# ───────────────────────── Utilidades ─────────────────────────
@st.cache_data(show_spinner=False)
def load_lottie(url):
    if not url or not LOTTIE_OK:
        return None
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def extract_pdf(file):
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"name": f"{file.name}, p. {i}", "text": text, "type": "pdf"})
    return pages


def transcribe_image(client, file):
    b64 = base64.b64encode(file.getvalue()).decode("utf-8")
    mime = file.type or "image/jpeg"
    prompt = (
        "Eres un asistente que digitaliza apuntes de clase. "
        "1) Transcribe fielmente todo el texto visible, incluida la letra manuscrita, "
        "fórmulas y listas. Si algo es ilegible, márcalo como [ilegible]. "
        "2) Después, en una sección llamada 'Descripción', explica brevemente los diagramas, "
        "esquemas, flechas o elementos visuales y cómo se relacionan. "
        "Responde en español y en Markdown."
    )
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
    )
    return resp.choices[0].message.content


def build_index(sources, key):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = []
    for src in sources:
        for chunk in splitter.split_text(src["text"]):
            docs.append(Document(page_content=chunk, metadata={"source": src["name"]}))
    if not docs:
        return None
    return FAISS.from_documents(docs, OpenAIEmbeddings(model=EMBED_MODEL, api_key=key))


def all_text(sources, limit=15000):
    text = "\n\n".join(f"### {s['name']}\n{s['text']}" for s in sources)
    return text[:limit]


def llm(key, temperature=0.3):
    return ChatOpenAI(model=TEXT_MODEL, temperature=temperature, api_key=key)


def make_summary(key, sources):
    prompt = (
        "Eres un tutor universitario. Con base en el material de clase de abajo, escribe en español "
        "unos apuntes de estudio en Markdown con: un párrafo de idea general, los conceptos clave "
        "(cada uno con una explicación de 1-2 líneas) y una lista corta de cosas que probablemente "
        "pregunten en un examen. No inventes información que no esté en el material.\n\n"
        f"MATERIAL:\n{all_text(sources)}"
    )
    return llm(key).invoke(prompt).content


def make_quiz(key, sources, n=5):
    prompt = (
        f"Crea {n} preguntas de opción múltiple en español sobre el material de clase de abajo. "
        "Responde SOLO con JSON válido, sin texto adicional ni bloques de código, con esta forma:\n"
        '{"preguntas": [{"pregunta": "...", "opciones": ["...", "...", "...", "..."], '
        '"correcta": 0, "explicacion": "..."}]}\n'
        "'correcta' es el índice (0-3) de la opción correcta. Las opciones deben ser distintas entre sí.\n\n"
        f"MATERIAL:\n{all_text(sources)}"
    )
    raw = llm(key, temperature=0.5).invoke(prompt).content
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    data = json.loads(raw)
    return data.get("preguntas", [])


def ask(key, index, question):
    docs = index.similarity_search(question, k=4)
    context = "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)
    prompt = (
        "Responde la pregunta en español usando solo el contexto de los apuntes. "
        "Si la respuesta no está en el contexto, dilo claramente. "
        "Al final, indica entre corchetes de qué fuentes sacaste la respuesta.\n\n"
        f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}"
    )
    return llm(key, temperature=0).invoke(prompt).content, docs


def export_md(sources, summary):
    parts = ["# Apuntes de clase\n"]
    if summary:
        parts.append("## Resumen\n\n" + summary + "\n")
    fotos = [s for s in sources if s["type"] == "foto"]
    if fotos:
        parts.append("## Transcripciones\n")
        for s in fotos:
            parts.append(f"### {s['name']}\n\n{s['text']}\n")
    return "\n".join(parts)


# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.markdown("### Configuración")
    api_key = st.text_input("Clave de OpenAI", type="password")
    st.caption("Tu clave solo se usa durante esta sesión.")
    st.divider()
    if st.button("Empezar de nuevo", use_container_width=True):
        reset_all()
        st.rerun()

# ───────────────────────── Encabezado ─────────────────────────
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
hoy = datetime.date.today()
st.markdown(
    f'<div class="page-date">Fecha: {hoy.day} {MESES[hoy.month - 1]} {hoy.year}</div>'
    '<div class="hero-title">Apuntes de clase</div>'
    '<div class="hero-sub">Sube la lectura en PDF y fotos del tablero o de tu cuaderno. '
    "Se transcriben, se unen y puedes resumir, practicar con un quiz o hacer preguntas.</div>",
    unsafe_allow_html=True,
)

# ───────────────────────── Carga ─────────────────────────
col1, col2 = st.columns(2)
with col1:
    pdf_file = st.file_uploader("Lectura (PDF)", type="pdf")
with col2:
    image_files = st.file_uploader(
        "Fotos de apuntes", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
    )

if st.button("Procesar material", type="primary", use_container_width=True):
    if not api_key:
        st.warning("Ingresa tu clave de OpenAI en la barra lateral para procesar el material.")
    elif not pdf_file and not image_files:
        st.warning("Sube al menos un PDF o una foto para empezar.")
    else:
        anim_slot = st.empty()
        lottie = load_lottie(LOTTIE_URL)
        if lottie:
            with anim_slot:
                st_lottie(lottie, height=140, key="loading")
        progress = st.progress(0.0, text="Preparando…")

        try:
            client = OpenAI(api_key=api_key)
            sources = []
            total = (1 if pdf_file else 0) + len(image_files or []) + 1
            step = 0

            if pdf_file:
                progress.progress(step / total, text="Leyendo el PDF…")
                pdf_pages = extract_pdf(pdf_file)
                if not pdf_pages:
                    st.info("El PDF no tiene texto seleccionable (puede ser escaneado). Súbelo como fotos.")
                sources.extend(pdf_pages)
                step += 1

            for i, img in enumerate(image_files or [], start=1):
                progress.progress(step / total, text=f"Transcribiendo foto {i} de {len(image_files)}…")
                text = transcribe_image(client, img)
                sources.append({
                    "name": f"Foto {i} ({img.name})",
                    "text": text,
                    "type": "foto",
                    "img": img.getvalue(),
                })
                step += 1

            progress.progress(step / total, text="Organizando tus apuntes…")
            st.session_state.sources = sources
            st.session_state.index = build_index(sources, api_key)
            st.session_state.summary = ""
            st.session_state.quiz = []
            st.session_state.quiz_checked = False
            st.session_state.last_answer = None
            clear_quiz_keys()
            progress.progress(1.0, text="Listo")
        except Exception as e:
            st.error(f"No se pudo procesar el material: {e}")
        finally:
            anim_slot.empty()
            progress.empty()

# ───────────────────────── Estudio ─────────────────────────
sources = st.session_state.sources

if not sources:
    st.info("Cuando proceses tu material, aquí aparecerán el resumen, el quiz y las preguntas.")
    st.stop()

n_pdf = sum(1 for s in sources if s["type"] == "pdf")
n_foto = sum(1 for s in sources if s["type"] == "foto")
st.caption(f"Material listo: {n_pdf} páginas de PDF y {n_foto} fotos.")

tab_sum, tab_quiz, tab_ask, tab_tx = st.tabs(["Resumen", "Quiz", "Pregunta", "Transcripciones"])

# Resumen
with tab_sum:
    if st.button("Generar resumen", use_container_width=True):
        if not api_key:
            st.warning("Ingresa tu clave de OpenAI en la barra lateral.")
        else:
            with st.spinner("Escribiendo el resumen…"):
                try:
                    st.session_state.summary = make_summary(api_key, sources)
                except Exception as e:
                    st.error(f"No se pudo generar el resumen: {e}")
    if st.session_state.summary:
        st.markdown(st.session_state.summary)

# Quiz
with tab_quiz:
    if st.button("Generar quiz", use_container_width=True):
        if not api_key:
            st.warning("Ingresa tu clave de OpenAI en la barra lateral.")
        else:
            with st.spinner("Armando las preguntas…"):
                try:
                    clear_quiz_keys()
                    st.session_state.quiz = make_quiz(api_key, sources)
                    st.session_state.quiz_checked = False
                except json.JSONDecodeError:
                    st.error("El quiz llegó con un formato inválido. Genera uno nuevo.")
                except Exception as e:
                    st.error(f"No se pudo generar el quiz: {e}")

    quiz = st.session_state.quiz
    if quiz:
        for i, q in enumerate(quiz):
            st.markdown(f"**{i + 1}. {q['pregunta']}**")
            st.radio(" ", q["opciones"], index=None, key=f"q_{i}", label_visibility="collapsed")
            if st.session_state.quiz_checked:
                chosen = st.session_state.get(f"q_{i}")
                correct = q["opciones"][q["correcta"]]
                if chosen == correct:
                    st.success(f"Correcto. {q.get('explicacion', '')}")
                elif chosen is None:
                    st.warning(f"Sin responder. La correcta era: {correct}. {q.get('explicacion', '')}")
                else:
                    st.error(f"La correcta era: {correct}. {q.get('explicacion', '')}")
            st.write("")

        if st.button("Revisar respuestas", type="primary", use_container_width=True):
            st.session_state.quiz_checked = True
            st.rerun()

        if st.session_state.quiz_checked:
            score = sum(
                1 for i, q in enumerate(quiz)
                if st.session_state.get(f"q_{i}") == q["opciones"][q["correcta"]]
            )
            st.metric("Puntaje", f"{score} / {len(quiz)}")

# Pregunta
with tab_ask:
    question = st.text_area("Pregunta sobre tus apuntes", placeholder="¿Qué diferencia hay entre…?")
    if st.button("Responder", use_container_width=True):
        if not api_key:
            st.warning("Ingresa tu clave de OpenAI en la barra lateral.")
        elif not question.strip():
            st.warning("Escribe una pregunta primero.")
        elif st.session_state.index is None:
            st.warning("No hay texto indexado. Procesa el material de nuevo.")
        else:
            with st.spinner("Buscando en tus apuntes…"):
                try:
                    st.session_state.last_answer = ask(api_key, st.session_state.index, question)
                except Exception as e:
                    st.error(f"No se pudo responder: {e}")

    if st.session_state.last_answer:
        answer, docs = st.session_state.last_answer
        st.markdown(answer)
        with st.expander("Ver fragmentos usados"):
            for d in docs:
                st.markdown(f'<span class="source-chip">{d.metadata["source"]}</span>', unsafe_allow_html=True)
                st.caption(d.page_content)

# Transcripciones
with tab_tx:
    fotos = [(i, s) for i, s in enumerate(sources) if s["type"] == "foto"]
    if not fotos:
        st.info("No subiste fotos. Aquí verás el texto extraído de cada una.")
    else:
        st.caption("Corrige lo que la IA haya leído mal y guarda para actualizar tus apuntes.")
        for idx, s in fotos:
            with st.expander(s["name"], expanded=False):
                st.image(s["img"], use_container_width=True)
                st.text_area("Transcripción", value=s["text"], key=f"tx_{idx}", height=260)

        if st.button("Guardar cambios", use_container_width=True):
            if not api_key:
                st.warning("Ingresa tu clave de OpenAI en la barra lateral.")
            else:
                for idx, _ in fotos:
                    st.session_state.sources[idx]["text"] = st.session_state[f"tx_{idx}"]
                with st.spinner("Actualizando tus apuntes…"):
                    try:
                        st.session_state.index = build_index(st.session_state.sources, api_key)
                        st.session_state.summary = ""
                        st.session_state.quiz = []
                        st.session_state.quiz_checked = False
                        clear_quiz_keys()
                        st.success("Cambios guardados. Genera de nuevo el resumen o el quiz si los necesitas.")
                    except Exception as e:
                        st.error(f"No se pudieron guardar los cambios: {e}")

# ───────────────────────── Descarga ─────────────────────────
st.divider()
st.download_button(
    "Descargar apuntes (.md)",
    data=export_md(sources, st.session_state.summary),
    file_name="apuntes.md",
    mime="text/markdown",
    use_container_width=True,
)
