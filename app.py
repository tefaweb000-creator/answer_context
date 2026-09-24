import base64
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
@import url('https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,500;7..72,700&family=Inter:wght@400;500;600&display=swap');

:root {
  --ink: #161B26;
  --surface: #1F2533;
  --line: #2C3445;
  --paper: #E8E4DA;
  --muted: #8A93A6;
  --marker: #F2D35B;
}

html, body, [class*="css"], .stMarkdown, p, label, li {
  font-family: 'Inter', system-ui, sans-serif;
}
h1, h2, h3 {
  font-family: 'Literata', Georgia, serif !important;
  color: var(--paper) !important;
  letter-spacing: -0.01em;
}

.hero-title {
  font-family: 'Literata', Georgia, serif;
  font-size: clamp(2.2rem, 6vw, 3.1rem);
  font-weight: 700;
  line-height: 1.1;
  color: var(--paper);
  margin: 0.4rem 0 0.6rem 0;
}
.hero-title .mark {
  background: linear-gradient(transparent 58%, rgba(242, 211, 91, 0.55) 58%);
  padding: 0 0.12em;
  border-radius: 4px;
}
.hero-sub {
  color: var(--muted);
  font-size: 1rem;
  max-width: 34rem;
  margin-bottom: 1.8rem;
}

.stButton > button, .stDownloadButton > button {
  border-radius: 12px;
  border: 1px solid var(--line);
  font-weight: 600;
  transition: border-color .15s ease, transform .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--marker);
  transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
  background: var(--marker);
  color: var(--ink);
  border: none;
}
.stButton > button:focus-visible { outline: 2px solid var(--marker); outline-offset: 2px; }

[data-testid="stFileUploader"] section,
.stTextArea textarea, .stTextInput input {
  border-radius: 14px !important;
}
[data-testid="stExpander"] {
  border-radius: 14px;
  border: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] { font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--marker) !important; }

.source-chip {
  display: inline-block;
  font-size: 0.78rem;
  color: var(--paper);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.15rem 0.7rem;
  margin: 0 0.35rem 0.35rem 0;
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
st.markdown(
    '<div class="hero-title">Tus apuntes, <span class="mark">listos para estudiar</span></div>'
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
