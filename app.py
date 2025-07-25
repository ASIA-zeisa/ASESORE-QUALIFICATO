import os
import base64
import requests
import traceback
from flask import Flask, request, render_template_string
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# ─── 0) Load env vars ─────────────────────────────────────────────────────
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

# ─── 0.5) Activation config ──────────────────────────────────────────────
EXAM_CONFIG     = {i: 'off' for i in range(1, 61)}
EXAM_CONFIG.update({1: 'on', 2: 'on', 3: 'off', 4: 'off', 5: 'off'})
SECTION_OPTIONS = ['Lectura', 'Redacción', 'Matemáticas', 'Variable']
PREGUNTA_CONFIG = {i: 'off' for i in range(1, 61)}

# ─── 0.7) Dummy vector for filter-only queries — must match index dimensions
DUMMY_VECTOR = [0.0] * 1536  # your Pinecone index has 1536 dimensions

# ─── 1) Init Pinecone & OpenAI ────────────────────────────────────────────
pc     = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index  = pc.Index(PINECONE_INDEX)
client = OpenAI(api_key=OPENAI_API_KEY)
app    = Flask(__name__)

# ─── 2) HTML + MathJax setup ──────────────────────────────────────────────
# (your existing HTML template here, unchanged)
HTML = '''…'''  # truncated for brevity

@app.route('/', methods=['GET'])
def home():
    return render_template_string(
        HTML,
        exam_config     = EXAM_CONFIG,
        section_options = SECTION_OPTIONS
    )

# ─── 4) Handle question ──────────────────────────────────────────────────
@app.route('/preguntar', methods=['POST'])
def preguntar():
    try:
        texto        = (request.form.get('texto') or "").strip()
        examen       = request.form.get('examen')
        seccion      = request.form.get('seccion')
        pregunta_num = request.form.get('pregunta')
        image_file   = request.files.get('image')

        # block mixed inputs
        if texto and (examen or seccion or pregunta_num or image_file):
            return (
                "Si escribes tu pregunta, no puedes usar “Examen”, “Sección”, "
                "“Pregunta” ni subir imagen al mismo tiempo."
            ), 400

        # require at least one input
        if not (texto or examen or seccion or pregunta_num or image_file):
            return (
                "Proporciona texto, selecciona examen/sección/pregunta o sube una imagen."
            ), 400

        # if exam-based lookup, require section & question
        if examen and not (seccion and pregunta_num):
            return "Cuando seleccionas examen, debes elegir sección y pregunta.", 400

        # 4a) Exact-match lookup by metadata
        snippet = None
        if examen and seccion and pregunta_num:
            pine = index.query(
                vector=DUMMY_VECTOR,
                top_k=1,
                include_metadata=True,
                filter={
                    "exam":     int(examen),
                    "section":  seccion,
                    "question": int(pregunta_num)
                }
            )
            if pine.matches:
                meta    = pine.matches[0].metadata
                snippet = meta.get("text") or meta.get("answer")

        # 4b) If exact-match found, wrap & generate concise explanation
        if snippet:
            clean = snippet.strip('$')
            # build your system & user prompts here…
            # call OpenAI…
            # set formatted_list accordingly
            formatted_list = f"<ol><li>\\({clean}\\)</li></ol>…"
        else:
            # 4c) Fallback: embedding → similarity → LLM formatter (unchanged)
            # …
            formatted_list = "…"  # your existing fallback code

        # 4f) Return response
        response_fragment = (
            f"<p><strong>Enunciado:</strong> {texto}</p>"
            f"<p><strong>Examen:</strong> {examen}</p>"
            f"<p><strong>Sección:</strong> {seccion}</p>"
            f"<p><strong>Pregunta nº:</strong> {pregunta_num}</p>"
            f"{formatted_list} 🤌"
        )
        return response_fragment

    except Exception as e:
        # log full traceback to your server logs
        traceback.print_exc()
        # return the error message in the HTTP response for debugging
        return f"⚠️ Error interno en el servidor:\n{e}", 500

# ─── 5) Run server ───────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT','8000')),
        debug=False
    )
