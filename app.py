import os
import base64
import requests
from flask import Flask, request, render_template_string
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# ─── 0) Load env vars ─────────────────────────────────────────────────────
load_dotenv()
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_ENV       = os.getenv("PINECONE_ENV")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
CUSTOM_GPT_MODEL   = os.getenv("CUSTOM_GPT_MODEL")   # p.ej. "gpt-4o-tu-PAA-completo"

# ─── 0.5) Activation config ──────────────────────────────────────────────
EXAM_CONFIG     = {i: 'off' for i in range(1, 61)}
EXAM_CONFIG.update({1: 'on', 2: 'on', 3: 'off', 4: 'off', 5: 'off'})
SECTION_OPTIONS = ['Lectura', 'Redacción', 'Matemáticas', 'Variable']

# ─── 0.7) Dummy vector for filter-only queries — must match index dimensions
DUMMY_VECTOR = [0.0] * 1536

# ─── 1) Init Pinecone & OpenAI ────────────────────────────────────────────
pc     = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index  = pc.Index(PINECONE_INDEX)
client = OpenAI(api_key=OPENAI_API_KEY)
app    = Flask(__name__)

# ─── 2) HTML + MathJax setup ──────────────────────────────────────────────
HTML = '''… (sin cambios) …'''  # mantenemos tu HTML tal cual

# ─── 3) Home route ───────────────────────────────────────────────────────
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
    texto        = (request.form.get('texto') or "").strip()
    examen       = request.form.get('examen')
    seccion      = request.form.get('seccion')
    pregunta_num = request.form.get('pregunta')
    image_file   = request.files.get('image')

    # 1) Validaciones básicas
    if texto and (examen or seccion or pregunta_num or image_file):
        return (
            "Si escribes tu pregunta, no puedes usar “Examen”, “Sección”, "
            "“Pregunta” ni subir imagen al mismo tiempo."
        ), 400
    if not (texto or examen or seccion or pregunta_num or image_file):
        return (
            "Proporciona texto, selecciona examen/sección/pregunta o sube una imagen."
        ), 400
    if examen and not (seccion and pregunta_num):
        return "Cuando seleccionas examen, debes elegir sección y pregunta.", 400

    # ─── 2) Branch: examen puro → GPT custom ────────────────────────────────
    if examen and seccion and pregunta_num and not texto and not image_file:
        system_prompt = (
            "Eres un profesor de matemáticas experto en PAA que tiene TODAS las "
            "preguntas y respuestas de todos los exámenes. Cuando te den Examen X, "
            "Sección Y, Pregunta Z, reproduces la pregunta exacta y la explicas en 5 "
            "pasos numerados usando delimitadores \\(…\\)."
        )
        user_prompt = (
            f"Examen {examen}, Sección {seccion}, Pregunta {pregunta_num}.\n"
            "Por favor, muestra la pregunta y da la explicación en 5 pasos."
        )
        chat = client.chat.completions.create(
            model=CUSTOM_GPT_MODEL,
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user",   "content":user_prompt}
            ]
        )
        # Devolvemos directamente lo que genere el custom GPT
        return chat.choices[0].message.content.strip()

    # ─── 3) Branch: texto libre o imagen → Pinecone + embed + formatter ────

    # 3a) Intentamos exact-match por metadata (opcional; puedes omitir si ya no lo usas)
    snippet = None
    if examen and seccion and pregunta_num:
        try:
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
        except Exception:
            snippet = None

    # 3b) Si hay snippet, usamos gpt-4o-mini para explicar de forma concisa
    if snippet:
        clean = snippet.strip('$')
        system_prompt = (
            "Eres un profesor de matemáticas que explica de forma muy concisa "
            "en español, en no más de 5 pasos numerados, usando delimitadores "
            "\\(…\\) para las expresiones matemáticas."
        )
        context = f"Examen {examen}, Sección {seccion}, Pregunta {pregunta_num}"
        user_prompt = (
            f"Ecuación: {context}\n"
            f"Respuesta: \\({clean}\\)\n\n"
            "Proporciona una lista numerada (1–5) de los pasos clave "
            "para completar el cuadrado rápidamente."
        )
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}
            ]
        )
        explanation = chat.choices[0].message.content.strip()
        formatted_list = (
            f"<ol><li>\\({clean}\\)</li></ol>"
            f"<p><strong>Pasos rápidos:</strong></p>"
            f"{explanation}"
        )

    else:
        # 3c) Fallback embedding (texto o imagen) → búsqueda → formatter
        try:
            if image_file and not texto:
                img_bytes = image_file.read()
                emb = client.embeddings.create(
                    model="image-embedding-001",
                    input=base64.b64encode(img_bytes).decode()
                )
            else:
                emb = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texto
                )
            vector = emb.data[0].embedding
            pine = index.query(
                vector=vector,
                top_k=5,
                include_metadata=True
            )
            raw_steps = [
                m.metadata.get('text') or m.metadata.get('answer')
                for m in pine.matches
                if m.metadata.get('text') or m.metadata.get('answer')
            ]
        except Exception:
            raw_steps = []

        if not raw_steps:
            try:
                wiki = requests.get(
                    'https://es.wikipedia.org/api/rest_v1/page/random/summary',
                    timeout=5
                ).json()
                raw_steps = [wiki.get('extract','Lo siento, nada')]
            except:
                return 'No hay datos en Pinecone y falló la búsqueda aleatoria.', 500

        format_msg = (
            'Eres un formateador HTML muy estricto. Toma estas frases y devuélvelas '
            'como una lista ordenada (<ol><li>…</li></ol>) en español, sin texto '
            'adicional. Usa siempre los delimitadores LaTeX \\(…\\) para las fórmulas.\n\n'
            + '\n'.join(f'- {s}' for s in raw_steps)
        )
        try:
            chat = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role':'system','content':format_msg},
                    {'role':'user','content':'Por favor formatea la lista.'}
                ]
            )
            formatted_list = chat.choices[0].message.content.strip()
        except Exception as e:
            return f'Error de formateo: {e}', 500

    formatted_list = formatted_list.replace("\\[", "\\(").replace("\\]", "\\)")

    # 4) Montaje final de la respuesta para texto libre / imagen
    response_fragment = (
        f"<p><strong>Enunciado:</strong> {texto}</p>"
        f"<p><strong>Examen:</strong> {examen}</p>"
        f"<p><strong>Sección:</strong> {seccion}</p>"
        f"<p><strong>Pregunta nº:</strong> {pregunta_num}</p>"
        f"{formatted_list} 🤌"
    )
    return response_fragment

# ─── 5) Run server ───────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT','8000')),
        debug=False
    )
