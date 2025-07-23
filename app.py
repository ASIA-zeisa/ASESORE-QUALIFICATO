import os
import base64
import requests
from flask import Flask, request, jsonify, render_template_string
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# ─── 0) Carga variables de entorno ───────────────────────────────────────
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

# ─── 1) Inicializa Pinecone & OpenAI ────────────────────────────────────
pc     = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index  = pc.Index(PINECONE_INDEX)
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

# ─── 2) Tu página HTML + JS loader ───────────────────────────────────────
HTML = '''<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Asesor Bebé Chat</title>
  <style>
    body            { max-width:720px; margin:2rem auto; font:18px/1.4 sans-serif; }
    h1              { text-align:center; margin-bottom:1.2rem; }
    form            { display:flex; flex-direction:column; gap:1rem; }
    input, button   { font-size:1rem; padding:0.6rem; }
    button          { background:#1450b4; color:#fff; border:none; border-radius:4px; cursor:pointer; }
    button:hover    { background:#0e3c86; }
    #loader         { margin-top:1rem; font-style:italic; display:none; }
    .answer         { margin-top:1.5rem; padding:1rem; background:#f9f9f9; border-left:4px solid #1450b4; }
    footer          { margin-top:3rem; text-align:center; color:#666; font-size:0.9rem; }
  </style>
  <script>
    window.MathJax = { tex:{inlineMath:[['\\\\(','\\\\)']]}, svg:{fontCache:'global'} };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
</head>
<body>
  <h1>Asesor Bebé: tu tutor de matemáticas 🤌</h1>
  <form id="qform">
    <input type="text" name="pregunta" placeholder="Escribe tu problema aquí" required>
    <label>— o sube una imagen:</label>
    <input type="file" name="image">
    <button type="submit">Enviar</button>
  </form>

  <div id="loader">⌛ Creando la mejor respuesta</div>
  <div class="answer" id="answer"></div>

  <footer>Asesor Bebé • Demo Flask + OpenAI + Pinecone</footer>

  <script>
    const form   = document.getElementById('qform');
    const loader = document.getElementById('loader');
    const ansDiv = document.getElementById('answer');

    form.addEventListener('submit', async e => {
      e.preventDefault();
      ansDiv.innerHTML = '';
      loader.style.display = 'block';

      let dots = 0, max = 3;
      const iv = setInterval(() => {
        dots = (dots + 1) % (max + 1);
        loader.textContent = '⌛ Creando la mejor respuesta' + '.'.repeat(dots);
      }, 500);

      const resp = await fetch('/preguntar', { method:'POST', body:new FormData(form) });
      clearInterval(iv);
      loader.style.display = 'none';

      const body = await resp.text();
      if (!resp.ok) {
        ansDiv.textContent = body;
      } else {
        ansDiv.innerHTML = body;
        MathJax.typeset();
      }
    });
  </script>
</body>
</html>'''

# ─── 3) Ruta de inicio ────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML)

# ─── 4) Procesa la pregunta ───────────────────────────────────────────────
@app.route('/preguntar', methods=['POST'])
def preguntar():
    question   = (request.form.get('pregunta') or "").strip()
    image_file = request.files.get('image')
    if not (question or image_file):
        return "Proporciona texto o sube una imagen.", 400

    # 4a) Crear embedding (texto o imagen)
    try:
        if image_file:
            img_bytes = image_file.read()
            emb_resp  = client.embeddings.create(
                model="image-embedding-001",
                input=base64.b64encode(img_bytes).decode()
            )
        else:
            emb_resp  = client.embeddings.create(
                model="text-embedding-3-small",
                input=question
            )
        vector = emb_resp.data[0].embedding
    except Exception as e:
        return f"Error de embedding: {e}", 500

    # 4b) Consultar Pinecone
    try:
        pine = index.query(vector=vector, top_k=5, include_metadata=True)
        snippets = [
            m.metadata.get("text") or m.metadata.get("answer")
            for m in pine.matches
            if m.metadata.get("text") or m.metadata.get("answer")
        ]
    except Exception:
        snippets = []

    # 4c) Fallback “basura de internet” si Pinecone está vacío
    if not snippets:
        try:
            wiki = requests.get(
                "https://es.wikipedia.org/api/rest_v1/page/random/summary", timeout=5
            )
            wiki.raise_for_status()
            fact = wiki.json().get("extract", "")
            snippets = [fact or "Lo siento, no encontré nada aleatorio."]
        except Exception:
            return "No hay datos en Pinecone y falló la búsqueda aleatoria.", 500

    # 4d) Preparar prompt RAG con sólo ese contexto
    context = "\n".join(f"- {s}" for s in snippets)
    rag_prompt = f"""Usa **solo** la información en la lista a continuación para responder.
No agregues nada que no esté aquí.

Contexto:
{context}

Pregunta:
{question}
"""

    # 4e) Llamar al LLM, permitiendo inferencia desde el contexto
    system_msg = (
        "Eres un asistente que sólo utiliza la información en el contexto, "
        "pero puedes hacer cálculos y deducciones basadas en ella. "
        "Si el contexto da una forma completada o factorizada, úsala para despejar x paso a paso."
    )
    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system", "content": system_msg},
                {"role":"user",   "content": rag_prompt}
            ]
        )
        answer = chat.choices[0].message.content.strip() + " 🤌"
    except Exception as e:
        return f"Error de chat: {e}", 500

    # 4f) Devuelve sólo el fragmento HTML con la lista numerada
    return render_template_string('{{ ans|safe }}', ans=answer)

# ─── 5) Ejecuta servidor ──────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT','8000')))
