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

# ─── 1) Inicializa servicios externos ─────────────────────────────────────
pc     = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index  = pc.Index(PINECONE_INDEX)
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

# ─── 2) Prompt‐plantilla genérico de 5 pasos para matemáticas ────────────
SYSTEM_TEMPLATE = """
Eres un tutor de matemáticas experto y muy paciente.
Cuando el usuario haga cualquier pregunta de matemáticas (simplificar una expresión,
resolver una ecuación, traducir un enunciado en palabras a álgebra, etc.):

1. Expresión inicial: \\({EXPR}\\)
2. Identifica el tipo de problema y aplica el método adecuado:
   – Si no hay “=”: simplifica paso a paso.
   – Si hay “=“: aísla la incógnita y despeja.
   – Si es un enunciado en palabras: traduce primero a álgebra y luego procede.
3. Usa notación LaTeX entre \\( … \\) para todas las fórmulas.
4. Presenta **al menos 4 pasos** numerados como lista HTML:
   <ol>
     <li>…primer paso…</li>
     <li>…</li>
     <li>…</li>
     <li>…</li>
   </ol>
5. Resultado final: \\(…\\)

No añadas nada fuera de esas etiquetas HTML, ni repitas la pregunta.
"""

# ─── 3) HTML con estilo y MathJax ─────────────────────────────────────────
HTML = '''
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Asesor Bebé Chat</title>
  <style>
    body      { max-width:720px; margin:2rem auto; font:18px/1.4 "Segoe UI", Roboto, sans-serif; color:#222; }
    h1        { font-size:1.8rem; text-align:center; margin-bottom:1.2rem; }
    form      { display:flex; flex-direction:column; gap:1rem; }
    input, button { font-size:1rem; padding:0.6rem; }
    button    { background:#1450b4; color:#fff; border:none; border-radius:4px; cursor:pointer; }
    button:hover { background:#0e3c86; }
    .answer   { background:#f9f9f9; padding:1rem; border-left:4px solid #1450b4; margin-top:2rem; }
    footer    { margin-top:3rem; text-align:center; font-size:0.9rem; color:#666; }
  </style>
  <script>
    window.MathJax = {
      tex: { inlineMath: [['\\\\(','\\\\)']] },
      svg: { fontCache: 'global' }
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
</head>
<body>
  <h1>Asesor Bebé: tu tutor de matemáticas 🤌</h1>
  <form method="post" enctype="multipart/form-data" action="/preguntar">
    <input type="text" name="pregunta" placeholder="Escribe tu problema aquí">
    <label>— o sube una imagen:</label>
    <input type="file" name="image">
    <button type="submit">Enviar</button>
  </form>

  {% if ans %}
  <section class="answer">
    {{ ans|safe }}
  </section>
  {% endif %}

  <footer>Asesor Bebé • Demo Flask + OpenAI + Pinecone</footer>
</body>
</html>
'''


# ─── 4) Rutas Flask ───────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML, ans=None)


@app.route('/preguntar', methods=['POST'])
def preguntar():
    # 4a) Entradas del usuario
    question   = (request.form.get('pregunta') or "").strip()
    image_file = request.files.get('image')
    image_url  = (request.form.get('image_url') or "").strip()

    if not (question or image_file or image_url):
        return jsonify({"error": "Proporciona texto o una imagen."}), 400

    # 4b) Crea embeddings
    try:
        if image_file or image_url:
            img_bytes = image_file.read() if image_file else requests.get(image_url, timeout=10).content
            b64       = base64.b64encode(img_bytes).decode()
            emb_resp  = client.embeddings.create(model="image-embedding-001", input=b64)
        else:
            emb_resp  = client.embeddings.create(model="text-embedding-3-small", input=question)
        vector = emb_resp.data[0].embedding
    except Exception as e:
        return jsonify({"error": f"Error de embedding: {e}"}), 500

    # 4c) (Opcional) busca snippets en Pinecone
    try:
        pine_resp = index.query(vector=vector, top_k=5, include_metadata=True)
        snippets  = [
            m.metadata.get("text") or m.metadata.get("answer")
            for m in pine_resp.matches
            if m.metadata.get("text") or m.metadata.get("answer")
        ]
    except Exception:
        snippets = []

    # 4d) Prepara y envía el prompt al LLM
    system_msg = SYSTEM_TEMPLATE.replace("{EXPR}", question)
    try:
        chat_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": question}
            ]
        )
        answer = chat_resp.choices[0].message.content.strip()
        answer = answer + " 🤌"
    except Exception as e:
        return jsonify({"error": f"Error de chat: {e}"}), 500

    # 4e) Renderiza la misma plantilla HTML con la respuesta
    return render_template_string(HTML, ans=answer)


# ─── 5) Lanza el servidor ─────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '8000')))
