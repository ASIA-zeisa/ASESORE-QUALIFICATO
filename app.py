import os, base64, requests
from flask import Flask, request, jsonify, render_template_string
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# ─── 0) Carga variables de entorno ────────────────────────────────────────
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

# ─── 2) Prompt‐plantilla genérico de 5 pasos para álgebra ────────────────
SYSTEM_TEMPLATE = """
Eres un tutor de matemáticas muy paciente y claro.
Cuando el usuario haga una pregunta de álgebra (por ejemplo, desarrollar una expresión,
completar cuadrados o resolver igual a cero, etc.), **responde siempre con
EXACTAMENTE** estas cinco líneas numeradas en español, usando notación LaTeX entre
\\( … \\) para todas las fórmulas, sin texto adicional:

1. Expresión inicial: \\({EXPR}\\)
2. Expandimos el binomio: \\( … \\)
3. Reemplazamos en la expresión: \\( … \\)
4. Simplificamos y obtenemos: \\( … \\) que es nuestra expresión final en forma simplificada.
5. Resultado final: …

Donde `{EXPR}` se sustituye automáticamente por la expresión que el usuario escribió.
"""

# ─── 3) HTML con estilo y MathJax ─────────────────────────────────────────
HTML = '''
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Asesor Bebé Chat</title>

  <!-- NEW: Estilos rápidos para una vista agradable -->
  <style>
    body      { max-width: 720px; margin: 2rem auto; font: 18px/1.4 "Segoe UI", Roboto, sans-serif; color:#222; }
    h1        { font-size: 1.8rem; text-align:center; margin-bottom:1.2rem; }
    form      { display:flex; flex-direction:column; gap:1rem; }
    input[type="text"], input[type="file"] { padding:0.6rem; font-size:1rem; width:100%; }
    button    { padding:0.8rem 1.2rem; font-size:1rem; background:#1450b4; color:#fff;
                border:none; border-radius:4px; cursor:pointer; }
    button:hover { background:#0e3c86; }
    .answer   { background:#f9f9f9; padding:1rem 1.2rem; border-left:4px solid #1450b4; }
    a         { color:#1450b4; }
    footer    { margin-top:3rem; text-align:center; font-size:0.9rem; color:#666; }
  </style>

  <!-- NEW: MathJax 3 para renderizar LaTeX -->
  <script>
    window.MathJax = {
      tex: { inlineMath: [['\\\\(','\\\\)']] },
      svg: { fontCache: 'global' }
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
</head>
<body>
  <h1>Pregunta al Bot Multimodal</h1>

  <form method="post" enctype="multipart/form-data" action="/preguntar">
    <input type="text" name="pregunta" placeholder="Escribe tu problema aquí">
    <label>— o sube una imagen:</label>
    <input type="file" name="image">
    <label>— o pega una URL de imagen:</label>
    <input type="text" name="image_url" placeholder="https://…">
    <button type="submit">Enviar</button>
  </form>

  {% if ans %}
  <section class="answer">
    <p><strong>Respuesta:</strong></p>
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
    # Render vacío, sin respuesta
    return render_template_string(HTML, ans=None)

@app.route('/preguntar', methods=['POST'])
def preguntar():
    # ── 4a) Entradas del usuario ──────────────────────────────────────────
    question   = (request.form.get('pregunta') or "").strip()
    image_file = request.files.get('image')
    image_url  = (request.form.get('image_url') or "").strip()

    if not (question or image_file or image_url):
        return jsonify({"error": "Proporciona texto o una imagen."}), 400

    # ── 4b) Embeddings con OpenAI ─────────────────────────────────────────
    try:
        if image_file or image_url:
            # imagen → bytes → base64
            img_bytes = image_file.read() if image_file else requests.get(image_url, timeout=10).content
            b64       = base64.b64encode(img_bytes).decode()
            emb_resp  = client.embeddings.create(model="image-embedding-001", input=b64)
        else:
            emb_resp  = client.embeddings.create(model="text-embedding-3-small", input=question)
        vector = emb_resp.data[0].embedding
    except Exception as e:
        return jsonify({"error": f"Error de embedding: {e}"}), 500

    # ── 4c) (Opcional) búsqueda en Pinecone ──────────────────────────────
    try:
        pine_resp = index.query(vector=vector, top_k=5, include_metadata=True)
        snippets  = [m.metadata.get("text") or m.metadata.get("answer") for m in pine_resp.matches
                     if m.metadata.get("text") or m.metadata.get("answer")]
    except Exception:
        snippets = []

    # ── 4d) Construir prompt dinámico y llamar al chat ───────────────────
    system_msg = SYSTEM_TEMPLATE.replace("{EXPR}", question or "…")
    try:
        chat_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": question}
            ]
        )
        # ─── Add the 🤌 at the very end ───────────────────────────────────
        answer = answer.rstrip() + " 🤌"
    except Exception as e:
        return jsonify({"error": f"Error de chat: {e}"}), 500

    # ── 4e) Devolver HTML con la respuesta (se renderiza MathJax) ────────
    return render_template_string(HTML, ans=answer)

# ─── 5) Lanzar servidor ───────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '8000')))
