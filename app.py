import os
import requests
import base64
from flask import Flask, request, jsonify, render_template_string
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# If you run locally, this will pick up a .env file.
# On Render it’s a no-op (Render injects the vars for you).
load_dotenv()


# ─── 1) Generic 5-step algebra template ───────────────────────────────────
SYSTEM_TEMPLATE = """
Eres un tutor de matemáticas muy paciente y claro.
Cuando el usuario haga una pregunta de álgebra (por ejemplo, desarrollar una expresión, completar cuadrados o resolver igual a cero, etc.),
siempre responde con **exactamente** estas cinco líneas numeradas en español, usando notación LaTeX entre \\( … \\) para las fórmulas, y sin texto adicional:

1. Expresión inicial: \\({EXPR}\\)
2. Expandimos el binomio: \\( … \\)
3. Reemplazamos en la expresión: \\( … \\)
4. Simplificamos y obtenemos: \\( … \\) que es nuestra expresión final en forma simplificada.
5. Resultado final: …

Donde `{EXPR}` se sustituye automáticamente por la expresión que el usuario escribió.
"""


# ─── 2) Initialize Pinecone, OpenAI & Flask ───────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

pc     = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index  = pc.Index(PINECONE_INDEX)
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)


# ─── 3) HTML form ─────────────────────────────────────────────────────────
HTML = '''
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>AsesorBebe Chat</title></head>
  <body>
    <h1>Pregunta al Bot Multimodal</h1>
    <form method="post" enctype="multipart/form-data" action="/preguntar">
      <input type="text" name="pregunta" size="60" placeholder="Escribe tu problema aquí"><br><br>
      <label>– o sube una imagen:</label>
      <input type="file" name="image"><br><br>
      <label>– o pega una URL de imagen:</label>
      <input type="text" name="image_url" size="60" placeholder="https://..."><br><br>
      <button type="submit">Enviar</button>
    </form>
  </body>
</html>
'''


@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML)


@app.route('/preguntar', methods=['POST'])
def preguntar():
    # Grab user inputs
    question   = (request.form.get('pregunta') or "").strip()
    image_file = request.files.get('image')
    image_url  = (request.form.get('image_url') or "").strip()

    if not (question or image_file or image_url):
        return jsonify({"error": "Proporciona texto o una imagen."}), 400

    # 4) Embed text or image with OpenAI Embeddings
    try:
        if image_file or image_url:
            # image branch
            if image_file:
                image_bytes = image_file.read()
            else:
                resp = requests.get(image_url, timeout=10); resp.raise_for_status()
                image_bytes = resp.content

            b64 = base64.b64encode(image_bytes).decode()
            emb = client.embeddings.create(
                model="image-embedding-001",
                input=b64
            )
        else:
            # text branch
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=question
            )
        vector = emb.data[0].embedding
    except Exception as e:
        return jsonify({"error": f"Error de embedding: {e}"}), 500

    # 5) (Optional) Query Pinecone for related snippets
    #    You can keep this or remove it; it’s not used for math steps below.
    try:
        pine_resp = index.query(vector=vector, top_k=5, include_metadata=True)
        # Just pulling these so you don’t break any existing logic:
        snippets = [
            m.metadata.get("text") or m.metadata.get("answer")
            for m in pine_resp.matches
            if m.metadata.get("text") or m.metadata.get("answer")
        ]
    except Exception:
        snippets = []

    # ─── 6) Build and send the dynamic system+user prompts ────────────────────
    # Inject the actual question into the template:
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
    except Exception as e:
        return jsonify({"error": f"Error de chat: {e}"}), 500

    # ─── 7) Render the HTML response, allowing GPT’s lists to show up ─────────
    return render_template_string(
        '''
        <p><strong>Respuesta:</strong></p>
        {{ ans|safe }}
        <p><a href="/">Hacer otra pregunta</a></p>
        ''',
        ans=answer
    )


if __name__ == '__main__':
    # Use PORT from environment (Render) or default to 8000 locally
    app.run(host='0.0.0.0', port=int(os.getenv('PORT','8000')))
