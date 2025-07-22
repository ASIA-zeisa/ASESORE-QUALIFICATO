import os
import requests
import base64
from flask import Flask, request, jsonify, render_template_string
from pinecone import Pinecone
from openai import OpenAI

# ─── Your API keys (embedded) ─────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone & OpenAI
pc     = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index  = pc.Index(PINECONE_INDEX)
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Flask
app = Flask(__name__)

# HTML form template
HTML = '''
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>AsesorBebe Chat</title>
  </head>
  <body>
    <h1>Pregunta al Bot Multimodal</h1>
    <form method="post" enctype="multipart/form-data" action="/preguntar">
      <input type="text" name="pregunta" size="60" placeholder="Escribe tu pregunta aquí"><br><br>
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
    question   = (request.form.get('pregunta') or "").strip()
    image_file = request.files.get('image')
    image_url  = (request.form.get('image_url') or "").strip()

    if not (question or image_file or image_url):
        return jsonify({"error": "Provide text or an image or URL."}), 400

    # 1) Embed text or image using OpenAI Embeddings
    try:
        if image_file or image_url:
            # Image branch
            if image_file:
                image_bytes = image_file.read()
            else:
                resp = requests.get(image_url, timeout=10)
                resp.raise_for_status()
                image_bytes = resp.content

            b64 = base64.b64encode(image_bytes).decode()
            emb = client.embeddings.create(
                model="image-embedding-001",
                input=b64
            )
        else:
            # Text branch
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=question
            )
        vector = emb.data[0].embedding
    except Exception as e:
        return jsonify({"error": f"Embedding error: {e}"}), 500

    # 2) Query Pinecone for top-K
    try:
        pine_resp = index.query(vector=vector, top_k=5, include_metadata=True)
        snippets = [m.metadata.get("text", m.metadata.get("answer")) for m in pine_resp.matches if m.metadata.get("text") or m.metadata.get("answer")
]
    except Exception as e:
        return jsonify({"error": f"Pinecone error: {e}"}), 500

    if not snippets:
        return jsonify({"error": "No relevant text found."}), 404

    # 3) Build retrieval-augmented prompt
    context = "\n\n---\n\n".join(snippets)
    prompt  = f"""You are a helpful assistant. Answer the user's question using ONLY the information in the context below.
Do NOT introduce any outside knowledge.

Context:
{context}

Question:
{question or 'Describe the image you uploaded.'}
"""

    # 4) Call OpenAI to generate the answer
    try:
        chat_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You only know what’s in the context."},
                {"role": "user",   "content": prompt}
            ]
        )
        answer = chat_resp.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": f"Chat error: {e}"}), 500

    # 5) Return the answer
    return render_template_string(
        '<p><strong>Respuesta:</strong> {{ans}}</p><a href="/">Hacer otra pregunta</a>',
        ans=answer
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT','8000')))
