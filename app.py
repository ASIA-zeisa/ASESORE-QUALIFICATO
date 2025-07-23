import os
import base64
import requests
from flask import Flask, request, jsonify, render_template_string
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

pc     = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index  = pc.Index(PINECONE_INDEX)
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

# Generic 5-step template (unchanged)
SYSTEM_TEMPLATE = """
Eres un tutor de matemáticas experto y muy paciente.
Cuando el usuario haga cualquier pregunta de matemáticas (simplificar una expresión,
resolver una ecuación, traducir un enunciado en palabras a álgebra, etc.):

1. Expresión inicial: \\({EXPR}\\)
2. Identifica el tipo de problema y aplica el método adecuado:
   – Si no hay “=”: simplifica paso a paso.
   – Si hay “=”: aísla la incógnita y despeja.
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

# HTML + loader JS
HTML = '''
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Asesor Bebé Chat</title>
  <style>
    body      { max-width:720px; margin:2rem auto; font:18px/1.4 sans-serif; }
    form      { display:flex; flex-direction:column; gap:1rem; }
    input, button { font-size:1rem; padding:0.6rem; }
    button    { background:#1450b4; color:#fff; border:none; border-radius:4px; }
    button:hover { background:#0e3c86; }
    #loader   { margin-top:1rem; font-style:italic; }
    .answer   { margin-top:1.5rem; padding:1rem; background:#f9f9f9; border-left:4px solid #1450b4; }
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

  <div id="loader" style="display:none;">⌛ Creando la mejor respuesta</div>
  <div class="answer" id="answer"></div>

  <footer style="margin-top:3rem; text-align:center; color:#666; font-size:0.9rem;">
    Asesor Bebé • Demo Flask + OpenAI + Pinecone
  </footer>

  <script>
    const form   = document.getElementById('qform');
    const loader = document.getElementById('loader');
    const ansDiv = document.getElementById('answer');
    form.addEventListener('submit', async e => {
      e.preventDefault();
      ansDiv.innerHTML = '';        // clear old answer
      loader.style.display = 'block';
      let dots = 0;
      const max  = 3;
      const iv = setInterval(()=>{
        dots = (dots + 1) % (max+1);
        loader.textContent = '⌛ Creando la mejor respuesta' + '.'.repeat(dots);
      }, 500);

      // send via fetch
      const resp = await fetch('/preguntar', { method:'POST', body:new FormData(form) });
      clearInterval(iv);
      loader.style.display = 'none';

      if(!resp.ok) {
        ansDiv.textContent = 'Error al obtener respuesta.';
      } else {
        // backend returns full HTML snippet for the answer container
        const html = await resp.text();
        ansDiv.innerHTML = html;
        MathJax.typeset();  // re-render LaTeX
      }
    });
  </script>
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
    if not (question or image_file):
        return jsonify(error="Proporciona texto o sube una imagen."),400

    try:
        if image_file:
            img = image_file.read()
            emb = client.embeddings.create(model="image-embedding-001",
                                          input=base64.b64encode(img).decode())
        else:
            emb = client.embeddings.create(model="text-embedding-3-small",input=question)
        vector = emb.data[0].embedding
    except Exception as e:
        return jsonify(error=f"Embedding error: {e}"),500

    # optional Pinecone snippet fetch (not shown here)
    # …

    system_msg = SYSTEM_TEMPLATE.replace("{EXPR}", question)
    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
              {"role":"system","content":system_msg},
              {"role":"user","content":question}
            ]
        )
        answer = chat.choices[0].message.content.strip() + " 🤌"
    except Exception as e:
        return jsonify(error=f"Chat error: {e}"),500

    # return only the inner HTML for the answer div
    return render_template_string('{{ans|safe}}', ans=answer)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.getenv('PORT','8000')))
