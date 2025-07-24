import os
import base64
import requests
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
EXAM_CONFIG = {i: 'off' for i in range(1, 61)}
EXAM_CONFIG.update({1: 'on', 2: 'on', 3: 'off', 4: 'off', 5: 'off'})
SECTION_CONFIG = {}
PREGUNTA_CONFIG = {i: 'off' for i in range(1, 61)}
PREGUNTA_CONFIG.update({})

# ─── 1) Init Pinecone & OpenAI ────────────────────────────────────────────
pc     = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index  = pc.Index(PINECONE_INDEX)
client = OpenAI(api_key=OPENAI_API_KEY)
app    = Flask(__name__)

# ─── 2) HTML + MathJax setup ──────────────────────────────────────────────
HTML = '''<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Asesore Qualificato</title>
  <style>
    body{max-width:720px;margin:2rem auto;font:18px/1.4 sans-serif;color:#222;}
    h1{text-align:center;margin-bottom:1.2rem;}
    form{display:flex;flex-direction:column;gap:1rem;}
    .inline-selects{display:flex;gap:1rem;}
    select,textarea,button{font-size:1rem;padding:0.6rem;}
    select{flex:1;}
    button{background:#1450b4;color:#fff;border:none;border-radius:4px;cursor:pointer;}
    button:hover{background:#0e3c86;}
    #loader{margin-top:1rem;font-style:italic;display:none;}
    .answer{margin-top:1.5rem;padding:1rem;background:#f9f9f9;border-left:4px solid #1450b4;}
    footer{margin-top:2rem;text-align:center;color:#666;font-size:0.9rem;}
  </style>
  <script>
    window.MathJax = {tex:{inlineMath:[['$','$'],['\\(','\\)']],displayMath:[['$$','$$']]},svg:{fontCache:'global'}};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
</head>
<body>
  <h1>Asesore Qualificato: tu tutore matemático 🤌</h1>
  <form id="qform">
    <textarea name="texto" rows="3" placeholder="Escribe tu pregunta aquí" required></textarea>

    <label>— o selecciona tu pregunta:</label>
    <div class="inline-selects">
      <select name="examen">
        <option value="">Examen</option>
        {% for num, status in exam_config.items()|sort %}
          {% if status == 'on' %}
        <option value="{{ num }}">{{ num }}</option>
          {% endif %}
        {% endfor %}
      </select>
      <select name="seccion">
        <option value="">Sección</option>
        {% for key, status in section_config.items()|sort %}
          {% if status == 'on' %}
        <option value="{{ key }}">{{ key }}</option>
          {% endif %}
        {% endfor %}
      </select>
      <select name="pregunta">
        <option value="">Pregunta</option>
        {% for num, status in pregunta_config.items()|sort %}
          {% if status == 'on' %}
        <option value="{{ num }}">{{ num }}</option>
          {% endif %}
        {% endfor %}
      </select>
    </div>

    <label>— o sube una imagen:</label>
    <input type="file" name="image">
    <button type="submit">Enviar</button>
  </form>

  <div id="loader">⌛ Creando la mejor respuesta</div>
  <div class="answer" id="answer"></div>

  <footer>Asesor Bebé • Demo Flask + OpenAI + Pinecone</footer>

  <script>
    const form = document.getElementById('qform');
    const loader = document.getElementById('loader');
    const ansDiv = document.getElementById('answer');

    form.addEventListener('submit', async e => {
      e.preventDefault();
      ansDiv.innerHTML = '';

      // Determine if it's a text-only query
      const textoVal = form.elements['texto'].value.trim();
      const examenVal = form.elements['examen'].value;
      const seccionVal = form.elements['seccion'].value;
      const preguntaVal = form.elements['pregunta'].value;
      const imageFiles = form.elements['image'].files.length;
      const isTextOnly = textoVal && !examenVal && !seccionVal && !preguntaVal && imageFiles === 0;

      // Set loading message accordingly
      const baseMsg = isTextOnly ? '⌛ Resolviendo tu pregunta' : '⌛ Creando la mejor respuesta';
      loader.textContent = baseMsg;
      loader.style.display = 'block';

      let dots = 0;
      const iv = setInterval(() => {
        dots = (dots + 1) % 4;
        loader.textContent = baseMsg + '.'.repeat(dots);
      }, 500);

      const resp = await fetch('/preguntar', {
        method: 'POST',
        body: new FormData(form)
      });

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
}]}
