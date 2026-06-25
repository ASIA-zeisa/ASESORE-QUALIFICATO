import hashlib
import os
import re
from itertools import product
from urllib.parse import urlparse

from dotenv import load_dotenv
from flask import Flask, render_template_string, request, url_for
from pinecone import Pinecone


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "asesor1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "preguntas")

if not PINECONE_API_KEY:
    raise RuntimeError("Falta la variable de entorno PINECONE_API_KEY.")

# Conserva el control de exámenes activos del proyecto anterior.
EXAM_CONFIG = {i: "off" for i in range(1, 200)}
EXAM_CONFIG.update({
    1: "off",
    2: "off",
    3: "off",
    4: "off",
    5: "off",
    6: "off",
    7: "off",
    8: "off",
    9: "off",
    10: "off",
    11: "off",
    12: "off",
    13: "off",
    14: "off",
    15: "off",
    16: "off",
    17: "off",
    18: "off",
    19: "off",
    20: "off",
    21: "off",
    22: "off",
    23: "off",
    24: "off",
    25: "off",
    26: "off",
    27: "off",
    28: "off",
    29: "off",
    30: "off",
    31: "off",
    32: "off",
    33: "off",
    34: "off",
    35: "off",
    36: "off",
    37: "off",
    38: "off",
    39: "off",
    40: "off",
    41: "off",
    42: "off",
    43: "off",
    44: "off",
    45: "off",
    46: "off",
    47: "off",
    48: "off",
    49: "off",
    50: "off",
    51: "off",
    52: "off",
    53: "off",
    54: "off",
    55: "off",
    56: "off",
    57: "off",
    58: "off",
    59: "off",
    60: "off",
    61: "off",
    62: "off",
    63: "off",
    64: "off",
    65: "off",
    66: "off",
    67: "off",
    68: "off",
    69: "off",
    70: "off",
    71: "off",
    72: "off",
    73: "off",
    74: "off",
    75: "off",
    76: "off",
    77: "off",
    78: "off",
    79: "off",
    80: "off",
    81: "off",
    82: "off",
    83: "off",
    84: "off",
    85: "off",
    86: "off",
    87: "off",
    88: "off",
    89: "off",
    90: "off",
    91: "off",
    92: "off",
    93: "off",
    94: "off",
    95: "off",
    96: "off",
    97: "off",
    98: "off",
    99: "off",
    100: "off",
    101: "off",
    102: "off",
    103: "off",
    104: "off",
    105: "off",
    106: "off",
    107: "off",
    108: "off",
    109: "off",
    110: "off",
    111: "off",
    112: "off",
    113: "off",
    114: "off",
    115: "off",
    116: "off",
    117: "off",
    118: "off",
    119: "off",
    120: "off",
    121: "off",
    122: "off",
    123: "off",
    124: "off",
    125: "off",
    126: "off",
    127: "off",
    128: "off",
    129: "off",
    130: "off",
    131: "off",
    132: "off",
    133: "off",
    134: "off",
    135: "off",
    136: "off",
    137: "off",
    138: "off",
    139: "off",
    140: "off",
    141: "off",
    142: "off",
    143: "off",
    144: "off",
    145: "off",
    146: "off",
    147: "off",
    148: "off",
    149: "off",
    150: "off",
    151: "off",
    152: "off",
    153: "off",
    154: "off",
    155: "off",
    156: "off",
    157: "off",
    158: "off",
    159: "off",
    160: "off",
    161: "off",
    162: "off",
    163: "off",
    164: "off",
    165: "off",
    166: "off",
    167: "off",
    168: "off",
    169: "off",
    170: "off",
    171: "off",
    172: "off",
    173: "off",
    174: "off",
    175: "off",
    176: "off",
    177: "off",
    178: "off",
    179: "off",
    180: "off",
    181: "off",
    182: "off",
    183: "off",
    184: "off",
    185: "off",
    186: "off",
    187: "on",
    188: "off",
    189: "off",
    190: "off",
    191: "off",
    192: "off",
    193: "off",
    194: "off",
    195: "off",
    196: "off",
    197: "off",
    198: "off",
    199: "off",
    200: "off",
})

# El valor enviado por la interfaz es numérico, pero el backend también prueba
# los nombres anteriores para mantener compatibilidad con el Excel.
SECTION_CONFIG = {
    "1": {
        "label": "Sección 1 — Lectura",
        "limit": 45,
        "aliases": ["1", "Sección 1", "Lectura"],
    },
    "2": {
        "label": "Sección 2 — Redacción",
        "limit": 25,
        "aliases": ["2", "Sección 2", "Redacción"],
    },
    "3": {
        "label": "Sección 3 — Matemáticas",
        "limit": 55,
        "aliases": ["3", "Sección 3", "Matemáticas"],
    },
    "4": {
        "label": "Sección 4 — Variable",
        "limit": 25,
        "aliases": ["4", "Sección 4", "Variable"],
    },
}

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
app = Flask(__name__)


# ---------------------------------------------------------------------------
# Funciones compatibles con Ocelote
# ---------------------------------------------------------------------------

def crear_id(examen: str, seccion: str, pregunta: str) -> str:
    """Reproduce exactamente la ID determinista generada por Ocelote."""
    clave = f"{examen}|{seccion}|{pregunta}".casefold().strip()
    digest = hashlib.sha256(clave.encode("utf-8")).hexdigest()[:24]
    return f"pregunta-{digest}"


def valores_unicos(valores):
    resultado = []
    vistos = set()

    for valor in valores:
        texto = str(valor).strip()
        clave = texto.casefold()
        if texto and clave not in vistos:
            vistos.add(clave)
            resultado.append(texto)

    return resultado


def variantes_examen(examen: str) -> list[str]:
    examen = str(examen).strip()
    variantes = [examen]

    if examen.isdigit():
        variantes.append(f"Examen {int(examen)}")

    return valores_unicos(variantes)


def variantes_seccion(seccion: str) -> list[str]:
    seccion = str(seccion).strip()
    variantes = [seccion]

    config = SECTION_CONFIG.get(seccion)
    if config:
        variantes.extend(config["aliases"])

    if seccion.isdigit():
        variantes.append(f"Sección {int(seccion)}")

    return valores_unicos(variantes)


def variantes_pregunta(pregunta: str) -> list[str]:
    pregunta = str(pregunta).strip()
    variantes = [pregunta]

    if pregunta.isdigit():
        variantes.append(str(int(pregunta)))
        variantes.append(f"Pregunta {int(pregunta)}")

    return valores_unicos(variantes)


def ids_candidatas(examen: str, seccion: str, pregunta: str) -> list[str]:
    """
    Genera varias ID posibles para tolerar valores como:
      1 / Examen 1
      1 / Sección 1 / Lectura
      1 / Pregunta 1
    """
    combinaciones = product(
        variantes_examen(examen),
        variantes_seccion(seccion),
        variantes_pregunta(pregunta),
    )

    return valores_unicos(
        crear_id(ex, sec, preg)
        for ex, sec, preg in combinaciones
    )


def extraer_vectores(resultado) -> dict:
    if isinstance(resultado, dict):
        return resultado.get("vectors", {}) or {}

    return getattr(resultado, "vectors", {}) or {}


def extraer_metadata(registro) -> dict:
    if isinstance(registro, dict):
        return registro.get("metadata", {}) or {}

    return getattr(registro, "metadata", {}) or {}


def buscar_pregunta(
    examen: str,
    seccion: str,
    pregunta: str,
) -> tuple[dict | None, str | None]:
    """
    Recupera el registro exacto mediante fetch dentro del namespace utilizado
    por Ocelote. No genera embeddings y no llama a OpenAI.
    """
    candidatas = ids_candidatas(examen, seccion, pregunta)

    resultado = index.fetch(
        ids=candidatas,
        namespace=PINECONE_NAMESPACE,
    )
    vectores = extraer_vectores(resultado)

    for vector_id in candidatas:
        registro = vectores.get(vector_id)
        if registro is not None:
            return extraer_metadata(registro), vector_id

    return None, None


def dividir_referencias_imagenes(texto: str) -> list[str]:
    if not texto:
        return []

    partes = re.split(r"[\n|]+", str(texto))
    return [parte.strip() for parte in partes if parte.strip()]


def es_url_directa(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def obtener_urls_imagenes(metadata: dict) -> list[str]:
    urls = []

    # Opción principal:
    # Una o varias URL en la columna imagen_url, separadas por salto de línea o por |
    for referencia in dividir_referencias_imagenes(metadata.get("imagen_url", "")):
        if es_url_directa(referencia):
            urls.append(referencia)

    # Compatibilidad opcional: imagen_url_1, imagen_url_2, imagen_url_3...
    for key in sorted(metadata.keys()):
        if re.fullmatch(r"imagen_url_\d+", str(key)):
            referencia = str(metadata.get(key) or "").strip()
            if referencia and es_url_directa(referencia):
                urls.append(referencia)

    # Opción local:
    # Una o varias imágenes dentro de static/, separadas por salto de línea o por |
    for referencia in dividir_referencias_imagenes(metadata.get("imagen_archivo", "")):
        ruta = referencia.replace("\\", "/").lstrip("/")

        if ruta.startswith("static/"):
            ruta = ruta[len("static/"):]

        if ruta:
            urls.append(url_for("static", filename=ruta))

    # Compatibilidad opcional: imagen_archivo_1, imagen_archivo_2...
    for key in sorted(metadata.keys()):
        if re.fullmatch(r"imagen_archivo_\d+", str(key)):
            referencia = str(metadata.get(key) or "").strip()
            if referencia:
                ruta = referencia.replace("\\", "/").lstrip("/")

                if ruta.startswith("static/"):
                    ruta = ruta[len("static/"):]

                urls.append(url_for("static", filename=ruta))

    return valores_unicos(urls)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

HTML = r'''<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Asesore Qualificato</title>

  <style>
    body {
      max-width: 760px;
      margin: 2rem auto;
      padding: 0 1rem;
      font: 18px/1.45 sans-serif;
      color: #222;
    }

    h1 {
      text-align: center;
      margin-bottom: 2rem;
    }

    form {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .inline-selects {
      display: flex;
      gap: 1rem;
    }

    .select-group {
      display: flex;
      flex: 1;
      flex-direction: column;
    }

    .select-label {
      margin-bottom: 0.25rem;
      color: #666;
      font-size: 0.9rem;
    }

    select,
    button {
      width: 100%;
      padding: 0.65rem;
      font-size: 1rem;
    }

    button {
      border: none;
      border-radius: 4px;
      background: #1450b4;
      color: #fff;
      cursor: pointer;
    }

    button:hover {
      background: #0e3c86;
    }

    button:disabled {
      cursor: wait;
      opacity: 0.7;
    }

    #loader {
      display: none;
      margin-top: 1rem;
      font-style: italic;
    }

    .answer {
      min-height: 1rem;
      margin-top: 1.5rem;
      padding: 1rem;
      border-left: 4px solid #1450b4;
      background: #f9f9f9;
    }

    .answer-section {
      margin-bottom: 1.25rem;
    }

    .answer-text {
      white-space: pre-wrap;
    }

    .answer-image {
      display: block;
      max-width: 100%;
      height: auto;
      margin: 1rem auto 0;
      border-radius: 6px;
    }

    .error {
      color: #9f1d1d;
    }

    footer {
      margin-top: 2rem;
      color: #666;
      text-align: center;
      font-size: 0.9rem;
    }

    @media (max-width: 680px) {
      .inline-selects {
        flex-direction: column;
      }
    }
  </style>

  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      },
      svg: { fontCache: 'global' }
    };
  </script>
  <script
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
    async>
  </script>
</head>

<body>
  <h1>Asesore Qualificato 🤌</h1>

  <form id="qform">
    <div class="inline-selects">
      <div class="select-group">
        <label class="select-label" for="examen">Examen</label>
        <select id="examen" name="examen" required>
          <option value="">Selecciona un examen</option>
          {% for num, status in exam_config.items()|sort %}
            {% if status == "on" %}
              <option value="{{ num }}">Examen {{ num }}</option>
            {% endif %}
          {% endfor %}
        </select>
      </div>

      <div class="select-group">
        <label class="select-label" for="seccion">Sección</label>
        <select id="seccion" name="seccion" required>
          <option value="">Selecciona una sección</option>
          {% for value, config in section_config.items() %}
            <option value="{{ value }}">{{ config.label }}</option>
          {% endfor %}
        </select>
      </div>

      <div class="select-group">
        <label class="select-label" for="pregunta">Pregunta</label>
        <select id="pregunta" name="pregunta" required disabled>
          <option value="">Selecciona una pregunta</option>
        </select>
      </div>
    </div>

    <button id="submitButton" type="submit">Enviar</button>
  </form>

  <div id="loader">⌛ Buscando la respuesta...</div>
  <div class="answer" id="answer"></div>


  <script>
    const form = document.getElementById('qform');
    const seccionEl = document.getElementById('seccion');
    const preguntaEl = document.getElementById('pregunta');
    const loader = document.getElementById('loader');
    const answer = document.getElementById('answer');
    const submitButton = document.getElementById('submitButton');

    const preguntaLimits = {{ pregunta_limits | tojson }};

    seccionEl.addEventListener('change', () => {
      const limit = Number(preguntaLimits[seccionEl.value] || 0);

      preguntaEl.innerHTML =
        '<option value="">Selecciona una pregunta</option>';

      for (let i = 1; i <= limit; i += 1) {
        const option = document.createElement('option');
        option.value = String(i);
        option.textContent = `Pregunta ${i}`;
        preguntaEl.appendChild(option);
      }

      preguntaEl.disabled = limit === 0;
    });

    form.addEventListener('submit', async event => {
      event.preventDefault();

      answer.innerHTML = '';
      loader.style.display = 'block';
      submitButton.disabled = true;

      try {
        const response = await fetch('/preguntar', {
          method: 'POST',
          body: new FormData(form)
        });

        const body = await response.text();

        if (!response.ok) {
          answer.innerHTML = `<p class="error">${body}</p>`;
          return;
        }

        answer.innerHTML = body;

        if (window.MathJax && window.MathJax.typesetPromise) {
          await window.MathJax.typesetPromise([answer]);
        }
      } catch (error) {
        answer.innerHTML =
          '<p class="error">No fue posible comunicarse con el servidor.</p>';
      } finally {
        loader.style.display = 'none';
        submitButton.disabled = false;
      }
    });
  </script>
</body>
</html>'''


ANSWER_FRAGMENT = r'''
{% if metadata.get("enunciado") %}
  <div class="answer-section">
    <strong>Enunciado</strong>
    <div class="answer-text">{{ metadata.get("enunciado") }}</div>
  </div>
{% endif %}

{% if metadata.get("opciones") %}
  <div class="answer-section">
    <strong>Opciones</strong>
    <div class="answer-text">{{ metadata.get("opciones") }}</div>
  </div>
{% endif %}

{% if metadata.get("respuesta") %}
  <div class="answer-section">
    <strong>Respuesta</strong>
    <div class="answer-text">{{ metadata.get("respuesta") }}</div>
  </div>
{% endif %}

{% if metadata.get("explicacion") %}
  <div class="answer-section">
    <strong>Explicación</strong>
    <div class="answer-text">{{ metadata.get("explicacion") }}</div>
  </div>
{% endif %}

{% for imagen_url in imagenes_url %}
  <div class="answer-section">
    <img
      class="answer-image"
      src="{{ imagen_url }}"
      alt="Imagen explicativa {{ loop.index }}">
  </div>
{% endfor %}
'''


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    pregunta_limits = {
        key: config["limit"]
        for key, config in SECTION_CONFIG.items()
    }

    return render_template_string(
        HTML,
        exam_config=EXAM_CONFIG,
        section_config=SECTION_CONFIG,
        pregunta_limits=pregunta_limits,
    )


@app.route("/preguntar", methods=["POST"])
def preguntar():
    examen = str(request.form.get("examen") or "").strip()
    seccion = str(request.form.get("seccion") or "").strip()
    pregunta_num = str(request.form.get("pregunta") or "").strip()

    if not examen or not seccion or not pregunta_num:
        return "Debes seleccionar el examen, la sección y la pregunta.", 400

    try:
        metadata, _vector_id = buscar_pregunta(
            examen=examen,
            seccion=seccion,
            pregunta=pregunta_num,
        )
    except Exception:
        app.logger.exception("Error consultando Pinecone")
        return (
            "No fue posible consultar Pinecone. "
            "Revisa el índice, el namespace y la API key.",
            500,
        )

    if metadata is None:
        return (
            "No se encontró esa combinación de examen, sección y pregunta "
            "en el índice.",
            404,
        )

    if not metadata.get("respuesta") and not metadata.get("explicacion"):
        return (
            "El registro existe, pero no contiene respuesta ni explicación.",
            500,
        )

    imagenes_url = obtener_urls_imagenes(metadata)

    return render_template_string(
        ANSWER_FRAGMENT,
        metadata=metadata,
        imagenes_url=imagenes_url,
    )


# ---------------------------------------------------------------------------
# Ejecución local
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        debug=False,
    )
