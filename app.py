import os
import base64
import requests
from flask import Flask, request, jsonify, render_template_string
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"),
                 environment=os.getenv("PINECONE_ENV"))
index  = pc.Index(os.getenv("PINECONE_INDEX"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

SYSTEM_TEMPLATE = """…your 5-step template…"""  # keep as before

HTML = """…your HTML template with JS loader…"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML, ans=None)

@app.route('/preguntar', methods=['POST'])
def preguntar():
    question = (request.form.get('pregunta') or "").strip()
    image_file = request.files.get('image')
    if not (question or image_file):
        return jsonify(error="Proporciona texto o sube una imagen."), 400

    # 1) Create embedding
    try:
        if image_file:
            img = image_file.read()
            emb = client.embeddings.create(
                model="image-embedding-001",
                input=base64.b64encode(img).decode()
            )
        else:
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=question
            )
        vector = emb.data[0].embedding
    except Exception as e:
        return jsonify(error=f"Embedding error: {e}"), 500

    # 2) Query Pinecone
    snippets = []
    try:
        pine = index.query(vector=vector, top_k=5, include_metadata=True)
        snippets = [
            m.metadata.get("text") or m.metadata.get("answer")
            for m in pine.matches
            if m.metadata.get("text") or m.metadata.get("answer")
        ]
    except:
        pass

    # 3) If Pinecone empty, do a DuckDuckGo web search
    if not snippets:
        try:
            ddg = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": question, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=5
            ).json()
            summary = ddg.get("AbstractText") or ddg.get("RelatedTopics",[{}])[0].get("Text","")
            if summary:
                snippets = [summary]
            else:
                snippets = ["Lo siento, no encontré información relevante en la web."]
        except Exception:
            snippets = ["Lo siento, no pude buscar en la web."]

    # 4) Build your dynamic system prompt
    system_msg = SYSTEM_TEMPLATE.replace("{EXPR}", question)

    # 5) Call the LLM using either the Pinecone + web snippets as context
    #    (You can inject snippets into your prompt if you like,
    #     or just let the template run with {EXPR}.)
    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system", "content": system_msg},
                {"role":"user",   "content": question}
            ]
        )
        answer = chat.choices[0].message.content.strip() + " 🤌"
    except Exception as e:
        return jsonify(error=f"Chat error: {e}"), 500

    return render_template_string(HTML, ans=answer)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT','8000')))
