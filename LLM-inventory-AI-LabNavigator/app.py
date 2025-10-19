# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM

from retriever_mongo import retrieve
from gpt4all import GPT4All
import nltk
from nltk.tokenize import sent_tokenize
# nltk cümle tokenizer kontrolü ve indirme
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = FastAPI()

# CORS ayarları (geliştirme için geniş, prod ortamında daralt)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GPT4All modeli yükle
# Q4_K_M HIZ OLARAK LLAMA-3-8B DEN DAHA İYİ
#MODEL_PATH = r"C:\Users\Ghost\Downloads\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH = r"C:\Users\Ghost\PycharmProjects\LLM_EnvanterAI\Phi-3-mini-4k-instruct-q4.gguf"

#MODEL_PATH = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

#model = GPT4All(MODEL_PATH, n_threads=6)
model = GPT4All(MODEL_PATH)

class Query(BaseModel):
    question: str

def generate_short_answer(question: str, top_k: int = 2):
    """
    Soruya göre en alakalı chunk'ları retrieve edip kısa ve öz cevap üretir.
    """
    context_chunks = retrieve(question, top_k=top_k)  # liste of chunk metinleri
    if not context_chunks:
        return {"answer": "Üzgünüm, bu konuda elimizde bilgi yok.", "sources": []}

    # Prompt tasarımı
    prompt = (
        "Aşağıdaki kaynaklardan yararlanarak soruya kısa, öz ve net cevap ver.\n"
        "Cevap 1-2 cümleyi geçmesin ve kaynakları belirt.\n\n"
        "Kaynaklar:\n" +
        "\n".join([f"{i+1}. {c[:1000]}" for i, c in enumerate(context_chunks)]) +
        f"\n\nSoru: {question}\nCevap:"
    )

    # Modelden cevap üret
    raw_answer = model.generate(prompt)

    # Cümle bazında kısaltma (Türkçe)
    answer_sentences = sent_tokenize(raw_answer, language='turkish')
    answer = ". ".join(answer_sentences[:2]).rstrip(".") + "."

    return {
        "answer": answer,
        "sources": context_chunks
    }

@app.get("/ask")
def ask_question(q: str, top_k: int = 2):
    return generate_short_answer(q, top_k=top_k)

@app.post("/ask")
def ask(q: Query, top_k: int = 2):
    return generate_short_answer(q.question, top_k=top_k)
