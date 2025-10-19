from fastapi import FastAPI # Resr API servislerini kurmak için kullanılır.
from fastapi.middleware.cors import CORSMiddleware  # Başka domainlerden bu API'ye erişim izni vermek için kullanılır.
from pydantic import BaseModel  #Post request içinde question alanını zorunlu hale getirir.
from retriever_mongo import retrieve    # Sorguya uygun PDF chunklarını getiriyor(FAISS üzerinden)
from gpt4all import GPT4All     # Yerel LLM çalıştırmak için kullanılan kütüphane

app = FastAPI()

# CORS ayarları
origins = ["*"]  # tüm originlere izin verir, production için bu daraltılabilir.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model yükleme
MODEL_NAME = r"C:\Users\Ghost\Downloads\mistral-7b-instruct-v0.2.Q3_K_S.gguf"
model = GPT4All(MODEL_NAME, n_threads=4, n_ctx=1048)  # işlemci 12 çekirdekli yükselirse cevap süresi kısalıyor ama cevap doğruluğu düşüyor

class Query(BaseModel):
    question: str

@app.get("/ask")
def ask_question(q: str):
    response = model.generate(q)
    return {"answer": response}


@app.post("/ask")
def ask(q: Query):
    # context al, top_k chunk
    context = retrieve(q.question, top_k=3)

    # prompt için karakter sınırı (isteğe bağlı)
    if context:
        context = [c[:1000] for c in context]  # 1000 karakter ile daha anlamlı

    prompt = (
            "PDF içeriğine dayanarak çok kısa ve öz cevap ver (1-2 cümle, tek paragraf):\n\n"
            + "\n\n".join(context)
            + f"\n\nSoru: {q.question}\nCevap:"
    )

    # Streaming ile cevap üret ve biriktir
    answer_tokens = []
    for token in model.generate(prompt, streaming=True):
        answer_tokens.append(token)

    answer = "".join(answer_tokens)
    # ilk 2 cümle
    answer = ". ".join(answer.split(".")[:2]) + "."

    return {"answer": answer, "sources": context}
