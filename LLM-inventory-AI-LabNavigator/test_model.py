import time
import csv
from gpt4all import GPT4All
from retriever_mongo import retrieve
from difflib import SequenceMatcher

# Model yükleme
MODEL_NAME = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
model = GPT4All(MODEL_NAME, n_threads=6)

# Test soruları ve beklenen cevapları
tests = [
    {
        "question": "Metal bir kaşık üreticisiyim hangi deneyleri yaptırabilirim?",
        "expected": "Malzeme dayanımı, korozyon ve sertlik testleri yapılabilir."
    },
    {
        "question": "Alan emisyonlu taramalı elektron mikroskobunda ne yapılır?",
        "expected": "SEM ile numune yüzey yapısı ve kompozisyonu incelenir."
    },
    {
        "question": "Seramik üretiminde kullanılan fırın tipleri nelerdir?",
        "expected": "Asansörlü fırın, kamara fırın ve kutu tipi fırınlar kullanılır."
    },
    {
        "question": "Spektrum analizi yaptırmak istiyorum hangi laboratuvarda yaptırabilirim?",
        "expected": "Spektrum analizini SEMAM laboratuvarında yaptırabilirsiniz."
    }
]

def similarity(a, b):
    """Basit string benzerliği ölçümü"""
    return SequenceMatcher(None, a, b).ratio()

results = []

for t in tests:
    start_time = time.time()

    # Retriever ile context al
    context = retrieve(t["question"], top_k=2)
    if context:
        context = [c[:500] for c in context]  # sadece ilk 500 karakter
    prompt = (
        "PDF içeriğine dayanarak 1-2 cümle ile kısa ve öz cevap ver:\n\n"
        + "\n\n".join(context)
        + f"\n\nSoru: {t['question']}\nCevap:"
    )

    # Model cevabı
    answer = model.generate(prompt, max_tokens=100)
    end_time = time.time()

    elapsed = end_time - start_time
    score = similarity(answer, t["expected"])

    print(f"Q: {t['question']}")
    print(f"A: {answer}")
    print(f"Expected: {t['expected']}")
    print(f"Similarity: {score:.2f}")
    print(f"Response Time: {elapsed:.2f} seconds\n")

    results.append({
        "question": t["question"],
        "expected": t["expected"],
        "response": answer,
        "similarity": score,
        "time_seconds": elapsed
    })

# CSV olarak kaydet
with open("model_test_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Test tamamlandı, sonuçlar model_test_results.csv içinde.")
