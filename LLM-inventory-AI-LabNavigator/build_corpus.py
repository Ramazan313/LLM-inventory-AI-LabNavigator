import pickle       #(list,dict,numpy array vs.) chunks ve embeddings diske kaydedilip tekrar açılmasını sağlar
import faiss        # büyük embedding vektörleri içinde en yakın komşuları hızlıca bulmak için kullanılır.
from sentence_transformers import SentenceTransformer   # Hugging face tabanlı transformer modellerini kullanarak embedding üretir
from utils import build_docx_chunks  # DOCX blok bazlı chunk çıkarma
from FlagEmbedding import BGEM3FlagModel
import numpy as np
INDEX_FILE = "faiss_index.idx"
CHUNKS_FILE = "chunks.pkl"
EMBEDDINGS_FILE = "embeddings.pkl"  # önbellek için

def build_index(docx_path: str):
    # DOCX cihaz bloklarını al
    chunks = build_docx_chunks(docx_path)

    #model = SentenceTransformer("intfloat/multilingual-e5-base")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    vectors = model.encode(chunks, convert_to_numpy=True)

    # FAISS index
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors)

    # Kaydet
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"{len(chunks)} blok ve FAISS index oluşturuldu ve cache’lendi.")

if __name__ == "__main__":
    docx_path = "Envanter.docx"
    build_index(docx_path)
