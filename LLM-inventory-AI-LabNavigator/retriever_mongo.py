import faiss    # Daha önce kaydedilen FAISS indexi(faiss_index.idx) okumak için
import pickle   # Kaydedilmiş chunksçpkl ve embeddings.pkl dosyalarını açmak için
from sentence_transformers import SentenceTransformer   # Sorguyu embedding'e çevirmek için tekrar kullanılır
from FlagEmbedding import BGEM3FlagModel

INDEX_FILE = "faiss_index.idx"
CHUNKS_FILE = "chunks.pkl"
#EMBEDDINGS_FILE = "embeddings.pkl"
#model = SentenceTransformer("intfloat/multilingual-e5-base")
model = SentenceTransformer("all-MiniLM-L6-v2")

#model = BGEM3FlagModel('BAAI/bge-m3',
                      # use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

#embeddings_1 = model.encode(sentences_1,
#                            batch_size=12,
#                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
#                            )['dense_vecs']
#embeddings_2 = model.encode(sentences_2)['dense_vecs']
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)

similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# Yükle
index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)
#with open(EMBEDDINGS_FILE, "rb") as f:
#    embeddings = pickle.load(f)
def retrieve(query, top_k=2):
    """
    Sorguyu embedding'e çevirip FAISS ile top-k chunk döndürür
    """
    q_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_vec, top_k)
    return [(chunks[i], float(distances[0][idx])) for idx, i in enumerate(indices[0])]

    #return [chunks[i] for i in indices[0]]