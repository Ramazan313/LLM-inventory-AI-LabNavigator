from docx import Document
from typing import List
import nltk
# Eksik paketleri otomatik indir
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

def sentence_split(text):
    """
    Verilen metni cümlelere böler.
    """
    # sent_tokenize default olarak punkt_tab modelini kullanacak
    sentences = sent_tokenize(text)
    return sentences
def extract_text_from_docx(docx_path: str) -> List[str]:
    """
    DOCX dosyasındaki cihaz bilgilerini blok blok çıkarır.
    Her cihaz bir chunk olarak döner.
    """
    doc = Document(docx_path)
    chunks = []
    cur_chunk = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue  # Boş satırları atla

        # "Cihaz Adı:" ile yeni blok başlıyor
        if text.startswith("Cihaz Adı:") and cur_chunk:
            chunks.append("\n".join(cur_chunk))
            cur_chunk = []

        cur_chunk.append(text)

    if cur_chunk:
        chunks.append("\n".join(cur_chunk))

    return chunks


def build_docx_chunks(docx_path: str) -> List[str]:


    blocks = extract_text_from_docx(docx_path)
    return blocks
