import os
import glob
import openai
import numpy as np
from pypdf import PdfReader

BASE_URL = os.environ.get("FOUNDRY_BASE_URL", "http://127.0.0.1:40975/v1")

client = openai.OpenAI(
    base_url=BASE_URL,
    api_key="not-needed"
)

CHAT_MODEL = "qwen2.5-1.5b"
EMBED_MODEL = "qwen3-embedding-0.6b"
DOCUMENTS_FOLDER = "documents"
SIMILARITY_THRESHOLD = 0.4

def clean_ligatures(text):
    known_fixes = {
        "predic5on": "prediction",
        "predic@on": "prediction",
        "5me": "time",
        "5mes": "times",
        "mul5ple": "multiple",
        "instruc5on": "instruction",
        "instruc5ons": "instructions",
        "Rela5ve": "Relative",
        "poten5al": "potential",
        "Eﬀec5veness": "Effectiveness",
        "Mispredic5on": "Misprediction",
        "run-­‐5me": "run-time",
        "correla5ng": "correlating",
        "paZern": "pattern",
        "pa<ern": "pattern",
        "beZer": "better",
        "Solu5on": "Solution",
        "predica@on": "predication",
        "Predic5on": "Prediction",
    }
    for wrong, right in known_fixes.items():
        text = text.replace(wrong, right)
    return text

def load_pdf_text(path):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return clean_ligatures(full_text)

def chunk_text(text, chunk_size=800, overlap=150):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def load_all_documents(folder):
    pdf_paths = glob.glob(os.path.join(folder, "*.pdf"))
    all_chunks = []
    for path in pdf_paths:
        filename = os.path.basename(path)
        print(f"Okunuyor: {filename}")
        text = load_pdf_text(path)
        doc_chunks = chunk_text(text)
        for chunk in doc_chunks:
            all_chunks.append({"source": filename, "text": chunk})
    return all_chunks

print(f"'{DOCUMENTS_FOLDER}' klasorundeki PDF'ler okunuyor...")
chunks = load_all_documents(DOCUMENTS_FOLDER)
print(f"\nToplam {len(chunks)} parca (chunk) bulundu.\n")

def get_embedding(text_input):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text_input
    )
    return np.array(response.data[0].embedding)

print("Parcalar embeddinge cevriliyor, bu biraz surebilir...")
chunk_embeddings = [get_embedding(c["text"]) for c in chunks]
print("Tamamlandi.\n")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_most_relevant_chunk(question, top_k=2):
    q_embedding = get_embedding(question)
    similarities = [cosine_similarity(q_embedding, ce) for ce in chunk_embeddings]
    best_indices = np.argsort(similarities)[::-1][:top_k]
    best_score = similarities[best_indices[0]]
    return [chunks[i] for i in best_indices], best_score

def ask(question):
    relevant_chunks, best_score = find_most_relevant_chunk(question, top_k=2)

    print(f"\n[Benzerlik skoru]: {best_score:.3f}")

    if best_score < SIMILARITY_THRESHOLD:
        print("[Cevap]: Bu konuda dokumanlarimda yeterli bilgi bulamadim. Baska bir soru deneyebilir misin?\n")
        return

    sources_used = sorted(set(c["source"] for c in relevant_chunks))
    print(f"[Kullanilan kaynaklar]: {', '.join(sources_used)}\n")

    context = "\n\n---\n\n".join(
        f"[Kaynak: {c['source']}]\n{c['text']}" for c in relevant_chunks
    )

    prompt = f"""Asagidaki baglami kullanarak soruyu cevapla. Eger baglamda sorunun cevabi yoksa, "Bu konuda dokumanlarda yeterli bilgi bulamadim" de. Baglamda olmayan bir bilgiyi uydurma.

Baglam:
{context}

Soru: {question}

Cevap:"""

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=200,
        temperature=0.3
    )

    print("[Cevap]:")
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    print("Ders Asistani hazir! Sorularini yaz, cikmak icin q yaz.\n")
    while True:
        question = input("Soru: ")
        if question.lower() in ["q", "quit", "exit"]:
            print("Gorusuruz!")
            break
        ask(question)
