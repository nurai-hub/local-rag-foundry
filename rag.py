
import openai
import numpy as np
from pypdf import PdfReader

client = openai.OpenAI(
    base_url="http://127.0.0.1:40975/v1",
    api_key="not-needed"
)

CHAT_MODEL = "qwen2.5-1.5b"
EMBED_MODEL = "qwen3-embedding-0.6b"
PDF_FILE = "branch_prediction.pdf"

def clean_ligatures(text):
    replacements = {
        "5": "ti",
        "@": "a",
        "Z": "tt",
        "<": "tt",
        "\uf0be": "ti",
    }
    # Sadece bilinen kelime kaliplarinda degistir, rastgele rakam/sembolleri bozmamak icin
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
        "beZer": "better",
        "Solu5on": "Solution",
        "predica@on": "predication",
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

print("PDF okunuyor...")
raw_text = load_pdf_text(PDF_FILE)
chunks = chunk_text(raw_text)
print(f"Toplam {len(chunks)} parca (chunk) bulundu.\n")

def get_embedding(text_input):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text_input
    )
    return np.array(response.data[0].embedding)

print("Parcalar embeddinge cevriliyor, bu biraz surebilir...")
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
print("Tamamlandi.\n")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_most_relevant_chunk(question, top_k=2):
    q_embedding = get_embedding(question)
    similarities = [cosine_similarity(q_embedding, ce) for ce in chunk_embeddings]
    best_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in best_indices]

def ask(question):
    relevant_chunks = find_most_relevant_chunk(question, top_k=2)
    context = "\n\n---\n\n".join(relevant_chunks)

    print(f"\n[Bulunan ilgili bolum]:\n{context}\n")

    prompt = f"""Asagidaki baglami kullanarak soruyu cevapla. Baglamda olmayan bir bilgiyi uydurma.

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
