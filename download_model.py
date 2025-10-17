from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("modelos/all-MiniLM-L6-v2")  # salva localmente
