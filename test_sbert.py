from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Deploy service in Docker",
    "How to run containerized application"
]

embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)
