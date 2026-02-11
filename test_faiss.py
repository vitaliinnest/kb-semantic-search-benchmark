import numpy as np
import faiss

# випадкові вектори (384 як у MiniLM)
vectors = np.random.rand(10, 384).astype("float32")

index = faiss.IndexFlatIP(384)  # cosine через нормалізацію
faiss.normalize_L2(vectors)

index.add(vectors)

query = np.random.rand(1, 384).astype("float32")
faiss.normalize_L2(query)

distances, indices = index.search(query, 3)

print("Top-3 indices:", indices)
print("Scores:", distances)
