import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # exact search, simple and reliable

    def add(self, embeddings):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)

    def save(self, path):
        faiss.write_index(self.index, path)

    @staticmethod
    def load(path):
    
        return faiss.read_index(path)
