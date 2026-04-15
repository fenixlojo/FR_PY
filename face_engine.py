import insightface
import numpy as np


class FaceEngine:

    def __init__(self):
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=-1)

    def get_embedding(self, image):
        faces = self.app.get(image)

        if len(faces) == 0:
            return None

        # Cara más grande
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        # IMPORTANTE: embedding normalizado
        return face.normed_embedding

    @staticmethod
    def cosine_similarity(a, b):
        return float(np.dot(a, b))

    def compare(self, face_embeddings, doc_embedding, threshold=0.35):
        matches = []
        similarities = []

        for emb in face_embeddings:
            sim = self.cosine_similarity(emb, doc_embedding)
            similarities.append(sim)
            matches.append(sim >= threshold)

        return matches, similarities

    @staticmethod
    def compute_score(similarities):
        return float(np.mean(similarities))