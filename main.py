from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import numpy as np
import cv2

from face_engine import FaceEngine
from schemas import CompareResponse
from config import SIMILARITY_THRESHOLD, MIN_MATCHES

app = FastAPI()
engine = FaceEngine()


def read_image(file: UploadFile):
    contents = file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Imagen inválida")

    return img  # InsightFace usa BGR directamente


@app.post("/fr/compare", response_model=CompareResponse)
async def compare_faces(
    faces: List[UploadFile] = File(...),
    document: UploadFile = File(...)
):
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="Se requiere al menos una imagen de rostro")

    # 📄 Documento
    doc_img = read_image(document)
    doc_embedding = engine.get_embedding(doc_img)

    if doc_embedding is None:
        raise HTTPException(status_code=400, detail="No se detectó rostro en el documento")

    # 👤 Faces
    face_embeddings = []

    for f in faces:
        img = read_image(f)
        emb = engine.get_embedding(img)

        if emb is not None:
            face_embeddings.append(emb)

    if len(face_embeddings) == 0:
        raise HTTPException(status_code=400, detail="No se detectaron rostros en las imágenes")

    # 🔍 Comparación
    matches, similarities = engine.compare(
        face_embeddings,
        doc_embedding,
        threshold=0.35
    )

    score = engine.compute_score(similarities)

    success = sum(matches) >= MIN_MATCHES

    best_similarity = max(similarities)

    return CompareResponse(
        matches=matches,
        similarities=similarities,
        best_similarity=max(similarities),
        score=score,
        success=success
    )