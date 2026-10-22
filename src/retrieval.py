"""Embed chunks with Sentence Transformers and search them with Qdrant."""
from dataclasses import asdict, dataclass

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.ingestion import Chunk, build_chunks, load_documents

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
COLLECTION = "northstar_kb"
TOP_K = 3
# Chosen on calibration.json (see src/calibration.py): highest abstain top score 0.441,
# lowest answer top score 0.496 -> midpoint. Frozen before running questions.json.
MIN_SCORE = 0.47


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float  # cosine similarity, higher = more similar


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device="cpu")
        self.client = QdrantClient(":memory:")

        # 1. Create an empty collection: one vector per chunk, compared by cosine similarity.
        self.client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=self.model.get_embedding_dimension(),  # 384 for MiniLM
                distance=models.Distance.COSINE,
            ),
        )

        # 2. Embed all chunk texts in one batch (shape: n_chunks x 384).
        vectors = self.model.encode([c.text for c in chunks], normalize_embeddings=True)

        # 3. Store each vector with its chunk metadata as the payload.
        self.client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(id=i, vector=vec.tolist(), payload=asdict(chunk))
                for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )

    def search(self, question: str, k: int = TOP_K) -> list[SearchHit]:
        """Return the k chunks most similar to the question, best first."""
        qvec = self.model.encode(question, normalize_embeddings=True)
        result = self.client.query_points(
            COLLECTION, query=qvec.tolist(), limit=k, with_payload=True
        )
        return [SearchHit(chunk=Chunk(**p.payload), score=p.score) for p in result.points]

    def retrieve_evidence(
        self, question: str, k: int = TOP_K, min_score: float = MIN_SCORE
    ) -> list[SearchHit]:
        """Top-k hits if the best one clears the threshold, else [] (= abstain).

        The gate looks only at the top score: filtering every hit would drop
        lower-ranked chunks that still hold the answer (e.g. billing-refunds#0 for cal01).
        """
        hits = self.search(question, k)
        if not hits or hits[0].score < min_score:
            return []
        return hits


if __name__ == "__main__":
    chunks = build_chunks(load_documents())
    retriever = Retriever(chunks)
    print(f"Indexed {len(chunks)} chunks with {MODEL_NAME}@{MODEL_REVISION[:8]}\n")

    question = "Which payment details must I include when reporting a duplicate charge?"
    print(f"Q: {question}")
    for hit in retriever.search(question):
        print(f"  {hit.score:.3f}  {hit.chunk.chunk_id}") 
        #print(hit.chunk)

