"""
Thin wrapper around qdrant_client for ticket search.

TicketVectorStore is an ANN filter/ranking index only -- it never becomes
the source of truth for ticket content. Callers (see src/search/indexing.py
and app/tools.py) always re-fetch authoritative field values from SQLite
after getting candidate ticket ids back from here.
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models

# Point payload carries only what's needed to filter server-side; it is
# deliberately not a copy of subject/body/summary (see indexing.py).
STATUS_FIELD = "status"
CATEGORY_FIELD = "category"


class TicketVectorStore:
    def __init__(self, client: QdrantClient, collection: str, vector_size: int):
        self._client = client
        self._collection = collection
        self._vector_size = vector_size

    def ensure_collection(self) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(
                    size=self._vector_size, distance=models.Distance.COSINE
                ),
            )

    def recreate_collection(self) -> None:
        """Drop and recreate the collection empty, for a full batch reindex."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self.ensure_collection()

    def upsert_many(self, points: list[tuple[int, list[float], dict]]) -> None:
        if not points:
            return
        self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(id=ticket_id, vector=vector, payload=payload)
                for ticket_id, vector, payload in points
            ],
        )

    def upsert_one(self, ticket_id: int, vector: list[float], payload: dict) -> None:
        self.upsert_many([(ticket_id, vector, payload)])

    def search(
        self,
        vector: list[float],
        *,
        category: str | None,
        statuses: tuple[str, ...],
        limit: int,
    ) -> list[tuple[int, float]]:
        conditions = [
            models.FieldCondition(
                key=STATUS_FIELD, match=models.MatchAny(any=list(statuses))
            )
        ]
        if category is not None:
            conditions.append(
                models.FieldCondition(
                    key=CATEGORY_FIELD, match=models.MatchValue(value=category)
                )
            )
        response = self._client.query_points(
            self._collection,
            query=vector,
            query_filter=models.Filter(must=conditions),
            limit=limit,
        )
        return [(point.id, point.score) for point in response.points]
