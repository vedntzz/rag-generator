"""Splits documents into overlapping chunks, cutting at the coarsest boundary that fits."""

import re

from rag_generator.domain import Chunk, Document

# Tried in order: paragraph break, then sentence end; a hard character cut is the fallback.
BOUNDARY_PATTERNS = (re.compile(r"\n\s*\n"), re.compile(r"(?<=[.!?])\s+"))


class RecursiveCharacterChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        if not document.text.strip():
            return []
        spans = self._chunk_spans(document.text)
        return [
            Chunk(source=document.source, index=index, text=document.text[start:end])
            for index, (start, end) in enumerate(spans)
        ]

    def _chunk_spans(self, text: str) -> list[tuple[int, int]]:
        # Each chunk starts chunk_overlap characters before the previous one ended.
        spans: list[tuple[int, int]] = []
        start = 0
        while True:
            end = self._choose_chunk_end(text, start)
            spans.append((start, end))
            if end == len(text):
                return spans
            start = end - self.chunk_overlap

    def _choose_chunk_end(self, text: str, start: int) -> int:
        limit = start + self.chunk_size
        if limit >= len(text):
            return len(text)
        # A cut must land past the overlap, or the next chunk would add no new text.
        earliest = start + self.chunk_overlap + 1
        boundary = find_preferred_boundary(text, start, earliest, limit)
        return limit if boundary is None else boundary


def find_preferred_boundary(text: str, start: int, earliest: int, latest: int) -> int | None:
    for pattern in BOUNDARY_PATTERNS:
        ends = [match.end() for match in pattern.finditer(text, start, latest)]
        valid = [end for end in ends if end >= earliest]
        if valid:
            return valid[-1]
    return None
