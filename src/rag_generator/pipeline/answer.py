"""Answer pipeline: embed -> search -> score filter -> LLM -> cited Answer."""

import re

from rag_generator.domain import Answer, Citation, LlmReply, ScoredChunk
from rag_generator.errors import CollectionNotFoundError
from rag_generator.llm.prompts import NOT_FOUND_MESSAGE, build_system_prompt, build_user_prompt
from rag_generator.ports import LLM, Embedder, VectorStore

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class AnswerService:
    def __init__(
        self, embedder: Embedder, store: VectorStore, llm: LLM, top_k: int, min_score: float
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def ask(self, collection: str, question: str) -> Answer:
        if not self.store.has_collection(collection):
            raise CollectionNotFoundError(collection)
        hits = self._retrieve_relevant_chunks(collection, question)
        if not hits:
            # Nothing relevant was retrieved, so the LLM is never given a chance to guess.
            return Answer(NOT_FOUND_MESSAGE, [], grounded=False)
        reply = self.llm.complete(build_system_prompt(), build_user_prompt(question, hits))
        return answer_from_reply(reply, hits)

    def _retrieve_relevant_chunks(self, collection: str, question: str) -> list[ScoredChunk]:
        hits = self.store.search(collection, self.embedder.embed_query(question), self.top_k)
        return [hit for hit in hits if hit.score >= self.min_score]


def answer_from_reply(reply: LlmReply, hits: list[ScoredChunk]) -> Answer:
    text = reply.text.strip()
    if not text or text == NOT_FOUND_MESSAGE:
        return Answer(NOT_FOUND_MESSAGE, [], grounded=False, truncated=reply.truncated)
    citations = cite_referenced_chunks(text, hits)
    return Answer(text, citations, grounded=bool(citations), truncated=reply.truncated)


def cite_referenced_chunks(text: str, hits: list[ScoredChunk]) -> list[Citation]:
    # [n] is 1-based into hits; numbers outside 1..len(hits) are ignored, not errors.
    numbers = sorted({int(number) for number in CITATION_PATTERN.findall(text)})
    return [citation_for(hits[n - 1], n) for n in numbers if 1 <= n <= len(hits)]


def citation_for(hit: ScoredChunk, reference: int) -> Citation:
    chunk = hit.chunk
    return Citation(chunk.source, chunk.index, hit.score, reference=reference)
