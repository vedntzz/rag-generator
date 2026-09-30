"""Tests for RecursiveCharacterChunker."""

import pytest

from rag_generator.chunking.recursive_chunker import RecursiveCharacterChunker
from rag_generator.domain import Chunk, Document

SIZE, OVERLAP = 60, 10
MIXED_TEXT = (
    "Annual leave is 24 days. It accrues monthly. Unused days carry over.\n\n"
    "Sick leave is separate! Ask HR? A doctor's note is needed after three days.\n\n"
    + "x" * 150
    + "\n\nFinal paragraph. The end."
)


def split(text: str, size: int = SIZE, overlap: int = OVERLAP) -> list[Chunk]:
    chunker = RecursiveCharacterChunker(chunk_size=size, chunk_overlap=overlap)
    return chunker.split(Document(source="policy.md", text=text))


def reconstruct(chunks: list[Chunk], overlap: int) -> str:
    return chunks[0].text + "".join(chunk.text[overlap:] for chunk in chunks[1:])


def test_chunker_never_exceeds_chunk_size() -> None:
    assert all(len(chunk.text) <= SIZE for chunk in split(MIXED_TEXT))


def test_chunker_keeps_overlap_between_consecutive_chunks() -> None:
    chunks = split(MIXED_TEXT)
    pairs = zip(chunks, chunks[1:], strict=False)
    assert all(after.text[:OVERLAP] == before.text[-OVERLAP:] for before, after in pairs)


def test_chunker_reconstructs_original_text_without_dropping_any() -> None:
    assert reconstruct(split(MIXED_TEXT), OVERLAP) == MIXED_TEXT


def test_chunker_prefers_paragraph_boundary_over_later_sentence_boundary() -> None:
    text = "Leave is 24 days. Ask HR.\n\nNext. Second. Third. Fourth. Fifth sentence here."
    assert split(text)[0].text == "Leave is 24 days. Ask HR.\n\n"


def test_chunker_falls_back_to_sentence_boundary_without_paragraphs() -> None:
    text = "Leave is 24 days per year. It accrues monthly. Unused days carry over to next year."
    assert split(text)[0].text == "Leave is 24 days per year. It accrues monthly. "


def test_chunker_falls_back_to_character_boundary_without_sentences() -> None:
    chunks = split("x" * 150)
    assert [len(chunk.text) for chunk in chunks] == [60, 60, 50]


def test_chunker_returns_single_chunk_when_text_fits() -> None:
    assert [chunk.text for chunk in split("Short text.")] == ["Short text."]


@pytest.mark.parametrize("text", ["", "   \n\n\t  "])
def test_chunker_returns_no_chunks_for_empty_or_whitespace_text(text: str) -> None:
    assert split(text) == []


def test_chunker_numbers_chunks_from_zero_consecutively() -> None:
    chunks = split(MIXED_TEXT)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_chunker_tags_every_chunk_with_document_source() -> None:
    assert {chunk.source for chunk in split(MIXED_TEXT)} == {"policy.md"}


@pytest.mark.parametrize("overlap", [60, 61])
def test_chunker_rejects_overlap_not_smaller_than_size(overlap: int) -> None:
    with pytest.raises(ValueError):
        RecursiveCharacterChunker(chunk_size=60, chunk_overlap=overlap)


def test_chunker_defaults_to_size_800_and_overlap_100() -> None:
    chunker = RecursiveCharacterChunker()
    assert (chunker.chunk_size, chunker.chunk_overlap) == (800, 100)
