"""Tests for the Knowledge Agent's RAG tools.

These tests require the databases and ChromaDB to be set up first
(run notebooks 01 and 02).
"""
import json
import pytest
from unittest.mock import patch, MagicMock


class TestKnowledgeRetriever:
    """Test the RAG retriever directly."""

    def test_retriever_search_returns_list(self):
        """Mock test to verify retriever returns expected structure."""
        from agentic.rag.retriever import KnowledgeRetriever

        retriever = KnowledgeRetriever()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1 content", "doc2 content"]],
            "metadatas": [[
                {"title": "Article 1", "tags": "tag1"},
                {"title": "Article 2", "tags": "tag2"},
            ]],
            "distances": [[0.3, 0.8]],
        }
        retriever._collection = mock_collection

        results = retriever.search("test query")
        assert len(results) == 2
        assert "article_id" in results[0]
        assert "title" in results[0]
        assert "confidence" in results[0]

    def test_confidence_scoring(self):
        """Verify confidence = 1 / (1 + distance)."""
        from agentic.rag.retriever import KnowledgeRetriever

        retriever = KnowledgeRetriever()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["content"]],
            "metadatas": [[{"title": "Test", "tags": ""}]],
            "distances": [[0.5]],
        }
        retriever._collection = mock_collection

        results = retriever.search("test")
        expected_confidence = 1.0 / (1.0 + 0.5)
        assert abs(results[0]["confidence"] - round(expected_confidence, 4)) < 0.001

    def test_search_above_threshold(self):
        """Test threshold filtering."""
        from agentic.rag.retriever import KnowledgeRetriever

        retriever = KnowledgeRetriever()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[
                {"title": "Relevant", "tags": ""},
                {"title": "Irrelevant", "tags": ""},
            ]],
            "distances": [[0.1, 5.0]],  # first high confidence, second low
        }
        retriever._collection = mock_collection

        results, has_confident = retriever.search_above_threshold("test")
        assert has_confident is True
        assert len(results) == 2

    def test_no_results(self):
        """Test empty results handling."""
        from agentic.rag.retriever import KnowledgeRetriever

        retriever = KnowledgeRetriever()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        retriever._collection = mock_collection

        results = retriever.search("nonsense query xyz")
        assert results == []


class TestKnowledgeTools:
    """Test the knowledge tools with mocked retriever."""

    def test_search_knowledge_returns_json(self):
        with patch("agentic.tools.knowledge_tools.retriever") as mock_ret:
            mock_ret.search_above_threshold.return_value = (
                [{"article_id": "1", "title": "Test", "content": "Content", "confidence": 0.9, "tags": ""}],
                True,
            )
            from agentic.tools.knowledge_tools import search_knowledge
            result = json.loads(search_knowledge.invoke({"query": "test"}))
            assert result["has_confident_match"] is True
            assert len(result["articles"]) == 1

    def test_search_knowledge_low_confidence(self):
        with patch("agentic.tools.knowledge_tools.retriever") as mock_ret:
            mock_ret.search_above_threshold.return_value = (
                [{"article_id": "1", "title": "Test", "content": "Content", "confidence": 0.3, "tags": ""}],
                False,
            )
            from agentic.tools.knowledge_tools import search_knowledge
            result = json.loads(search_knowledge.invoke({"query": "unknown topic"}))
            assert result["has_confident_match"] is False
            assert "message" in result
