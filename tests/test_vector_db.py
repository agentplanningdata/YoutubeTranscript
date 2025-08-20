"""
벡터 데이터베이스 (LangChain Chroma) 테스트

LangChain Chroma와 OpenAI/HuggingFace Embeddings를 사용한 
벡터 데이터베이스 기능을 테스트합니다.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import tempfile
import shutil
import os
from pathlib import Path
from typing import List, Dict, Any

# LangChain imports
from langchain_core.documents import Document

from src.database.vector_db import (
    VectorDatabase, 
    ChromaDBConfig,
    create_openai_vector_db,
    create_huggingface_vector_db
)
from src.utils.exceptions import ChromaDBError, ConfigurationError


@pytest.fixture
def sample_documents():
    """테스트용 샘플 문서들"""
    return [
        Document(
            page_content="LangChain은 LLM 애플리케이션을 구축하기 위한 프레임워크입니다.",
            metadata={"source": "docs", "topic": "framework"}
        ),
        Document(
            page_content="ChromaDB는 AI 네이티브 벡터 데이터베이스입니다.",
            metadata={"source": "docs", "topic": "database"}
        ),
        Document(
            page_content="OpenAI의 임베딩 모델은 텍스트를 벡터로 변환합니다.",
            metadata={"source": "docs", "topic": "embeddings"}
        )
    ]


@pytest.fixture
def temp_directory():
    """임시 디렉토리"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestChromaDBConfig:
    """ChromaDBConfig 클래스 테스트"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = ChromaDBConfig()
        
        assert config.collection_name == "youtube_transcripts"
        assert config.persist_directory == os.path.abspath("./chromadb_langchain")
        assert config.embedding_provider == "openai"
        assert config.openai_model == "text-embedding-3-large"
        assert config.huggingface_model == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_custom_config(self):
        """커스텀 설정 테스트"""
        config = ChromaDBConfig(
            collection_name="test_collection",
            persist_directory="./test_db",
            embedding_provider="huggingface",
            huggingface_model="sentence-transformers/paraphrase-MiniLM-L6-v2"
        )
        
        assert config.collection_name == "test_collection"
        assert config.persist_directory == os.path.abspath("./test_db")
        assert config.embedding_provider == "huggingface"
        assert config.huggingface_model == "sentence-transformers/paraphrase-MiniLM-L6-v2"
    
    def test_invalid_collection_name(self):
        """잘못된 컬렉션 이름 검증"""
        with pytest.raises(ValueError, match="Collection name cannot be empty"):
            ChromaDBConfig(collection_name="")
        
        with pytest.raises(ValueError, match="Collection name cannot be empty"):
            ChromaDBConfig(collection_name="   ")
    
    def test_openai_api_key_detection(self):
        """OpenAI API 키 감지 테스트"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            config = ChromaDBConfig(embedding_provider="openai")
            assert config.openai_api_key == "test-api-key"


class TestVectorDatabaseWithHuggingFace:
    """HuggingFace Embeddings을 사용한 VectorDatabase 테스트"""
    
    @patch('src.database.vector_db.Chroma')
    @patch('src.database.vector_db.HuggingFaceEmbeddings')
    def test_huggingface_initialization(self, mock_hf_embeddings, mock_chroma, temp_directory):
        """HuggingFace 임베딩으로 초기화 테스트"""
        # Mock 설정
        mock_embeddings_instance = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings_instance
        
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        
        # 설정
        config = ChromaDBConfig(
            collection_name="test_collection",
            persist_directory=temp_directory,
            embedding_provider="huggingface"
        )
        
        # VectorDatabase 생성
        vector_db = VectorDatabase(config)
        
        # 검증
        assert vector_db.vectorstore == mock_vectorstore
        assert vector_db.embeddings == mock_embeddings_instance
        
        # HuggingFaceEmbeddings가 올바른 파라미터로 호출되었는지 확인
        mock_hf_embeddings.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Chroma가 올바른 파라미터로 호출되었는지 확인
        mock_chroma.assert_called_once_with(
            collection_name="test_collection",
            embedding_function=mock_embeddings_instance,
            persist_directory=temp_directory
        )
    
    @patch('src.database.vector_db.Chroma')
    @patch('src.database.vector_db.HuggingFaceEmbeddings')
    def test_add_documents(self, mock_hf_embeddings, mock_chroma, sample_documents):
        """문서 추가 테스트"""
        # Mock 설정
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        mock_vectorstore.add_documents.return_value = ["id1", "id2", "id3"]
        
        # VectorDatabase 생성
        config = ChromaDBConfig(embedding_provider="huggingface")
        vector_db = VectorDatabase(config)
        
        # 문서 추가
        result_ids = vector_db.add_documents(sample_documents)
        
        # 검증
        assert result_ids == ["id1", "id2", "id3"]
        mock_vectorstore.add_documents.assert_called_once_with(
            documents=sample_documents, ids=None
        )
    
    @patch('src.database.vector_db.Chroma')
    @patch('src.database.vector_db.HuggingFaceEmbeddings')
    def test_add_texts(self, mock_hf_embeddings, mock_chroma):
        """텍스트 추가 테스트"""
        # Mock 설정
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        mock_vectorstore.add_texts.return_value = ["id1", "id2"]
        
        # VectorDatabase 생성
        config = ChromaDBConfig(embedding_provider="huggingface")
        vector_db = VectorDatabase(config)
        
        # 텍스트 추가
        texts = ["첫 번째 텍스트", "두 번째 텍스트"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        
        result_ids = vector_db.add_texts(texts, metadatas=metadatas)
        
        # 검증
        assert result_ids == ["id1", "id2"]
        mock_vectorstore.add_texts.assert_called_once_with(
            texts=texts, metadatas=metadatas, ids=None
        )
    
    @patch('src.database.vector_db.Chroma')
    @patch('src.database.vector_db.HuggingFaceEmbeddings')
    def test_similarity_search(self, mock_hf_embeddings, mock_chroma, sample_documents):
        """유사도 검색 테스트"""
        # Mock 설정
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        mock_vectorstore.similarity_search.return_value = sample_documents[:2]
        
        # VectorDatabase 생성
        config = ChromaDBConfig(embedding_provider="huggingface")
        vector_db = VectorDatabase(config)
        
        # 유사도 검색
        results = vector_db.similarity_search("LangChain 프레임워크", k=2)
        
        # 검증
        assert len(results) == 2
        assert results == sample_documents[:2]
        mock_vectorstore.similarity_search.assert_called_once_with(
            "LangChain 프레임워크", k=2
        )
    
    @patch('src.database.vector_db.Chroma')
    @patch('src.database.vector_db.HuggingFaceEmbeddings')
    def test_similarity_search_with_score(self, mock_hf_embeddings, mock_chroma, sample_documents):
        """점수 포함 유사도 검색 테스트"""
        # Mock 설정
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        mock_results = [(sample_documents[0], 0.85), (sample_documents[1], 0.75)]
        mock_vectorstore.similarity_search_with_score.return_value = mock_results
        
        # VectorDatabase 생성
        config = ChromaDBConfig(embedding_provider="huggingface")
        vector_db = VectorDatabase(config)
        
        # 점수 포함 검색
        results = vector_db.similarity_search_with_score("벡터 데이터베이스", k=2)
        
        # 검증
        assert len(results) == 2
        assert results[0][1] == 0.85
        assert results[1][1] == 0.75
        mock_vectorstore.similarity_search_with_score.assert_called_once_with(
            "벡터 데이터베이스", k=2
        )
    
    @patch('src.database.vector_db.Chroma')
    @patch('src.database.vector_db.HuggingFaceEmbeddings')
    def test_as_retriever(self, mock_hf_embeddings, mock_chroma):
        """Retriever 변환 테스트"""
        # Mock 설정
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        # VectorDatabase 생성
        config = ChromaDBConfig(embedding_provider="huggingface")
        vector_db = VectorDatabase(config)
        
        # Retriever 변환
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        
        # 검증
        assert retriever == mock_retriever
        mock_vectorstore.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )


class TestVectorDatabaseWithOpenAI:
    """OpenAI Embeddings를 사용한 VectorDatabase 테스트"""
    
    @patch('src.database.vector_db.Chroma')
    @patch('src.database.vector_db.OpenAIEmbeddings')
    def test_openai_initialization_success(self, mock_openai_embeddings, mock_chroma, temp_directory):
        """OpenAI 임베딩으로 성공적인 초기화 테스트"""
        # Mock 설정
        mock_embeddings_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        
        # 설정
        config = ChromaDBConfig(
            collection_name="openai_test",
            persist_directory=temp_directory,
            embedding_provider="openai",
            openai_api_key="test-api-key"
        )
        
        # VectorDatabase 생성
        vector_db = VectorDatabase(config)
        
        # 검증
        assert vector_db.vectorstore == mock_vectorstore
        assert vector_db.embeddings == mock_embeddings_instance
        
        # OpenAIEmbeddings가 올바른 파라미터로 호출되었는지 확인
        mock_openai_embeddings.assert_called_once_with(
            model="text-embedding-3-large",
            openai_api_key="test-api-key"
        )
    
    def test_openai_initialization_failure_no_api_key(self):
        """OpenAI API 키 없을 때 실패 테스트"""
        # 환경 변수에서도 API 키가 없다고 가정
        with patch.dict(os.environ, {}, clear=True):
            config = ChromaDBConfig(
                embedding_provider="openai",
                openai_api_key=None
            )
            
            with pytest.raises(ChromaDBError):
                VectorDatabase(config)


class TestVectorDatabaseCollectionManagement:
    """컬렉션 관리 기능 테스트"""
    
    @patch('src.database.vector_db.Chroma')
    @patch('src.database.vector_db.HuggingFaceEmbeddings')
    def test_get_collection_info(self, mock_hf_embeddings, mock_chroma):
        """컬렉션 정보 조회 테스트"""
        # Mock 설정
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        
        # Mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.metadata = {"description": "test collection"}
        mock_vectorstore._collection = mock_collection
        
        # VectorDatabase 생성
        config = ChromaDBConfig(
            collection_name="info_test",
            embedding_provider="huggingface"
        )
        vector_db = VectorDatabase(config)
        
        # 컬렉션 정보 조회
        info = vector_db.get_collection_info()
        
        # 검증
        assert info["name"] == "info_test"
        assert info["count"] == 100
        assert info["metadata"] == {"description": "test collection"}
        assert info["embedding_provider"] == "huggingface"
        assert "embedding_model" in info
    
    @patch('src.database.vector_db.Chroma')
    @patch('src.database.vector_db.HuggingFaceEmbeddings')
    def test_delete_documents(self, mock_hf_embeddings, mock_chroma):
        """문서 삭제 테스트"""
        # Mock 설정
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        mock_vectorstore.delete.return_value = True
        
        # VectorDatabase 생성
        config = ChromaDBConfig(embedding_provider="huggingface")
        vector_db = VectorDatabase(config)
        
        # 문서 삭제
        result = vector_db.delete(["id1", "id2", "id3"])
        
        # 검증
        assert result is True
        mock_vectorstore.delete.assert_called_once_with(ids=["id1", "id2", "id3"])
    
    @patch('src.database.vector_db.Chroma')
    @patch('src.database.vector_db.HuggingFaceEmbeddings')
    def test_reset_collection(self, mock_hf_embeddings, mock_chroma):
        """컬렉션 리셋 테스트"""
        # Mock 설정
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        
        # Mock collection with data
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["id1", "id2", "id3"]}
        mock_vectorstore._collection = mock_collection
        
        # VectorDatabase 생성
        config = ChromaDBConfig(embedding_provider="huggingface")
        vector_db = VectorDatabase(config)
        
        # 컬렉션 리셋
        result = vector_db.reset_collection()
        
        # 검증
        assert result is True
        mock_collection.get.assert_called_once()
        mock_collection.delete.assert_called_once_with(ids=["id1", "id2", "id3"])


class TestFactoryFunctions:
    """팩토리 함수 테스트"""
    
    @patch('src.database.vector_db.VectorDatabase')
    def test_create_openai_vector_db(self, mock_vector_db):
        """OpenAI 벡터DB 생성 팩토리 함수 테스트"""
        mock_instance = MagicMock()
        mock_vector_db.return_value = mock_instance
        
        # 팩토리 함수 호출
        result = create_openai_vector_db(
            collection_name="test_openai",
            persist_directory="./test_openai_db",
            openai_model="text-embedding-3-small"
        )
        
        # 검증
        assert result == mock_instance
        
        # VectorDatabase가 올바른 설정으로 호출되었는지 확인
        args, kwargs = mock_vector_db.call_args
        config = args[0]
        
        assert config.collection_name == "test_openai"
        assert config.persist_directory == os.path.abspath("./test_openai_db")
        assert config.embedding_provider == "openai"
        assert config.openai_model == "text-embedding-3-small"
    
    @patch('src.database.vector_db.VectorDatabase')
    def test_create_huggingface_vector_db(self, mock_vector_db):
        """HuggingFace 벡터DB 생성 팩토리 함수 테스트"""
        mock_instance = MagicMock()
        mock_vector_db.return_value = mock_instance
        
        # 팩토리 함수 호출
        result = create_huggingface_vector_db(
            collection_name="test_hf",
            persist_directory="./test_hf_db",
            model_name="sentence-transformers/distilbert-base-nli-mean-tokens"
        )
        
        # 검증
        assert result == mock_instance
        
        # VectorDatabase가 올바른 설정으로 호출되었는지 확인
        args, kwargs = mock_vector_db.call_args
        config = args[0]
        
        assert config.collection_name == "test_hf"
        assert config.persist_directory == os.path.abspath("./test_hf_db")
        assert config.embedding_provider == "huggingface"
        assert config.huggingface_model == "sentence-transformers/distilbert-base-nli-mean-tokens"


class TestVectorDatabaseIntegration:
    """통합 테스트 (실제 기능 사용)"""
    
    @pytest.mark.skip(reason="실제 모델을 사용하므로 CI/CD에서는 스킵")
    def test_real_huggingface_integration(self, temp_directory):
        """실제 HuggingFace 모델을 사용한 통합 테스트"""
        config = ChromaDBConfig(
            collection_name="real_test",
            persist_directory=temp_directory,
            embedding_provider="huggingface",
            huggingface_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        with VectorDatabase(config) as vector_db:
            # 텍스트 추가
            texts = [
                "Python은 프로그래밍 언어입니다.",
                "LangChain은 LLM 프레임워크입니다.",
                "ChromaDB는 벡터 데이터베이스입니다."
            ]
            
            ids = vector_db.add_texts(texts)
            assert len(ids) == 3
            
            # 검색 테스트
            results = vector_db.similarity_search("프로그래밍", k=2)
            assert len(results) <= 2
            
            # 컬렉션 정보 확인
            info = vector_db.get_collection_info()
            assert info["count"] == 3


class TestErrorHandling:
    """에러 처리 테스트"""
    
    @patch('src.database.vector_db.Chroma')
    @patch('src.database.vector_db.HuggingFaceEmbeddings')
    def test_uninitialized_vectorstore_error(self, mock_hf_embeddings, mock_chroma):
        """초기화되지 않은 벡터스토어 에러 테스트"""
        # VectorDatabase 생성 후 vectorstore를 None으로 설정
        config = ChromaDBConfig(embedding_provider="huggingface")
        vector_db = VectorDatabase(config)
        vector_db.vectorstore = None
        
        # 에러가 발생해야 하는 메서드들 테스트
        with pytest.raises(ChromaDBError, match="VectorStore not initialized"):
            vector_db.add_documents([])
        
        with pytest.raises(ChromaDBError, match="VectorStore not initialized"):
            vector_db.similarity_search("test")
        
        with pytest.raises(ChromaDBError, match="VectorStore not initialized"):
            vector_db.as_retriever()
    
    @patch('src.database.vector_db.HuggingFaceEmbeddings')
    def test_embedding_function_fallback(self, mock_hf_embeddings):
        """임베딩 함수 생성 실패 시 fallback 테스트"""
        # HuggingFaceEmbeddings가 실패하도록 설정
        mock_hf_embeddings.side_effect = [Exception("Model load failed"), MagicMock()]
        
        config = ChromaDBConfig(embedding_provider="unknown_provider")
        
        with patch('src.database.vector_db.Chroma'):
            # 경고가 발생하지만 fallback으로 동작해야 함
            vector_db = VectorDatabase(config)
            
            # fallback HuggingFaceEmbeddings가 호출되었는지 확인
            assert mock_hf_embeddings.call_count == 2  # 첫 시도 실패 후 fallback 호출 