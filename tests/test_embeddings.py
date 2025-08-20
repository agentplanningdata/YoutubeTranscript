"""
핵심 임베딩 기능 테스트 - LangChain 기반

Phase 4.2의 핵심 기능들을 테스트합니다:
- OpenAI 임베딩 생성
- 자막 청크 벡터 저장
- 메타데이터와 함께 저장
- 중복 청크 처리
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

# LangChain imports
from langchain_core.documents import Document

from src.services.embedding_service import EmbeddingService, EmbeddingConfig
from src.utils.exceptions import EmbeddingError, VectorStoreError


@pytest.fixture
def sample_chunks():
    """테스트용 자막 청크들"""
    return [
        {
            'text': 'LangChain은 LLM 애플리케이션 프레임워크입니다.',
            'start_time': 0.0,
            'end_time': 3.0,
            'chunk_id': 'chunk_001',
            'word_count': 6,
            'char_count': 32
        },
        {
            'text': '벡터 저장소와 임베딩을 지원합니다.',
            'start_time': 3.0,
            'end_time': 6.0,
            'chunk_id': 'chunk_002',
            'word_count': 5,
            'char_count': 22
        }
    ]


class TestEmbeddingConfig:
    """임베딩 설정 테스트"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = EmbeddingConfig()
        
        assert config.provider == "openai"
        assert config.model_name == "text-embedding-ada-002"
        assert config.batch_size == 100
        assert config.max_retries == 3
        assert config.normalize_embeddings == True
    
    def test_openai_config(self):
        """OpenAI 설정 테스트"""
        config = EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-large",
            api_key="test_key"
        )
        
        assert config.provider == "openai"
        assert config.model_name == "text-embedding-3-large"
        assert config.api_key == "test_key"
    
    def test_huggingface_config(self):
        """HuggingFace 설정 테스트"""
        config = EmbeddingConfig(
            provider="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        assert config.provider == "huggingface"
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"


class TestGenerateEmbeddingsOpenAI:
    """OpenAI 임베딩 생성 테스트"""
    
    @patch('src.services.embedding_service.OpenAIEmbeddings')
    @patch('src.services.embedding_service.VectorDatabase')
    def test_generate_embeddings_openai(self, mock_vector_db, mock_openai_class, sample_chunks):
        """OpenAI 임베딩 생성 테스트"""
        
        # Mock OpenAI Embeddings
        mock_embeddings = MagicMock()
        mock_openai_class.return_value = mock_embeddings
        
        # Mock embed_documents return value
        mock_embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3],  # 첫 번째 문서
            [0.4, 0.5, 0.6]   # 두 번째 문서
        ]
        
        # 서비스 생성
        config = EmbeddingConfig(provider="openai", api_key="test_key")
        service = EmbeddingService(config)
        
        # 텍스트 리스트로 임베딩 생성
        texts = [chunk['text'] for chunk in sample_chunks]
        embeddings = service.generate_embeddings(texts)
        
        # 결과 검증
        mock_embeddings.embed_documents.assert_called_once_with(texts)
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
    
    @patch('src.services.embedding_service.OpenAIEmbeddings')
    @patch('src.services.embedding_service.VectorDatabase')
    def test_generate_single_embedding(self, mock_vector_db, mock_openai_class):
        """단일 쿼리 임베딩 생성 테스트"""
        
        # Mock setup
        mock_embeddings = MagicMock()
        mock_openai_class.return_value = mock_embeddings
        mock_embeddings.embed_query.return_value = [0.2, 0.3, 0.4]
        
        # 서비스 생성
        config = EmbeddingConfig(provider="openai", api_key="test_key")
        service = EmbeddingService(config)
        
        # 단일 쿼리 임베딩
        query = "LangChain에 대해 설명해주세요"
        embedding = service.generate_query_embedding(query)
        
        # 결과 검증
        mock_embeddings.embed_query.assert_called_once_with(query)
        assert embedding == [0.2, 0.3, 0.4]
    
    @patch('src.services.embedding_service.OpenAIEmbeddings')
    @patch('src.services.embedding_service.VectorDatabase') 
    def test_empty_input_error(self, mock_vector_db, mock_openai_class):
        """빈 입력 시 에러 테스트"""
        config = EmbeddingConfig(provider="openai")
        service = EmbeddingService(config)
        
        with pytest.raises(EmbeddingError):
            service.generate_embeddings([])


class TestStoreTranscriptChunks:
    """자막 청크 벡터 저장 테스트"""
    
    @patch('src.services.embedding_service.OpenAIEmbeddings')
    @patch('src.services.embedding_service.VectorDatabase')
    def test_store_transcript_chunks(self, mock_vector_db_class, mock_openai_class, sample_chunks):
        """자막 청크 벡터 저장 테스트"""
        
        # Mock setup
        mock_embeddings = MagicMock()
        mock_openai_class.return_value = mock_embeddings
        
        mock_vector_db = MagicMock()
        mock_vector_db_class.return_value = mock_vector_db
        mock_vector_db.add_documents.return_value = ['doc1', 'doc2']
        
        # 서비스 생성
        config = EmbeddingConfig(provider="openai")
        service = EmbeddingService(config)
        
        # 청크 저장
        result = service.store_transcript_chunks(sample_chunks)
        
        # 검증
        mock_vector_db.add_documents.assert_called_once()
        assert result['stored_count'] == 2
        assert result['document_ids'] == ['doc1', 'doc2']
        assert result['duplicates_removed'] == 0
        assert result['embedding_provider'] == 'openai'
    
    @patch('src.services.embedding_service.OpenAIEmbeddings')
    @patch('src.services.embedding_service.VectorDatabase')
    def test_chunks_to_documents_conversion(self, mock_vector_db, mock_openai_class, sample_chunks):
        """청크를 Document로 변환 테스트"""
        
        # 서비스 생성
        config = EmbeddingConfig(provider="openai")
        service = EmbeddingService(config)
        
        # Document 변환
        documents = service._chunks_to_documents(sample_chunks)
        
        # 검증
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        
        # 첫 번째 Document 검증
        doc = documents[0]
        assert doc.page_content == sample_chunks[0]['text']
        assert doc.metadata['chunk_id'] == 'chunk_001'
        assert doc.metadata['start_time'] == 0.0
        assert doc.metadata['end_time'] == 3.0


class TestStoreWithMetadata:
    """메타데이터와 함께 저장 테스트"""
    
    @patch('src.services.embedding_service.OpenAIEmbeddings')
    @patch('src.services.embedding_service.VectorDatabase')
    def test_store_with_metadata(self, mock_vector_db_class, mock_openai_class, sample_chunks):
        """메타데이터 보존하여 저장 테스트"""
        
        # Mock setup
        mock_vector_db = MagicMock()
        mock_vector_db_class.return_value = mock_vector_db
        mock_vector_db.add_documents.return_value = ['doc1', 'doc2']
        
        # 서비스 생성
        config = EmbeddingConfig(provider="openai")
        service = EmbeddingService(config)
        
        # 추가 메타데이터
        metadata = {
            'video_id': 'test_video_123',
            'video_title': 'LangChain Tutorial'
        }
        
        # 저장
        result = service.store_transcript_chunks(sample_chunks, global_metadata=metadata)
        
        # add_documents에 전달된 문서들 확인
        call_args = mock_vector_db.add_documents.call_args[0]
        documents = call_args[0]
        
        # 메타데이터 확인
        for doc in documents:
            assert doc.metadata['video_id'] == 'test_video_123'
            assert doc.metadata['video_title'] == 'LangChain Tutorial'
            assert 'chunk_id' in doc.metadata
            assert 'start_time' in doc.metadata


class TestDuplicateChunkHandling:
    """중복 청크 처리 테스트"""
    
    @patch('src.services.embedding_service.OpenAIEmbeddings')
    @patch('src.services.embedding_service.VectorDatabase')
    def test_duplicate_chunk_handling(self, mock_vector_db_class, mock_openai_class):
        """중복 청크 처리 테스트"""
        
        # Mock setup
        mock_vector_db = MagicMock()
        mock_vector_db_class.return_value = mock_vector_db
        mock_vector_db.add_documents.return_value = ['doc1', 'doc2']
        
        # 서비스 생성
        config = EmbeddingConfig(provider="openai")
        service = EmbeddingService(config)
        
        # 중복 청크 포함 데이터
        chunks_with_duplicates = [
            {
                'text': '같은 내용의 텍스트입니다.',
                'chunk_id': 'chunk_001',
                'start_time': 0.0,
                'end_time': 5.0
            },
            {
                'text': '다른 내용의 텍스트입니다.',
                'chunk_id': 'chunk_002', 
                'start_time': 5.0,
                'end_time': 10.0
            },
            {
                'text': '같은 내용의 텍스트입니다.',  # 중복
                'chunk_id': 'chunk_003',
                'start_time': 10.0,
                'end_time': 15.0
            }
        ]
        
        # 중복 제거 옵션 활성화
        result = service.store_transcript_chunks(
            chunks_with_duplicates,
            remove_duplicates=True
        )
        
        # 중복이 제거되어 2개만 저장되었는지 확인
        assert result['stored_count'] == 2
        assert result['duplicates_removed'] == 1
    
    @patch('src.services.embedding_service.OpenAIEmbeddings')
    @patch('src.services.embedding_service.VectorDatabase')
    def test_content_hash_generation(self, mock_vector_db, mock_openai_class):
        """콘텐츠 해시 생성 테스트"""
        
        # 서비스 생성
        config = EmbeddingConfig(provider="openai")
        service = EmbeddingService(config)
        
        # 동일한 텍스트, 다른 ID
        chunks = [
            {'text': 'Test content', 'chunk_id': 'chunk_001'},
            {'text': 'Test content', 'chunk_id': 'chunk_002'}  # 같은 내용
        ]
        
        # 해시 생성 테스트
        hashes = service._generate_content_hashes(chunks)
        
        # 동일한 내용은 같은 해시를 가져야 함
        assert hashes[0] == hashes[1]


class TestFactoryFunction:
    """팩토리 함수 테스트"""
    
    @patch('src.services.embedding_service.OpenAIEmbeddings')
    @patch('src.services.embedding_service.VectorDatabase')
    def test_create_embedding_service_openai(self, mock_vector_db, mock_openai):
        """OpenAI 임베딩 서비스 생성 테스트"""
        from src.services.embedding_service import create_embedding_service
        
        service = create_embedding_service(
            provider="openai",
            api_key="test_key",
            model_name="text-embedding-3-large"
        )
        
        assert isinstance(service, EmbeddingService)
        assert service.config.provider == "openai"
        assert service.config.model_name == "text-embedding-3-large" 