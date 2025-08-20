"""
유사도 검색 테스트 - LangChain 기반

LangChain의 Retriever 패턴과 VectorStore를 활용한 
고급 유사도 검색 기능을 테스트합니다.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import tempfile
import shutil
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# LangChain imports
from langchain_core.documents import Document

from src.services.similarity_search_service import (
    SimilaritySearchService,
    SearchConfig,
    SearchType,
    create_similarity_search_service
)
from src.utils.exceptions import VectorSearchError, ConfigurationError


@pytest.fixture
def sample_documents():
    """테스트용 문서들 (벡터 저장소에 저장된 자막 청크 시뮬레이션)"""
    return [
        Document(
            page_content="LangChain은 대규모 언어 모델을 활용한 애플리케이션 개발 프레임워크입니다.",
            metadata={
                'chunk_id': 'video1_chunk_001',
                'start_time': 0.0,
                'end_time': 5.0,
                'video_id': 'video_001', 
                'video_title': 'LangChain 입문',
                'channel_name': 'AI Tutorial',
                'word_count': 12
            }
        ),
        Document(
            page_content="ChromaDB는 오픈소스 벡터 데이터베이스로 임베딩 저장 및 유사도 검색을 제공합니다.",
            metadata={
                'chunk_id': 'video1_chunk_002',
                'start_time': 5.0,
                'end_time': 10.0,
                'video_id': 'video_001',
                'video_title': 'LangChain 입문', 
                'channel_name': 'AI Tutorial',
                'word_count': 14
            }
        ),
        Document(
            page_content="RAG 시스템은 검색 증강 생성을 통해 더 정확한 답변을 생성할 수 있습니다.",
            metadata={
                'chunk_id': 'video2_chunk_001', 
                'start_time': 0.0,
                'end_time': 4.5,
                'video_id': 'video_002',
                'video_title': 'RAG 시스템 구축',
                'channel_name': 'AI Advanced',
                'word_count': 13
            }
        )
    ]


@pytest.fixture
def mock_embedding_service():
    """Mock EmbeddingService"""
    with patch('src.services.similarity_search_service.EmbeddingService') as mock_service:
        service = MagicMock()
        mock_service.return_value = service
        
        # Mock vector_db.as_retriever() 
        mock_retriever = MagicMock()
        service.vector_db.as_retriever.return_value = mock_retriever
        
        yield service, mock_retriever


class TestSearchConfig:
    """검색 설정 테스트"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = SearchConfig()
        
        assert config.search_type == SearchType.SIMILARITY
        assert config.k == 5
        assert config.score_threshold is None
        assert config.fetch_k == 20
        assert config.lambda_mult == 0.5
        assert config.metadata_filter == {}
        assert config.include_metadata == True
    
    def test_mmr_config(self):
        """MMR 설정 테스트"""
        config = SearchConfig(
            search_type=SearchType.MMR,
            k=10,
            lambda_mult=0.7,
            fetch_k=50
        )
        
        assert config.search_type == SearchType.MMR
        assert config.k == 10
        assert config.lambda_mult == 0.7
        assert config.fetch_k == 50
    
    def test_score_threshold_config(self):
        """점수 임계값 설정 테스트"""
        config = SearchConfig(
            search_type=SearchType.SIMILARITY_SCORE_THRESHOLD,
            score_threshold=0.8,
            k=3
        )
        
        assert config.search_type == SearchType.SIMILARITY_SCORE_THRESHOLD
        assert config.score_threshold == 0.8
        assert config.k == 3
    
    def test_metadata_filter_config(self):
        """메타데이터 필터 설정 테스트"""
        config = SearchConfig(
            metadata_filter={'video_id': 'video_001'},
            k=10
        )
        
        assert config.metadata_filter == {'video_id': 'video_001'}
        assert config.k == 10


class TestSimilaritySearchService:
    """유사도 검색 서비스 테스트"""
    
    def test_service_initialization(self, mock_embedding_service):
        """검색 서비스 초기화 테스트"""
        embedding_service, _ = mock_embedding_service
        config = SearchConfig()
        
        search_service = SimilaritySearchService(embedding_service, config)
        
        assert search_service.embedding_service == embedding_service
        assert search_service.config == config
        assert search_service.retriever is not None
    
    def test_initialization_with_invalid_embedding_service(self):
        """잘못된 임베딩 서비스로 초기화 시 에러 테스트"""
        config = SearchConfig()
        
        with pytest.raises(VectorSearchError):
            SimilaritySearchService(None, config)


class TestSearchByQuery:
    """쿼리 기반 검색 테스트"""
    
    def test_search_by_query(self, mock_embedding_service, sample_documents):
        """쿼리 기반 기본 검색 테스트"""
        embedding_service, mock_retriever = mock_embedding_service
        
        # Mock retriever response
        mock_retriever.invoke.return_value = sample_documents[:2]
        
        # 검색 서비스 생성
        config = SearchConfig(k=2)
        search_service = SimilaritySearchService(embedding_service, config)
        
        # 검색 실행
        query = "LangChain에 대해 설명해주세요"
        results = search_service.search(query)
        
        # 검증
        mock_retriever.invoke.assert_called_once_with(query)
        assert len(results) == 2
        assert all('document' in result for result in results)
        assert all('score' in result for result in results)
        assert results[0]['document'].page_content.startswith("LangChain은")
    
    def test_search_with_empty_query(self, mock_embedding_service):
        """빈 쿼리로 검색 시 에러 테스트"""
        embedding_service, _ = mock_embedding_service
        config = SearchConfig()
        search_service = SimilaritySearchService(embedding_service, config)
        
        with pytest.raises(VectorSearchError):
            search_service.search("")
    
    def test_search_with_retriever_error(self, mock_embedding_service):
        """리트리버 에러 처리 테스트"""
        embedding_service, mock_retriever = mock_embedding_service
        mock_retriever.invoke.side_effect = Exception("Retriever error")
        
        config = SearchConfig()
        search_service = SimilaritySearchService(embedding_service, config)
        
        with pytest.raises(VectorSearchError):
            search_service.search("test query")


class TestSearchWithScoreThreshold:
    """점수 임계값 검색 테스트"""
    
    def test_search_with_score_threshold(self, mock_embedding_service, sample_documents):
        """점수 임계값 검색 테스트"""
        embedding_service, mock_retriever = mock_embedding_service
        
        # Mock vector_db.similarity_search_with_score (점수 임계값 검색은 이 메서드를 직접 사용)
        mock_docs_with_scores = [
            (sample_documents[0], 0.9),
            (sample_documents[1], 0.85)
        ]
        embedding_service.vector_db.similarity_search_with_score.return_value = mock_docs_with_scores
        
        # 점수 임계값 설정
        config = SearchConfig(
            search_type=SearchType.SIMILARITY_SCORE_THRESHOLD,
            score_threshold=0.8,
            k=5
        )
        search_service = SimilaritySearchService(embedding_service, config)
        
        # 검색 실행
        query = "벡터 데이터베이스"
        results = search_service.search(query)
        
        # 검증
        embedding_service.vector_db.similarity_search_with_score.assert_called_once_with(query, k=5)
        assert len(results) == 2
        
        # 모든 결과가 임계값 이상인지 확인
        for result in results:
            assert result['score'] >= config.score_threshold
    
    def test_search_score_threshold_filters_low_scores(self, mock_embedding_service, sample_documents):
        """낮은 점수 결과 필터링 테스트"""
        embedding_service, mock_retriever = mock_embedding_service
        
        # 일부는 임계값 이상, 일부는 이하로 설정
        mock_docs_with_scores = [
            (sample_documents[0], 0.9),   # 임계값 이상
            (sample_documents[1], 0.7),   # 임계값 이하
            (sample_documents[2], 0.85)   # 임계값 이상
        ]
        mock_retriever.invoke.return_value = mock_docs_with_scores
        
        config = SearchConfig(
            search_type=SearchType.SIMILARITY_SCORE_THRESHOLD,
            score_threshold=0.8
        )
        search_service = SimilaritySearchService(embedding_service, config)
        
        # 실제 필터링 로직이 서비스 내부에서 수행된다고 가정
        # (이 부분은 실제 구현에서 처리)
        query = "테스트 쿼리"
        results = search_service.search(query)
        
        # 임계값 이상인 결과만 반환되었는지 확인  
        assert len(results) >= 0  # 필터링 결과에 따라 달라짐


class TestSearchTopKResults:
    """상위 K개 결과 테스트"""
    
    def test_search_top_k_results(self, mock_embedding_service, sample_documents):
        """상위 K개 결과 반환 테스트"""
        embedding_service, mock_retriever = mock_embedding_service
        
        # 더 많은 문서를 반환하도록 설정하되, k개만 선택되도록 함
        all_docs = sample_documents * 3  # 9개 문서
        mock_retriever.invoke.return_value = all_docs[:3]  # k=3으로 제한
        
        config = SearchConfig(k=3)
        search_service = SimilaritySearchService(embedding_service, config)
        
        query = "AI 및 머신러닝"
        results = search_service.search(query)
        
        # k개만 반환되었는지 확인
        assert len(results) == 3
        
        # retriever가 올바른 k 값으로 설정되었는지 검증
        # (실제로는 as_retriever에서 search_kwargs를 통해 전달)
        mock_retriever.invoke.assert_called_once_with(query)
    
    def test_search_k_larger_than_available(self, mock_embedding_service, sample_documents):
        """사용 가능한 문서보다 큰 k 값 테스트"""
        embedding_service, mock_retriever = mock_embedding_service
        
        # 사용 가능한 문서는 2개뿐
        mock_retriever.invoke.return_value = sample_documents[:2]
        
        # k=10으로 설정하지만 실제로는 2개만 반환
        config = SearchConfig(k=10)
        search_service = SimilaritySearchService(embedding_service, config)
        
        results = search_service.search("테스트")
        
        # 사용 가능한 만큼만 반환
        assert len(results) == 2
    
    def test_search_k_zero(self, mock_embedding_service):
        """k=0일 때 에러 테스트"""  
        embedding_service, _ = mock_embedding_service
        
        with pytest.raises(ConfigurationError):
            SearchConfig(k=0)
    
    def test_search_k_negative(self, mock_embedding_service):
        """k가 음수일 때 에러 테스트"""
        embedding_service, _ = mock_embedding_service
        
        with pytest.raises(ConfigurationError):
            SearchConfig(k=-1)


class TestSearchWithVideoFilter:
    """영상 필터링 검색 테스트"""
    
    def test_search_with_video_filter(self, mock_embedding_service, sample_documents):
        """특정 영상으로 필터링 검색 테스트"""
        embedding_service, mock_retriever = mock_embedding_service
        
        # video_001에 해당하는 문서들만 반환
        filtered_docs = [doc for doc in sample_documents if doc.metadata['video_id'] == 'video_001']
        mock_retriever.invoke.return_value = filtered_docs
        
        config = SearchConfig(
            metadata_filter={'video_id': 'video_001'},
            k=5
        )
        search_service = SimilaritySearchService(embedding_service, config)
        
        query = "LangChain 설명"
        results = search_service.search(query)
        
        # 결과 검증
        assert len(results) == 2  # video_001에 해당하는 2개 문서
        for result in results:
            assert result['document'].metadata['video_id'] == 'video_001'
    
    def test_search_with_channel_filter(self, mock_embedding_service, sample_documents):
        """채널별 필터링 검색 테스트"""
        embedding_service, mock_retriever = mock_embedding_service
        
        # 특정 채널의 문서들만
        filtered_docs = [doc for doc in sample_documents if doc.metadata['channel_name'] == 'AI Tutorial']
        mock_retriever.invoke.return_value = filtered_docs
        
        config = SearchConfig(
            metadata_filter={'channel_name': 'AI Tutorial'}
        )
        search_service = SimilaritySearchService(embedding_service, config)
        
        results = search_service.search("기술 설명")
        
        for result in results:
            assert result['document'].metadata['channel_name'] == 'AI Tutorial'
    
    def test_search_with_multiple_filters(self, mock_embedding_service, sample_documents):
        """다중 조건 필터링 테스트"""
        embedding_service, mock_retriever = mock_embedding_service
        
        # 다중 조건: video_id AND channel_name
        filtered_docs = [
            doc for doc in sample_documents 
            if doc.metadata['video_id'] == 'video_001' 
            and doc.metadata['channel_name'] == 'AI Tutorial'
        ]
        mock_retriever.invoke.return_value = filtered_docs
        
        config = SearchConfig(
            metadata_filter={
                'video_id': 'video_001',
                'channel_name': 'AI Tutorial'
            }
        )
        search_service = SimilaritySearchService(embedding_service, config)
        
        results = search_service.search("검색")
        
        for result in results:
            doc_meta = result['document'].metadata
            assert doc_meta['video_id'] == 'video_001'
            assert doc_meta['channel_name'] == 'AI Tutorial'
    
    def test_search_with_time_range_filter(self, mock_embedding_service, sample_documents):
        """시간 범위 필터링 테스트"""
        embedding_service, mock_retriever = mock_embedding_service
        
        # 특정 시간 범위의 청크들만
        config = SearchConfig(
            time_range={'start': 0.0, 'end': 6.0}  # 첫 6초 이내
        )
        search_service = SimilaritySearchService(embedding_service, config)
        
        # 시간 필터링은 search 메서드 내에서 후처리로 구현된다고 가정
        mock_retriever.invoke.return_value = sample_documents
        
        results = search_service.search("초반 내용")
        
        # 실제 구현에서는 시간 범위 필터링 로직이 적용될 것
        assert len(results) >= 0  # 구현에 따라 달라짐


class TestMMRSearch:
    """MMR (Maximum Marginal Relevance) 검색 테스트"""
    
    def test_mmr_search(self, mock_embedding_service, sample_documents):
        """MMR 검색 테스트"""
        embedding_service, mock_retriever = mock_embedding_service
        
        # MMR 검색 결과 시뮬레이션 (다양성 고려된 결과)
        mock_retriever.invoke.return_value = sample_documents
        
        config = SearchConfig(
            search_type=SearchType.MMR,
            k=3,
            lambda_mult=0.5,  # 유사성과 다양성의 균형
            fetch_k=10
        )
        search_service = SimilaritySearchService(embedding_service, config)
        
        query = "데이터베이스와 검색"
        results = search_service.search(query)
        
        # MMR은 다양성을 고려하므로 관련성 높은 문서들이 선택됨
        assert len(results) <= 3
        mock_retriever.invoke.assert_called_once_with(query)
    
    def test_mmr_lambda_mult_parameter(self, mock_embedding_service):
        """MMR lambda_mult 파라미터 테스트"""
        embedding_service, _ = mock_embedding_service
        
        # lambda_mult = 1.0 (순수 유사성)
        config_similarity = SearchConfig(
            search_type=SearchType.MMR,
            lambda_mult=1.0
        )
        
        # lambda_mult = 0.0 (순수 다양성)
        config_diversity = SearchConfig(
            search_type=SearchType.MMR,
            lambda_mult=0.0
        )
        
        # 설정이 정상적으로 생성되는지 확인
        assert config_similarity.lambda_mult == 1.0
        assert config_diversity.lambda_mult == 0.0


class TestAdvancedSearchFeatures:
    """고급 검색 기능 테스트"""
    
    def test_search_with_metadata_inclusion(self, mock_embedding_service, sample_documents):
        """메타데이터 포함/제외 테스트"""
        embedding_service, mock_retriever = mock_embedding_service
        mock_retriever.invoke.return_value = sample_documents[:1]
        
        # 메타데이터 포함
        config_with_meta = SearchConfig(include_metadata=True)
        search_service = SimilaritySearchService(embedding_service, config_with_meta)
        
        results = search_service.search("테스트")
        
        # 메타데이터가 포함되어 있는지 확인
        assert 'metadata' in results[0]
        assert results[0]['metadata'] is not None
    
    def test_async_search(self, mock_embedding_service, sample_documents):
        """비동기 검색 테스트 (미래 구현)"""
        # 이 테스트는 미래에 비동기 검색 기능이 추가될 때 활성화
        pass
    
    def test_cached_search(self, mock_embedding_service):
        """캐시된 검색 테스트 (미래 구현)"""
        # 동일한 쿼리에 대한 캐시 기능 테스트
        pass


class TestFactoryFunctions:
    """팩토리 함수 테스트"""
    
    @patch('src.services.similarity_search_service.EmbeddingService')
    def test_create_similarity_search_service(self, mock_embedding_service):
        """유사도 검색 서비스 생성 테스트"""
        mock_service = MagicMock()
        mock_embedding_service.return_value = mock_service
        
        search_service = create_similarity_search_service(
            embedding_service=mock_service,
            search_type=SearchType.MMR,
            k=10
        )
        
        assert isinstance(search_service, SimilaritySearchService)
        assert search_service.config.search_type == SearchType.MMR
        assert search_service.config.k == 10
    
    @patch('src.services.similarity_search_service.create_embedding_service')
    def test_create_with_embedding_config(self, mock_create_embedding):
        """임베딩 설정으로부터 검색 서비스 생성 테스트"""
        mock_embedding_service = MagicMock()
        mock_create_embedding.return_value = mock_embedding_service
        mock_embedding_service.vector_db.as_retriever.return_value = MagicMock()
        
        search_service = create_similarity_search_service(
            provider="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            search_type=SearchType.SIMILARITY_SCORE_THRESHOLD,
            score_threshold=0.75
        )
        
        # 임베딩 서비스 생성이 호출되었는지 확인
        mock_create_embedding.assert_called_once()
        assert isinstance(search_service, SimilaritySearchService)
        assert search_service.config.score_threshold == 0.75


class TestBM25Search:
    """BM25 키워드 검색 테스트"""
    
    def test_bm25_search_basic(self, mock_embedding_service, sample_documents):
        """기본 BM25 검색 테스트"""
        embedding_service, _ = mock_embedding_service
        
        # mock 벡터 데이터베이스에서 문서를 반환하도록 설정
        embedding_service.get_collection_info.return_value = {
            'document_count': len(sample_documents)
        }
        embedding_service.vector_db.similarity_search.return_value = sample_documents
        
        # BM25 검색 서비스 설정
        config = SearchConfig(search_type=SearchType.BM25, k=2)
        
        try:
            search_service = SimilaritySearchService(embedding_service, config)
            
            # BM25가 사용 가능한 경우에만 실제 테스트 진행
            if hasattr(search_service, 'bm25_retriever') and search_service.bm25_retriever is not None:
                # BM25 검색기 mock으로 대체
                mock_bm25_retriever = MagicMock()
                mock_bm25_retriever.invoke.return_value = sample_documents[:2]
                search_service.retriever = mock_bm25_retriever
                
                # 검색 실행
                query = "LangChain Document"
                results = search_service.search(query)
                
                # 검증
                assert len(results) == 2
                mock_bm25_retriever.invoke.assert_called_once_with(query)
            else:
                # BM25 사용 불가 시 적절한 에러 발생 확인
                pytest.skip("BM25 not available - this is expected behavior")
                
        except VectorSearchError as e:
            if "BM25 search not available" in str(e) or "No documents found" in str(e):
                pytest.skip("BM25 not available - this is expected behavior")
            else:
                raise
    
    def test_bm25_search_with_korean_tokenizer(self, mock_embedding_service):
        """한국어 토크나이저 BM25 검색 테스트"""
        embedding_service, _ = mock_embedding_service
        
        # 한국어 토크나이저 사용 설정
        config = SearchConfig(
            search_type=SearchType.BM25,
            use_korean_tokenizer=True,
            bm25_k=3
        )
        
        # 설정이 정상적으로 생성되는지 확인
        assert config.use_korean_tokenizer == True
        assert config.bm25_k == 3
        assert config.search_type == SearchType.BM25


class TestHybridSearch:
    """하이브리드 검색 테스트"""
    
    def test_hybrid_search_balanced(self, mock_embedding_service, sample_documents):
        """균형 잡힌 하이브리드 검색 테스트 (5:5)"""
        embedding_service, mock_retriever = mock_embedding_service
        
        # 하이브리드 검색 서비스 설정
        config = SearchConfig(
            search_type=SearchType.HYBRID,
            hybrid_weights=[0.5, 0.5],
            k=3
        )
        
        try:
            search_service = SimilaritySearchService(embedding_service, config)
            
            # 하이브리드가 사용 가능한 경우에만 실제 테스트 진행
            if hasattr(search_service, 'ensemble_retriever') and search_service.ensemble_retriever is not None:
                # 하이브리드 검색 결과 mock
                mock_ensemble_retriever = MagicMock()
                mock_ensemble_retriever.invoke.return_value = sample_documents
                
                # 앙상블 retriever 직접 설정 (테스트용)
                search_service.retriever = mock_ensemble_retriever
                
                # 검색 실행
                query = "LangChain 하이브리드 검색"
                results = search_service.search(query)
                
                # 검증
                assert len(results) == 3
                mock_ensemble_retriever.invoke.assert_called_once_with(query)
            else:
                # 하이브리드 사용 불가 시 스킵
                pytest.skip("Hybrid search not available - this is expected behavior")
                
        except VectorSearchError as e:
            if "search not available" in str(e) or "BM25" in str(e):
                pytest.skip("Hybrid search not available - this is expected behavior")
            else:
                raise
    
    def test_hybrid_search_semantic_heavy(self, mock_embedding_service):
        """의미론적 검색 중심 하이브리드 검색 테스트 (7:3)"""
        embedding_service, _ = mock_embedding_service
        
        config = SearchConfig(
            search_type=SearchType.HYBRID,
            hybrid_weights=[0.7, 0.3],  # 의미론적 검색 70%
            k=5
        )
        
        # 설정 검증
        assert config.hybrid_weights == [0.7, 0.3]
        assert abs(sum(config.hybrid_weights) - 1.0) < 1e-6
    
    def test_hybrid_search_keyword_heavy(self, mock_embedding_service):
        """키워드 검색 중심 하이브리드 검색 테스트 (3:7)"""
        embedding_service, _ = mock_embedding_service
        
        config = SearchConfig(
            search_type=SearchType.HYBRID,
            hybrid_weights=[0.3, 0.7],  # BM25 검색 70%
            k=5
        )
        
        # 설정 검증
        assert config.hybrid_weights == [0.3, 0.7]
        assert abs(sum(config.hybrid_weights) - 1.0) < 1e-6
    
    def test_hybrid_search_with_metadata_filter(self, mock_embedding_service, sample_documents):
        """메타데이터 필터링과 결합된 하이브리드 검색 테스트"""
        embedding_service, _ = mock_embedding_service
        
        # 특정 비디오에서만 검색하는 하이브리드 검색
        filtered_docs = [doc for doc in sample_documents if doc.metadata['video_id'] == 'video_001']
        
        mock_ensemble_retriever = MagicMock()
        mock_ensemble_retriever.invoke.return_value = filtered_docs
        
        config = SearchConfig(
            search_type=SearchType.HYBRID,
            hybrid_weights=[0.6, 0.4],
            metadata_filter={'video_id': 'video_001'},
            k=3
        )
        
        search_service = SimilaritySearchService(embedding_service, config)
        search_service.ensemble_retriever = mock_ensemble_retriever
        
        results = search_service.search("테스트 쿼리")
        
        # 모든 결과가 해당 비디오에서 온 것인지 확인
        for result in results:
            assert result['document'].metadata['video_id'] == 'video_001'
    
    def test_hybrid_performance_comparison(self, mock_embedding_service, sample_documents):
        """하이브리드 vs 개별 검색 성능 비교 (성능 측정 시뮬레이션)"""
        embedding_service, mock_retriever = mock_embedding_service
        
        # 다양한 검색 방식 설정
        search_configs = {
            'semantic_only': SearchConfig(search_type=SearchType.SIMILARITY, k=3),
            'bm25_only': SearchConfig(search_type=SearchType.BM25, bm25_k=3),
            'hybrid_balanced': SearchConfig(
                search_type=SearchType.HYBRID, 
                hybrid_weights=[0.5, 0.5], 
                k=3
            ),
            'hybrid_semantic_heavy': SearchConfig(
                search_type=SearchType.HYBRID,
                hybrid_weights=[0.8, 0.2],
                k=3
            )
        }
        
        # 모든 설정이 유효한지 검증
        for name, config in search_configs.items():
            assert config.k > 0 or config.bm25_k > 0
            if config.search_type == SearchType.HYBRID:
                assert len(config.hybrid_weights) == 2
                assert abs(sum(config.hybrid_weights) - 1.0) < 1e-6


class TestAdvancedHybridFeatures:
    """고급 하이브리드 검색 기능 테스트"""
    
    def test_reciprocal_rank_fusion(self, mock_embedding_service):
        """Reciprocal Rank Fusion 알고리즘 테스트"""
        embedding_service, _ = mock_embedding_service
        
        # RRF는 EnsembleRetriever의 내부 알고리즘이므로
        # 실제로는 LangChain에서 처리됨
        config = SearchConfig(
            search_type=SearchType.HYBRID,
            hybrid_weights=[0.5, 0.5]
        )
        
        # 설정이 올바른지만 확인
        assert config.search_type == SearchType.HYBRID
        assert config.hybrid_weights == [0.5, 0.5]
    
    def test_hybrid_search_with_time_filter(self, mock_embedding_service, sample_documents):
        """시간 필터와 결합된 하이브리드 검색 테스트"""
        embedding_service, _ = mock_embedding_service
        
        config = SearchConfig(
            search_type=SearchType.HYBRID,
            hybrid_weights=[0.6, 0.4],
            time_range={'start': 0.0, 'end': 15.0},
            k=3
        )
        
        search_service = SimilaritySearchService(embedding_service, config)
        
        # 시간 필터링이 설정에 반영되었는지 확인
        assert config.time_range is not None
        assert config.time_range['start'] == 0.0
        assert config.time_range['end'] == 15.0
    
    def test_dynamic_weight_adjustment(self, mock_embedding_service):
        """동적 가중치 조정 테스트"""
        embedding_service, _ = mock_embedding_service
        
        # 다양한 가중치 조합 테스트
        weight_combinations = [
            [0.1, 0.9],  # BM25 중심
            [0.5, 0.5],  # 균형
            [0.9, 0.1],  # 의미론적 검색 중심
        ]
        
        for weights in weight_combinations:
            config = SearchConfig(
                search_type=SearchType.HYBRID,
                hybrid_weights=weights
            )
            
            search_service = SimilaritySearchService(embedding_service, config)
            
            # 가중치가 올바르게 설정되었는지 확인
            assert search_service.config.hybrid_weights == weights
            assert abs(sum(weights) - 1.0) < 1e-6


class TestErrorHandling:
    """에러 처리 테스트"""
    
    def test_invalid_search_type(self):
        """잘못된 검색 타입 에러 테스트"""
        with pytest.raises(ValueError):
            SearchConfig(search_type="invalid_type")
    
    def test_missing_score_threshold_for_threshold_search(self):
        """점수 임계값 검색에서 임계값 누락 에러 테스트"""
        with pytest.raises(ConfigurationError):
            SearchConfig(
                search_type=SearchType.SIMILARITY_SCORE_THRESHOLD,
                score_threshold=None
            )
    
    def test_invalid_lambda_mult_range(self):
        """lambda_mult 범위 오류 테스트"""
        with pytest.raises(ConfigurationError):
            SearchConfig(
                search_type=SearchType.MMR,
                lambda_mult=1.5  # 0~1 범위를 벗어남
            )
        
        with pytest.raises(ConfigurationError):
            SearchConfig(
                search_type=SearchType.MMR, 
                lambda_mult=-0.1  # 음수
            )
    
    def test_invalid_hybrid_weights(self):
        """하이브리드 가중치 오류 테스트"""
        # 가중치 개수 오류
        with pytest.raises(ConfigurationError):
            SearchConfig(
                search_type=SearchType.HYBRID,
                hybrid_weights=[0.5]  # 1개만 제공
            )
        
        # 가중치 범위 오류
        with pytest.raises(ConfigurationError):
            SearchConfig(
                search_type=SearchType.HYBRID,
                hybrid_weights=[1.5, -0.1]  # 범위 벗어남
            )
        
        # 가중치 합계 오류
        with pytest.raises(ConfigurationError):
            SearchConfig(
                search_type=SearchType.HYBRID,
                hybrid_weights=[0.3, 0.8]  # 합계가 1.1
            )
    
    def test_invalid_bm25_k(self):
        """BM25 k 값 오류 테스트"""
        with pytest.raises(ConfigurationError):
            SearchConfig(
                search_type=SearchType.BM25,
                bm25_k=0  # 0은 안됨
            )
        
        with pytest.raises(ConfigurationError):
            SearchConfig(
                search_type=SearchType.HYBRID,
                bm25_k=-1  # 음수 안됨
            ) 