"""
유사도 검색 서비스 - LangChain 기반

LangChain의 Retriever 패턴과 VectorStore를 활용하여
고급 유사도 검색 기능을 제공합니다.
"""

from typing import List, Dict, Any, Optional, Union
import enum
from dataclasses import dataclass, field
import logging

# LangChain Core Imports  
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.services.embedding_service import EmbeddingService, create_embedding_service
from src.utils.exceptions import VectorSearchError, ConfigurationError
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)

# LangChain Community Imports (BM25 & Ensemble)
try:
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("BM25Retriever not available. Install 'rank_bm25' and 'langchain-community' for BM25 search.")

# Korean Tokenizer (Optional)
try:
    from kiwipiepy import Kiwi
    KOREAN_TOKENIZER_AVAILABLE = True
except ImportError:
    KOREAN_TOKENIZER_AVAILABLE = False


class SearchType(enum.Enum):
    """검색 타입 열거형"""
    SIMILARITY = "similarity"  # 기본 유사도 검색
    MMR = "mmr"  # Maximum Marginal Relevance
    SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"  # 점수 임계값 검색
    BM25 = "bm25"  # BM25 키워드 검색
    HYBRID = "hybrid"  # 하이브리드 검색 (의미론적 + BM25)


@dataclass
class SearchConfig:
    """검색 설정 클래스"""
    
    # 기본 검색 설정
    search_type: SearchType = SearchType.SIMILARITY
    k: int = 5  # 반환할 결과 수
    
    # 점수 임계값 설정
    score_threshold: Optional[float] = None
    
    # MMR 설정
    fetch_k: int = 20  # MMR을 위해 가져올 후보 문서 수
    lambda_mult: float = 0.5  # 유사성(1.0) vs 다양성(0.0) 균형
    
    # 필터링 설정
    metadata_filter: Dict[str, Any] = field(default_factory=dict)
    time_range: Optional[Dict[str, float]] = None  # {'start': 0.0, 'end': 10.0}
    
    # BM25 및 하이브리드 검색 설정
    hybrid_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])  # [semantic, bm25]
    use_korean_tokenizer: bool = False  # 한국어 토크나이저 사용 여부
    bm25_k: int = 5  # BM25 검색 결과 수
    
    # 결과 설정
    include_metadata: bool = True
    include_score: bool = True
    
    def __post_init__(self):
        """설정 후처리 및 검증"""
        self._validate_config()
    
    def _validate_config(self):
        """설정 검증"""
        # search_type 검증
        if not isinstance(self.search_type, SearchType):
            raise ValueError(f"search_type must be a SearchType enum, got {type(self.search_type).__name__}")
        
        # k 값 검증
        if self.k <= 0:
            raise ConfigurationError(f"k must be positive, got {self.k}")
        
        # 점수 임계값 검증
        if self.search_type == SearchType.SIMILARITY_SCORE_THRESHOLD:
            if self.score_threshold is None:
                raise ConfigurationError("score_threshold is required for SIMILARITY_SCORE_THRESHOLD search")
            if not 0.0 <= self.score_threshold <= 1.0:
                raise ConfigurationError(f"score_threshold must be between 0.0 and 1.0, got {self.score_threshold}")
        
        # MMR lambda_mult 검증
        if self.search_type == SearchType.MMR:
            if not 0.0 <= self.lambda_mult <= 1.0:
                raise ConfigurationError(f"lambda_mult must be between 0.0 and 1.0, got {self.lambda_mult}")
        
        # 하이브리드 검색 가중치 검증
        if self.search_type == SearchType.HYBRID:
            if len(self.hybrid_weights) != 2:
                raise ConfigurationError("hybrid_weights must contain exactly 2 values [semantic_weight, bm25_weight]")
            if not all(0.0 <= w <= 1.0 for w in self.hybrid_weights):
                raise ConfigurationError("All hybrid_weights must be between 0.0 and 1.0")
            if abs(sum(self.hybrid_weights) - 1.0) > 1e-6:
                raise ConfigurationError("hybrid_weights must sum to 1.0")
        
        # BM25 k 검증
        if self.search_type in [SearchType.BM25, SearchType.HYBRID]:
            if self.bm25_k <= 0:
                raise ConfigurationError(f"bm25_k must be positive, got {self.bm25_k}")


class SimilaritySearchService:
    """LangChain 기반 유사도 검색 서비스"""
    
    def __init__(self, embedding_service: EmbeddingService, config: SearchConfig):
        """
        유사도 검색 서비스를 초기화합니다.
        
        Args:
            embedding_service: 임베딩 서비스 
            config: 검색 설정
        """
        if embedding_service is None:
            raise VectorSearchError("EmbeddingService is required")
        
        self.embedding_service = embedding_service
        self.config = config
        
        # 한국어 토크나이저 설정 (필요 시)
        self.korean_tokenizer = None
        if config.use_korean_tokenizer and KOREAN_TOKENIZER_AVAILABLE:
            self.korean_tokenizer = self._setup_korean_tokenizer()
        
        # 다양한 retriever 인스턴스들
        self.retriever = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
        # 메인 retriever 생성
        self.retriever = self._create_retriever()
        
        logger.info(f"SimilaritySearchService initialized with {config.search_type.value} search")
    
    def _setup_korean_tokenizer(self):
        """한국어 토크나이저를 설정합니다."""
        try:
            if not KOREAN_TOKENIZER_AVAILABLE:
                logger.warning("Korean tokenizer requested but kiwipiepy not available")
                return None
            
            kiwi = Kiwi()
            
            # 일반적인 AI/ML 용어들 추가
            custom_words = [
                ('리비안', 'NNP'),
                ('테슬라', 'NNP'),  
                ('전기차', 'NNG'),
                ('랭체인', 'NNP'),
                ('LangChain', 'SL'),
                ('ChromaDB', 'SL'),
                ('OpenAI', 'SL'),
                ('HuggingFace', 'SL'),
                ('임베딩', 'NNG'),
                ('벡터', 'NNG'),
                ('검색', 'NNG'),
            ]
            
            for word, pos in custom_words:
                kiwi.add_user_word(word, pos)
            
            logger.info("Korean tokenizer setup completed with custom vocabulary")
            return kiwi
            
        except Exception as e:
            logger.error(f"Failed to setup Korean tokenizer: {e}")
            return None
    
    def _korean_tokenizer_func(self, text: str) -> List[str]:
        """한국어 텍스트를 토큰화합니다."""
        if self.korean_tokenizer is None:
            # Fallback to simple split
            return text.split()
        
        try:
            tokens = [token.form for token in self.korean_tokenizer.tokenize(text)]
            return tokens
        except Exception as e:
            logger.error(f"Korean tokenization failed: {e}")
            return text.split()
    
    def _create_bm25_retriever(self, documents: List[Document]) -> Optional[BaseRetriever]:
        """BM25 검색기를 생성합니다."""
        if not BM25_AVAILABLE:
            logger.error("BM25Retriever not available. Please install rank_bm25")
            return None
        
        try:
            # 전처리 함수 설정
            preprocess_func = None
            if self.config.use_korean_tokenizer and self.korean_tokenizer is not None:
                preprocess_func = self._korean_tokenizer_func
            
            # BM25 retriever 생성
            bm25_retriever = BM25Retriever.from_documents(
                documents=documents,
                preprocess_func=preprocess_func,
                k=self.config.bm25_k
            )
            
            logger.info(f"BM25 retriever created with k={self.config.bm25_k}")
            return bm25_retriever
            
        except Exception as e:
            logger.error(f"Failed to create BM25 retriever: {e}")
            return None
    
    def _create_ensemble_retriever(self, semantic_retriever: BaseRetriever, 
                                 bm25_retriever: BaseRetriever) -> Optional[BaseRetriever]:
        """앙상블 검색기를 생성합니다."""
        try:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=self.config.hybrid_weights
            )
            
            logger.info(f"Ensemble retriever created with weights {self.config.hybrid_weights}")
            return ensemble_retriever
            
        except Exception as e:
            logger.error(f"Failed to create ensemble retriever: {e}")
            return None
    
    def _create_retriever(self) -> BaseRetriever:
        """LangChain Retriever를 생성합니다."""
        try:
            if self.config.search_type in [SearchType.SIMILARITY, SearchType.MMR, SearchType.SIMILARITY_SCORE_THRESHOLD]:
                # 기존 벡터 스토어 기반 검색
                return self._create_vector_retriever()
                
            elif self.config.search_type == SearchType.BM25:
                # BM25 검색
                return self._create_bm25_only_retriever()
                
            elif self.config.search_type == SearchType.HYBRID:
                # 하이브리드 검색 (의미론적 + BM25)
                return self._create_hybrid_retriever()
                
            else:
                raise VectorSearchError(f"Unsupported search type: {self.config.search_type}")
                
        except Exception as e:
            logger.error(f"Failed to create retriever: {e}")
            raise VectorSearchError(f"Failed to create retriever: {e}")
    
    def _create_vector_retriever(self) -> BaseRetriever:
        """벡터 기반 검색기를 생성합니다."""
        search_kwargs = self._build_search_kwargs()
        
        retriever = self.embedding_service.vector_db.as_retriever(
            search_type=self.config.search_type.value,
            search_kwargs=search_kwargs
        )
        
        logger.debug(f"Vector retriever created with search_type: {self.config.search_type.value}")
        return retriever
    
    def _create_bm25_only_retriever(self) -> BaseRetriever:
        """BM25 전용 검색기를 생성합니다."""
        if not BM25_AVAILABLE:
            raise VectorSearchError("BM25 search not available. Please install rank_bm25 and langchain-community")
        
        # 벡터 데이터베이스에서 모든 문서 가져오기
        documents = self._get_all_documents_from_vector_db()
        
        if not documents:
            raise VectorSearchError("No documents found in vector database for BM25 search")
        
        bm25_retriever = self._create_bm25_retriever(documents)
        if bm25_retriever is None:
            raise VectorSearchError("Failed to create BM25 retriever")
        
        self.bm25_retriever = bm25_retriever
        return bm25_retriever
    
    def _create_hybrid_retriever(self) -> BaseRetriever:
        """하이브리드 검색기를 생성합니다."""
        if not BM25_AVAILABLE:
            raise VectorSearchError("Hybrid search not available. Please install rank_bm25 and langchain-community")
        
        # 벡터 기반 검색기 생성
        semantic_retriever = self._create_vector_retriever()
        
        # BM25 검색기 생성
        documents = self._get_all_documents_from_vector_db()
        if not documents:
            raise VectorSearchError("No documents found in vector database for hybrid search")
        
        bm25_retriever = self._create_bm25_retriever(documents)
        if bm25_retriever is None:
            raise VectorSearchError("Failed to create BM25 retriever for hybrid search")
        
        # 앙상블 검색기 생성
        ensemble_retriever = self._create_ensemble_retriever(semantic_retriever, bm25_retriever)
        if ensemble_retriever is None:
            raise VectorSearchError("Failed to create ensemble retriever")
        
        # 인스턴스 저장
        self.bm25_retriever = bm25_retriever
        self.ensemble_retriever = ensemble_retriever
        
        return ensemble_retriever
    
    def _get_all_documents_from_vector_db(self) -> List[Document]:
        """벡터 데이터베이스에서 모든 문서를 가져옵니다."""
        try:
            # ChromaDB에서 모든 문서 가져오기
            collection_info = self.embedding_service.get_collection_info()
            total_docs = collection_info.get('document_count', 0)
            
            if total_docs == 0:
                return []
            
            # 큰 k 값으로 모든 문서 검색 (더미 쿼리 사용)
            # 이는 비효율적이므로 실제로는 vector_db에서 직접 문서를 가져오는 메서드를 추가하는 것이 좋습니다
            all_docs = self.embedding_service.vector_db.similarity_search(
                query="",  # 빈 쿼리
                k=min(total_docs, 1000)  # 최대 1000개로 제한
            )
            
            logger.info(f"Retrieved {len(all_docs)} documents from vector database")
            return all_docs
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents from vector database: {e}")
            return []
    
    def _build_search_kwargs(self) -> Dict[str, Any]:
        """검색 파라미터를 구성합니다."""
        search_kwargs = {
            "k": self.config.k
        }
        
        # 점수 임계값
        if self.config.score_threshold is not None:
            search_kwargs["score_threshold"] = self.config.score_threshold
        
        # MMR 파라미터들
        if self.config.search_type == SearchType.MMR:
            search_kwargs["fetch_k"] = self.config.fetch_k
            search_kwargs["lambda_mult"] = self.config.lambda_mult
        
        # 메타데이터 필터
        if self.config.metadata_filter:
            search_kwargs["filter"] = self.config.metadata_filter
        
        logger.debug(f"Search kwargs: {search_kwargs}")
        return search_kwargs
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        쿼리를 사용하여 유사도 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            
        Returns:
            검색 결과 리스트 (document, score, metadata 포함)
            
        Raises:
            VectorSearchError: 검색 실패 시
        """
        if not query or not query.strip():
            raise VectorSearchError("Query cannot be empty")
        
        try:
            logger.info(f"Searching for: '{query[:50]}...'")
            
            # Retriever를 통한 검색
            if self.config.search_type == SearchType.SIMILARITY_SCORE_THRESHOLD:
                # 점수와 함께 검색
                docs_with_scores = self._search_with_score(query)
                results = self._format_results_with_scores(docs_with_scores)
            else:
                # 일반 검색 (similarity, mmr)
                documents = self.retriever.invoke(query)
                results = self._format_results(documents)
            
            # 추가 필터링 적용
            filtered_results = self._apply_additional_filters(results)
            
            logger.info(f"Search completed: {len(filtered_results)} results")
            return filtered_results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorSearchError(f"Search failed: {e}")
    
    def _search_with_score(self, query: str) -> List[tuple]:
        """점수와 함께 검색을 수행합니다."""
        try:
            # VectorStore의 similarity_search_with_score 직접 호출
            return self.embedding_service.vector_db.similarity_search_with_score(
                query, 
                k=self.config.k
            )
        except Exception as e:
            logger.error(f"Score-based search failed: {e}")
            raise VectorSearchError(f"Score-based search failed: {e}")
    
    def _format_results(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """문서를 결과 형식으로 변환합니다."""
        results = []
        
        for doc in documents:
            result = {
                'document': doc,
                'score': None,  # 기본 검색은 점수 없음
            }
            
            if self.config.include_metadata:
                result['metadata'] = doc.metadata
            
            results.append(result)
        
        return results
    
    def _format_results_with_scores(self, docs_with_scores: List[tuple]) -> List[Dict[str, Any]]:
        """점수와 함께 문서를 결과 형식으로 변환합니다."""
        results = []
        
        for doc, score in docs_with_scores:
            # 점수 임계값 필터링
            if self.config.score_threshold and score < self.config.score_threshold:
                continue
            
            result = {
                'document': doc,
                'score': score if self.config.include_score else None,
            }
            
            if self.config.include_metadata:
                result['metadata'] = doc.metadata
                
            results.append(result)
        
        return results
    
    def _apply_additional_filters(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """추가 필터링을 적용합니다."""
        filtered_results = results
        
        # 시간 범위 필터링
        if self.config.time_range:
            filtered_results = self._filter_by_time_range(filtered_results)
        
        return filtered_results
    
    def _filter_by_time_range(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """시간 범위로 결과를 필터링합니다."""
        if not self.config.time_range:
            return results
        
        start_time = self.config.time_range.get('start', 0.0)
        end_time = self.config.time_range.get('end', float('inf'))
        
        filtered = []
        for result in results:
            doc_start = result['document'].metadata.get('start_time', 0.0)
            doc_end = result['document'].metadata.get('end_time', 0.0)
            
            # 시간 범위와 겹치는지 확인
            if doc_start <= end_time and doc_end >= start_time:
                filtered.append(result)
        
        logger.debug(f"Time range filter: {len(results)} -> {len(filtered)} results")
        return filtered
    
    def search_by_video(self, query: str, video_id: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        특정 영상에서만 검색합니다.
        
        Args:
            query: 검색 쿼리
            video_id: 대상 영상 ID
            k: 반환할 결과 수 (None이면 config의 k 사용)
            
        Returns:
            검색 결과 리스트
        """
        # 임시로 메타데이터 필터 변경
        original_filter = self.config.metadata_filter.copy()
        original_k = self.config.k
        
        try:
            # 비디오 필터 추가
            self.config.metadata_filter['video_id'] = video_id
            if k is not None:
                self.config.k = k
            
            # 새로운 retriever 생성
            self.retriever = self._create_retriever()
            
            # 검색 수행
            return self.search(query)
        
        finally:
            # 원래 설정으로 복원
            self.config.metadata_filter = original_filter
            self.config.k = original_k
            self.retriever = self._create_retriever()
    
    def search_by_channel(self, query: str, channel_name: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        특정 채널에서만 검색합니다.
        
        Args:
            query: 검색 쿼리
            channel_name: 대상 채널명  
            k: 반환할 결과 수 (None이면 config의 k 사용)
            
        Returns:
            검색 결과 리스트
        """
        # 임시로 메타데이터 필터 변경
        original_filter = self.config.metadata_filter.copy()
        original_k = self.config.k
        
        try:
            # 채널 필터 추가
            self.config.metadata_filter['channel_name'] = channel_name
            if k is not None:
                self.config.k = k
            
            # 새로운 retriever 생성
            self.retriever = self._create_retriever()
            
            # 검색 수행
            return self.search(query)
        
        finally:
            # 원래 설정으로 복원
            self.config.metadata_filter = original_filter
            self.config.k = original_k
            self.retriever = self._create_retriever()
    
    def search_by_time_range(self, query: str, start_time: float, end_time: float, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        특정 시간 범위에서만 검색합니다.
        
        Args:
            query: 검색 쿼리
            start_time: 시작 시간 (초)
            end_time: 종료 시간 (초)
            k: 반환할 결과 수 (None이면 config의 k 사용)
            
        Returns:
            검색 결과 리스트
        """
        # 임시로 시간 범위 설정 변경
        original_time_range = self.config.time_range
        original_k = self.config.k
        
        try:
            # 시간 범위 설정
            self.config.time_range = {'start': start_time, 'end': end_time}
            if k is not None:
                self.config.k = k
            
            # 검색 수행
            return self.search(query)
        
        finally:
            # 원래 설정으로 복원
            self.config.time_range = original_time_range
            self.config.k = original_k
    
    def get_search_stats(self) -> Dict[str, Any]:
        """검색 서비스 통계 정보를 반환합니다."""
        try:
            vector_info = self.embedding_service.get_collection_info()
            
            return {
                'search_type': self.config.search_type.value,
                'k': self.config.k,
                'score_threshold': self.config.score_threshold,
                'lambda_mult': self.config.lambda_mult if self.config.search_type == SearchType.MMR else None,
                'metadata_filter': self.config.metadata_filter,
                'vector_db_info': vector_info,
                'retriever_type': type(self.retriever).__name__
            }
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {'error': str(e)}


def create_similarity_search_service(
    embedding_service: Optional[EmbeddingService] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    search_type: SearchType = SearchType.SIMILARITY,
    k: int = 5,
    score_threshold: Optional[float] = None,
    lambda_mult: float = 0.5,
    metadata_filter: Optional[Dict[str, Any]] = None,
    **kwargs
) -> SimilaritySearchService:
    """
    유사도 검색 서비스를 생성하는 팩토리 함수입니다.
    
    Args:
        embedding_service: 기존 임베딩 서비스 (제공되지 않으면 새로 생성)
        provider: 임베딩 제공자 ("openai" | "huggingface")
        model_name: 사용할 모델명
        search_type: 검색 타입
        k: 반환할 결과 수
        score_threshold: 점수 임계값 (점수 임계값 검색용)
        lambda_mult: MMR lambda_mult 파라미터
        metadata_filter: 메타데이터 필터
        **kwargs: 추가 설정
        
    Returns:
        SimilaritySearchService 인스턴스
        
    Raises:
        VectorSearchError: 생성 실패 시
    """
    try:
        # 임베딩 서비스 생성 (제공되지 않은 경우)
        if embedding_service is None:
            if provider is None:
                provider = "huggingface"  # 기본값
            
            embedding_service = create_embedding_service(
                provider=provider,
                model_name=model_name,
                **kwargs
            )
            
            logger.info(f"Created embedding service with provider: {provider}")
        
        # 검색 설정 생성
        config = SearchConfig(
            search_type=search_type,
            k=k,
            score_threshold=score_threshold,
            lambda_mult=lambda_mult,
            metadata_filter=metadata_filter or {}
        )
        
        return SimilaritySearchService(embedding_service, config)
    
    except Exception as e:
        logger.error(f"Failed to create similarity search service: {e}")
        raise VectorSearchError(f"Failed to create similarity search service: {e}") 