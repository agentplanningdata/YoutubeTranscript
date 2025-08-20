"""
임베딩 및 벡터 저장 서비스 - LangChain 기반

LangChain의 Embeddings와 VectorStore를 활용하여
자막 청크의 임베딩 생성 및 벡터 저장소 관리 기능을 제공합니다.
"""

from typing import List, Dict, Any, Optional, Union
import hashlib
import json
from dataclasses import dataclass, field
import os

# LangChain Core Imports
from langchain_core.documents import Document

# LangChain Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# VectorDatabase from our existing implementation
from src.database.vector_db import VectorDatabase, ChromaDBConfig, create_huggingface_vector_db, create_openai_vector_db

from src.utils.exceptions import EmbeddingError, VectorStoreError
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)


@dataclass
class EmbeddingConfig:
    """임베딩 서비스 설정"""
    
    # 기본 설정
    provider: str = "openai"  # "openai" | "huggingface"
    model_name: str = "text-embedding-ada-002"  # OpenAI 기본 모델
    api_key: Optional[str] = None
    
    # 임베딩 처리 설정
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 60.0
    normalize_embeddings: bool = True
    
    # 벡터 저장소 설정
    collection_name: str = "transcript_chunks"
    persist_directory: str = "./chromadb"
    
    # HuggingFace 설정
    device: str = "cpu"  # "cpu" | "cuda"
    
    def __post_init__(self):
        """설정 후처리"""
        # API 키 환경변수에서 가져오기
        if not self.api_key and self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        # HuggingFace 기본 모델 설정
        if self.provider == "huggingface" and self.model_name == "text-embedding-ada-002":
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """LangChain 기반 임베딩 서비스"""
    
    def __init__(self, config: EmbeddingConfig):
        """
        임베딩 서비스를 초기화합니다.
        
        Args:
            config: 임베딩 서비스 설정
        """
        self.config = config
        self.embeddings = self._create_embedding_model()
        self.vector_db = self._create_vector_database()
        
        logger.info(f"EmbeddingService initialized with provider: {config.provider}")
    
    def _create_embedding_model(self):
        """임베딩 모델을 생성합니다."""
        try:
            if self.config.provider == "openai":
                return OpenAIEmbeddings(
                    model=self.config.model_name,
                    openai_api_key=self.config.api_key
                )
            elif self.config.provider == "huggingface":
                return HuggingFaceEmbeddings(
                    model_name=self.config.model_name,
                    model_kwargs={'device': self.config.device},
                    encode_kwargs={'normalize_embeddings': self.config.normalize_embeddings}
                )
            else:
                raise EmbeddingError(f"Unsupported embedding provider: {self.config.provider}")
        
        except Exception as e:
            logger.error(f"Failed to create embedding model: {e}")
            raise EmbeddingError(f"Failed to initialize embedding model: {e}")
    
    def _create_vector_database(self):
        """벡터 데이터베이스를 생성합니다."""
        try:
            # ChromaDB 설정
            chroma_config = ChromaDBConfig(
                collection_name=self.config.collection_name,
                persist_directory=self.config.persist_directory,
                embedding_provider=self.config.provider,
                openai_api_key=self.config.api_key if self.config.provider == "openai" else None,
                huggingface_model=self.config.model_name if self.config.provider == "huggingface" else None
            )
            
            return VectorDatabase(chroma_config)
        
        except Exception as e:
            logger.error(f"Failed to create vector database: {e}")
            raise VectorStoreError(f"Failed to initialize vector database: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트 리스트에 대한 임베딩을 생성합니다.
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
            
        Raises:
            EmbeddingError: 임베딩 생성 실패 시
        """
        if not texts:
            raise EmbeddingError("Cannot generate embeddings for empty text list")
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using {self.config.provider}")
            
            # LangChain embed_documents 사용
            embeddings = self.embeddings.embed_documents(texts)
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        단일 쿼리에 대한 임베딩을 생성합니다.
        
        Args:
            query: 임베딩할 쿼리 텍스트
            
        Returns:
            임베딩 벡터
            
        Raises:
            EmbeddingError: 임베딩 생성 실패 시
        """
        if not query.strip():
            raise EmbeddingError("Cannot generate embedding for empty query")
        
        try:
            logger.debug(f"Generating query embedding for: {query[:50]}...")
            
            # LangChain embed_query 사용
            embedding = self.embeddings.embed_query(query)
            
            return embedding
        
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise EmbeddingError(f"Failed to generate query embedding: {e}")
    
    def store_transcript_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        global_metadata: Optional[Dict[str, Any]] = None,
        remove_duplicates: bool = False
    ) -> Dict[str, Any]:
        """
        자막 청크들을 벡터 저장소에 저장합니다.
        
        Args:
            chunks: 저장할 자막 청크들
            global_metadata: 모든 청크에 적용할 공통 메타데이터
            remove_duplicates: 중복 제거 여부
            
        Returns:
            저장 결과 정보
            
        Raises:
            VectorStoreError: 저장 실패 시
        """
        if not chunks:
            return {'stored_count': 0, 'document_ids': [], 'duplicates_removed': 0}
        
        try:
            # 중복 제거
            processed_chunks = chunks
            duplicates_removed = 0
            
            if remove_duplicates:
                processed_chunks, duplicates_removed = self._remove_duplicates(chunks)
            
            # 청크를 LangChain Document로 변환
            documents = self._chunks_to_documents(processed_chunks, global_metadata)
            
            # 벡터 저장소에 저장
            logger.info(f"Storing {len(documents)} documents to vector store")
            document_ids = self.vector_db.add_documents(documents)
            
            result = {
                'stored_count': len(documents),
                'document_ids': document_ids if document_ids else [],
                'duplicates_removed': duplicates_removed,
                'embedding_provider': self.config.provider,
                'model_name': self.config.model_name
            }
            
            logger.info(f"Successfully stored {len(documents)} transcript chunks")
            return result
        
        except Exception as e:
            logger.error(f"Failed to store transcript chunks: {e}")
            raise VectorStoreError(f"Failed to store documents in vector store: {e}")
    
    def _chunks_to_documents(
        self, 
        chunks: List[Dict[str, Any]], 
        global_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        자막 청크를 LangChain Document 객체로 변환합니다.
        
        Args:
            chunks: 자막 청크들
            global_metadata: 공통 메타데이터
            
        Returns:
            Document 객체 리스트
        """
        documents = []
        
        for chunk in chunks:
            text = chunk.get('text', '').strip()
            if not text:
                continue
            
            # 기본 메타데이터
            metadata = {
                'chunk_id': chunk.get('chunk_id', ''),
                'start_time': chunk.get('start_time', 0.0),
                'end_time': chunk.get('end_time', 0.0),
                'word_count': chunk.get('word_count', 0),
                'char_count': chunk.get('char_count', len(text)),
                'chunking_method': chunk.get('chunking_method', 'unknown')
            }
            
            # source_segments 처리 (복잡한 객체는 요약)
            if 'source_segments' in chunk:
                source_segments = chunk['source_segments']
                metadata['source_segments_count'] = len(source_segments) if source_segments else 0
            
            # 공통 메타데이터 추가
            if global_metadata:
                metadata.update(global_metadata)
            
            # Document 생성
            document = Document(page_content=text, metadata=metadata)
            documents.append(document)
        
        return documents
    
    def _remove_duplicates(self, chunks: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
        """
        중복 청크를 제거합니다 (첫 번째 발생 보존).
        
        Args:
            chunks: 청크 리스트
            
        Returns:
            (중복 제거된 청크 리스트, 제거된 중복 개수)
        """
        seen_hashes = set()
        unique_chunks = []
        duplicates_removed = 0
        
        for chunk in chunks:
            content_hash = self._generate_content_hash(chunk)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)
            else:
                duplicates_removed += 1
                logger.debug(f"Removed duplicate chunk: {chunk.get('chunk_id', 'unknown')}")
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate chunks out of {len(chunks)}")
        
        return unique_chunks, duplicates_removed
    
    def _generate_content_hash(self, chunk: Dict[str, Any]) -> str:
        """
        청크 내용의 해시를 생성합니다.
        
        Args:
            chunk: 청크 데이터
            
        Returns:
            내용 해시
        """
        text = chunk.get('text', '').strip().lower()
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _generate_content_hashes(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        여러 청크의 내용 해시를 생성합니다.
        
        Args:
            chunks: 청크 리스트
            
        Returns:
            해시 리스트
        """
        return [self._generate_content_hash(chunk) for chunk in chunks]
    
    def search_similar_chunks(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        쿼리와 유사한 청크를 검색합니다.
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            score_threshold: 최소 유사도 점수
            
        Returns:
            유사한 청크 리스트
            
        Raises:
            VectorStoreError: 검색 실패 시
        """
        try:
            logger.info(f"Searching for similar chunks: {query[:50]}...")
            
            if score_threshold is not None:
                # 점수와 함께 검색
                results = self.vector_db.similarity_search_with_score(query, k=k)
                
                # 점수 필터링
                filtered_results = [
                    {'document': doc, 'score': score}
                    for doc, score in results
                    if score >= score_threshold
                ]
                
                logger.info(f"Found {len(filtered_results)} chunks above threshold {score_threshold}")
                return filtered_results
            else:
                # 일반 검색
                documents = self.vector_db.similarity_search(query, k=k)
                results = [{'document': doc, 'score': None} for doc in documents]
                
                logger.info(f"Found {len(results)} similar chunks")
                return results
        
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            raise VectorStoreError(f"Failed to search vector store: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보를 반환합니다.
        
        Returns:
            컬렉션 정보
        """
        try:
            return {
                'collection_name': self.config.collection_name,
                'embedding_provider': self.config.provider,
                'embedding_model': self.config.model_name,
                'persist_directory': self.config.persist_directory,
                'vector_db_info': self.vector_db.get_collection_info()
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {'error': str(e)}


def create_embedding_service(
    provider: str = "openai",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    collection_name: str = "transcript_chunks",
    persist_directory: str = "./chromadb",
    **kwargs
) -> EmbeddingService:
    """
    임베딩 서비스를 생성하는 팩토리 함수입니다.
    
    Args:
        provider: 임베딩 제공자 ("openai" | "huggingface")
        model_name: 사용할 모델명
        api_key: API 키 (OpenAI용)
        collection_name: 컬렉션명
        persist_directory: 저장 디렉토리
        **kwargs: 추가 설정
        
    Returns:
        EmbeddingService 인스턴스
        
    Raises:
        EmbeddingError: 생성 실패 시
    """
    try:
        config = EmbeddingConfig(
            provider=provider,
            model_name=model_name or ("text-embedding-ada-002" if provider == "openai" else "sentence-transformers/all-MiniLM-L6-v2"),
            api_key=api_key,
            collection_name=collection_name,
            persist_directory=persist_directory,
            **kwargs
        )
        
        return EmbeddingService(config)
    
    except Exception as e:
        logger.error(f"Failed to create embedding service: {e}")
        raise EmbeddingError(f"Failed to create embedding service: {e}") 