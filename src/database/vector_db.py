"""
벡터 데이터베이스 (ChromaDB) - LangChain 기반

LangChain Chroma와 OpenAI Embeddings를 사용한 벡터 데이터베이스 관리
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
import os
from pathlib import Path

# LangChain Core Imports
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

# LangChain Vector Store & Embedding Imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.exceptions import ChromaDBError, ConfigurationError
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)


@dataclass
class ChromaDBConfig:
    """LangChain Chroma 설정"""
    
    # 기본 설정
    persist_directory: str = "./chromadb_langchain"
    collection_name: str = "youtube_transcripts"
    
    # 임베딩 설정
    embedding_provider: str = "openai"  # "openai", "huggingface", "sentence-transformers"
    openai_model: str = "text-embedding-3-large"
    openai_api_key: Optional[str] = None
    huggingface_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # ChromaDB 설정
    collection_metadata: Dict[str, Any] = field(
        default_factory=lambda: {"description": "YouTube transcript embeddings with LangChain"}
    )
    
    # Chroma Cloud 설정 (선택사항)
    chroma_cloud_api_key: Optional[str] = None
    chroma_tenant: Optional[str] = None
    chroma_database: Optional[str] = None
    host: Optional[str] = None  # 로컬 ChromaDB 서버용
    port: Optional[int] = None
    
    def __post_init__(self):
        """설정 검증 및 정규화"""
        if not self.collection_name or not self.collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        
        # 컬렉션 이름 정규화
        self.collection_name = self.collection_name.strip()
        
        # 디렉토리 경로 정규화 (persist_directory가 제공된 경우)
        if self.persist_directory:
            self.persist_directory = os.path.abspath(self.persist_directory)
        
        # OpenAI API 키 확인
        if self.embedding_provider.lower() == "openai":
            if not self.openai_api_key:
                self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key and self.embedding_provider.lower() == "openai":
                logger.warning("OpenAI API key not found. Consider setting OPENAI_API_KEY environment variable.")


class VectorDatabase:
    """LangChain Chroma 기반 벡터 데이터베이스 관리 클래스"""
    
    def __init__(self, config: ChromaDBConfig):
        """
        벡터 데이터베이스를 초기화합니다.
        
        Args:
            config: ChromaDB 설정
            
        Raises:
            ChromaDBError: 초기화 실패 시
        """
        self.config = config
        self.vectorstore: Optional[Chroma] = None
        self.embeddings = None
        
        try:
            # 임베딩 함수 초기화
            self.embeddings = self._create_embedding_function()
            
            # Chroma 벡터스토어 초기화
            self.vectorstore = self._initialize_chroma()
            
            logger.info(f"✅ LangChain Chroma 초기화 완료: {config.collection_name}")
            logger.info(f"   📁 저장 경로: {config.persist_directory}")
            logger.info(f"   🤖 임베딩: {config.embedding_provider} ({self._get_embedding_model_info()})")
            
        except Exception as e:
            logger.error(f"❌ VectorDatabase 초기화 실패: {e}")
            raise ChromaDBError(f"Failed to initialize VectorDatabase: {e}")
    
    def _create_embedding_function(self):
        """임베딩 함수를 생성합니다."""
        provider = self.config.embedding_provider.lower()
        
        try:
            if provider == "openai":
                if not self.config.openai_api_key:
                    raise ConfigurationError("OpenAI API key is required for OpenAI embeddings")
                
                return OpenAIEmbeddings(
                    model=self.config.openai_model,
                    openai_api_key=self.config.openai_api_key
                )
                
            elif provider in ["huggingface", "sentence-transformers"]:
                return HuggingFaceEmbeddings(
                    model_name=self.config.huggingface_model,
                    model_kwargs={'device': 'cpu'},  # CPU 사용 (GPU 사용시 'cuda')
                    encode_kwargs={'normalize_embeddings': True}
                )
                
            else:
                # 기본값: HuggingFace
                logger.warning(f"Unknown embedding provider '{provider}', using HuggingFace as fallback")
                return HuggingFaceEmbeddings(
                    model_name=self.config.huggingface_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
        except ConfigurationError:
            # ConfigurationError는 re-raise (API 키 없음 등)
            raise
        except Exception as e:
            logger.error(f"임베딩 함수 생성 실패: {e}")
            # 다른 예외인 경우만 Fallback to HuggingFace
            if provider == "openai":
                # OpenAI 설정이었는데 실패한 경우는 fallback 안 함
                raise
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def _initialize_chroma(self) -> Chroma:
        """LangChain Chroma 벡터스토어를 초기화합니다."""
        chroma_kwargs = {
            "collection_name": self.config.collection_name,
            "embedding_function": self.embeddings,
        }
        
        # Chroma Cloud 사용
        if self.config.chroma_cloud_api_key:
            chroma_kwargs.update({
                "chroma_cloud_api_key": self.config.chroma_cloud_api_key,
                "tenant": self.config.chroma_tenant,
                "database": self.config.chroma_database,
            })
        # 로컬 ChromaDB 서버 연결
        elif self.config.host:
            chroma_kwargs.update({
                "host": self.config.host,
                "port": self.config.port or 8000,
            })
        # 로컬 파일 시스템 (기본값)
        else:
            if self.config.persist_directory:
                # 디렉토리 생성
                Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)
                chroma_kwargs["persist_directory"] = self.config.persist_directory
        
        return Chroma(**chroma_kwargs)
    
    def _get_embedding_model_info(self) -> str:
        """현재 사용중인 임베딩 모델 정보를 반환합니다."""
        try:
            if isinstance(self.embeddings, OpenAIEmbeddings):
                return self.config.openai_model
            elif hasattr(self.embeddings, '__class__') and 'HuggingFace' in str(self.embeddings.__class__):
                return self.config.huggingface_model
            elif self.config.embedding_provider.lower() == "openai":
                return self.config.openai_model
            elif self.config.embedding_provider.lower() in ["huggingface", "sentence-transformers"]:
                return self.config.huggingface_model
            else:
                return "Unknown"
        except Exception:
            # isinstance 체크 실패 시 설정값 기반으로 반환
            if self.config.embedding_provider.lower() == "openai":
                return self.config.openai_model
            elif self.config.embedding_provider.lower() in ["huggingface", "sentence-transformers"]:
                return self.config.huggingface_model
            else:
                return "Unknown"
    
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """
        문서들을 벡터스토어에 추가합니다.
        
        Args:
            documents: 추가할 문서 목록
            ids: 문서 ID 목록 (선택사항)
            
        Returns:
            추가된 문서의 ID 목록
        """
        try:
            if not self.vectorstore:
                raise ChromaDBError("VectorStore not initialized")
            
            # LangChain Chroma는 add_documents 메서드 사용
            added_ids = self.vectorstore.add_documents(documents=documents, ids=ids)
            
            logger.info(f"📄 {len(documents)}개 문서 추가 완료 (컬렉션: {self.config.collection_name})")
            return added_ids
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            raise ChromaDBError(f"Failed to add documents: {e}")
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, 
                  ids: Optional[List[str]] = None) -> List[str]:
        """
        텍스트들을 벡터스토어에 추가합니다.
        
        Args:
            texts: 추가할 텍스트 목록
            metadatas: 메타데이터 목록
            ids: 텍스트 ID 목록
            
        Returns:
            추가된 텍스트의 ID 목록
        """
        try:
            if not self.vectorstore:
                raise ChromaDBError("VectorStore not initialized")
            
            added_ids = self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"📝 {len(texts)}개 텍스트 추가 완료")
            return added_ids
            
        except Exception as e:
            logger.error(f"텍스트 추가 실패: {e}")
            raise ChromaDBError(f"Failed to add texts: {e}")
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None,
                          **kwargs) -> List[Document]:
        """
        유사도 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            filter: 검색 필터
            **kwargs: 추가 검색 파라미터
            
        Returns:
            검색된 문서 목록
        """
        try:
            if not self.vectorstore:
                raise ChromaDBError("VectorStore not initialized")
            
            # LangChain Chroma의 similarity_search 사용
            search_kwargs = {"k": k}
            if filter:
                search_kwargs["filter"] = filter
            search_kwargs.update(kwargs)
            
            results = self.vectorstore.similarity_search(query, **search_kwargs)
            
            logger.info(f"🔍 검색 완료: '{query}' -> {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"유사도 검색 실패: {e}")
            raise ChromaDBError(f"Failed to perform similarity search: {e}")
    
    def similarity_search_with_score(self, query: str, k: int = 4, 
                                     filter: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """
        점수와 함께 유사도 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            filter: 검색 필터
            
        Returns:
            (문서, 점수) 튜플의 목록
        """
        try:
            if not self.vectorstore:
                raise ChromaDBError("VectorStore not initialized")
            
            search_kwargs = {"k": k}
            if filter:
                search_kwargs["filter"] = filter
            
            results = self.vectorstore.similarity_search_with_score(query, **search_kwargs)
            
            logger.info(f"🎯 점수 포함 검색 완료: '{query}' -> {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"점수 포함 유사도 검색 실패: {e}")
            raise ChromaDBError(f"Failed to perform similarity search with score: {e}")
    
    def as_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict] = None):
        """
        벡터스토어를 Retriever로 변환합니다.
        
        Args:
            search_type: 검색 타입 ("similarity", "mmr" 등)
            search_kwargs: 검색 파라미터
            
        Returns:
            LangChain Retriever 객체
        """
        try:
            if not self.vectorstore:
                raise ChromaDBError("VectorStore not initialized")
            
            return self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs or {}
            )
            
        except Exception as e:
            logger.error(f"Retriever 생성 실패: {e}")
            raise ChromaDBError(f"Failed to create retriever: {e}")
    
    def delete(self, ids: List[str]) -> bool:
        """
        문서를 삭제합니다.
        
        Args:
            ids: 삭제할 문서 ID 목록
            
        Returns:
            삭제 성공 여부
        """
        try:
            if not self.vectorstore:
                raise ChromaDBError("VectorStore not initialized")
            
            # LangChain Chroma의 delete 메서드 사용
            result = self.vectorstore.delete(ids=ids)
            
            logger.info(f"🗑️ {len(ids)}개 문서 삭제 완료")
            return True
            
        except Exception as e:
            logger.error(f"문서 삭제 실패: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보를 반환합니다.
        
        Returns:
            컬렉션 정보 딕셔너리
        """
        try:
            if not self.vectorstore:
                return {"error": "VectorStore not initialized"}
            
            # ChromaDB의 underlying collection에 접근
            if hasattr(self.vectorstore, '_collection'):
                collection = self.vectorstore._collection
                count = collection.count()
                metadata = getattr(collection, 'metadata', {})
                
                return {
                    "name": self.config.collection_name,
                    "count": count,
                    "metadata": metadata,
                    "embedding_provider": self.config.embedding_provider,
                    "embedding_model": self._get_embedding_model_info(),
                    "persist_directory": self.config.persist_directory
                }
            else:
                return {
                    "name": self.config.collection_name,
                    "embedding_provider": self.config.embedding_provider,
                    "embedding_model": self._get_embedding_model_info(),
                    "persist_directory": self.config.persist_directory
                }
                
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {"error": str(e)}
    
    def reset_collection(self) -> bool:
        """
        컬렉션을 초기화합니다.
        
        Returns:
            초기화 성공 여부
        """
        try:
            if not self.vectorstore:
                raise ChromaDBError("VectorStore not initialized")
            
            # 기존 컬렉션의 모든 문서 삭제 (전체 리셋)
            if hasattr(self.vectorstore, '_collection'):
                collection = self.vectorstore._collection
                # Get all IDs and delete them
                all_data = collection.get()
                if all_data and 'ids' in all_data and all_data['ids']:
                    collection.delete(ids=all_data['ids'])
                    logger.info(f"🔄 컬렉션 '{self.config.collection_name}' 초기화 완료")
                else:
                    logger.info("컬렉션이 이미 비어있습니다")
                return True
            else:
                logger.warning("컬렉션 직접 접근 불가능")
                return False
                
        except Exception as e:
            logger.error(f"컬렉션 초기화 실패: {e}")
            return False
    
    def close(self):
        """데이터베이스 연결을 종료합니다."""
        try:
            # LangChain Chroma는 명시적인 close가 필요 없음
            self.vectorstore = None
            self.embeddings = None
            logger.info("💤 VectorDatabase 연결 종료")
        except Exception as e:
            logger.error(f"VectorDatabase 연결 종료 중 오류: {e}")
    
    def __enter__(self):
        """Context manager 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.close()


# Factory 함수들
def create_openai_vector_db(collection_name: str = "youtube_transcripts", 
                           persist_directory: str = "./chromadb_langchain",
                           openai_model: str = "text-embedding-3-large") -> VectorDatabase:
    """OpenAI 임베딩을 사용하는 벡터 데이터베이스를 생성합니다."""
    config = ChromaDBConfig(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_provider="openai",
        openai_model=openai_model
    )
    return VectorDatabase(config)


def create_huggingface_vector_db(collection_name: str = "youtube_transcripts",
                                 persist_directory: str = "./chromadb_langchain",
                                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> VectorDatabase:
    """HuggingFace 임베딩을 사용하는 벡터 데이터베이스를 생성합니다."""
    config = ChromaDBConfig(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_provider="huggingface",
        huggingface_model=model_name
    )
    return VectorDatabase(config) 