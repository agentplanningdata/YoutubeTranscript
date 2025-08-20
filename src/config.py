"""
환경 설정 관리

애플리케이션에서 사용하는 환경 변수들을 중앙에서 관리합니다.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    """환경 설정 클래스"""
    
    # YouTube API 설정
    youtube_api_key: Optional[str] = None
    
    # OpenAI API 설정
    openai_api_key: Optional[str] = None
    
    # Google API 설정 (Gemini 모델용)
    google_api_key: Optional[str] = None
    
    # ChromaDB 설정  
    chroma_db_path: str = "./data/chroma_db"
    chroma_collection_name: str = "youtube_transcripts"
    
    # SQLite 설정
    sqlite_db_path: str = "./data/youtube_transcript.db"
    
    # Langfuse 설정
    langfuse_secret_key: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"
    
    def __post_init__(self):
        """환경 변수에서 값 로드"""
        # YouTube API
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        
        # OpenAI API
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Google API
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # ChromaDB 설정
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", self.chroma_db_path)
        self.chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", self.chroma_collection_name)
        
        # SQLite 설정
        self.sqlite_db_path = os.getenv("SQLITE_DB_PATH", self.sqlite_db_path)
        
        # Langfuse 설정
        self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_host = os.getenv("LANGFUSE_HOST", self.langfuse_host)
    
    def validate_youtube_api_key(self) -> bool:
        """YouTube API 키 유효성 검사"""
        return self.youtube_api_key is not None and len(self.youtube_api_key.strip()) > 0
    
    def validate_openai_api_key(self) -> bool:
        """OpenAI API 키 유효성 검사"""  
        return self.openai_api_key is not None and len(self.openai_api_key.strip()) > 0
    
    def validate_langfuse_config(self) -> bool:
        """Langfuse 설정 유효성 검사"""
        return (self.langfuse_secret_key is not None and 
                self.langfuse_public_key is not None and
                len(self.langfuse_secret_key.strip()) > 0 and
                len(self.langfuse_public_key.strip()) > 0)
    
    def is_langfuse_enabled(self) -> bool:
        """Langfuse 사용 가능 여부 확인"""
        return self.validate_langfuse_config()


# 전역 설정 객체
config = Config() 