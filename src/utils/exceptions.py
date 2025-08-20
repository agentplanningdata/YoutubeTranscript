"""
커스텀 예외 클래스들

YouTube Transcript PoC 프로젝트에서 사용하는 예외 클래스들을 정의합니다.
"""

from typing import Optional, Any


class YouTubeTranscriptError(Exception):
    """YouTube Transcript PoC 프로젝트의 기본 예외 클래스"""
    
    def __init__(self, message: str = "", **kwargs):
        super().__init__(message)
        # 추가 속성들을 동적으로 설정
        for key, value in kwargs.items():
            setattr(self, key, value)


class YouTubeAPIError(YouTubeTranscriptError):
    """YouTube API 관련 예외"""
    
    def __init__(self, message: str = "", status_code: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code


class TranscriptNotFoundError(YouTubeTranscriptError):
    """자막을 찾을 수 없을 때 발생하는 예외"""
    
    def __init__(self, message: str = "", video_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.video_id = video_id


class TranscriptProcessingError(YouTubeTranscriptError):
    """자막 처리 중 발생하는 예외"""
    pass


class RAGError(YouTubeTranscriptError):
    """RAG 시스템 관련 예외"""
    pass


class VectorSearchError(RAGError):
    """벡터 검색 관련 예외"""
    pass


class DatabaseConnectionError(YouTubeTranscriptError):
    """데이터베이스 연결 관련 예외"""
    
    def __init__(self, message: str = "", database_url: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.database_url = database_url


class ConfigurationError(YouTubeTranscriptError):
    """설정 관련 예외"""
    pass


class DatabaseError(YouTubeTranscriptError):
    """데이터베이스 관련 예외"""
    
    def __init__(self, message: str = "", operation: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.operation = operation


class ChannelNotFoundError(YouTubeAPIError):
    """채널을 찾을 수 없을 때 발생하는 예외"""
    
    def __init__(self, message: str = "", channel_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.channel_name = channel_name


class VideoProcessingError(YouTubeTranscriptError):
    """비디오 처리 관련 예외"""
    
    def __init__(self, message: str = "", video_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.video_id = video_id


class EmbeddingError(RAGError):
    """임베딩 생성 관련 예외"""
    pass


class ChromaDBError(VectorSearchError):
    """ChromaDB 관련 예외"""
    pass


class VectorStoreError(VectorSearchError):
    """벡터 저장소 관련 예외"""
    pass 