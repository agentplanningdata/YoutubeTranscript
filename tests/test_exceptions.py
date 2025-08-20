"""
커스텀 예외 클래스 테스트

프로젝트에서 사용하는 커스텀 예외 클래스들이 올바르게 정의되고 동작하는지 확인합니다.
"""

import pytest

from src.utils.exceptions import (
    YouTubeTranscriptError,
    YouTubeAPIError,
    TranscriptNotFoundError,
    TranscriptProcessingError,
    RAGError,
    VectorSearchError,
    DatabaseConnectionError,
    ConfigurationError
)


def test_base_exception_exists():
    """기본 예외 클래스가 존재하는지 확인"""
    assert YouTubeTranscriptError is not None
    assert issubclass(YouTubeTranscriptError, Exception)


def test_youtube_api_error_exists():
    """YouTube API 관련 예외 클래스가 존재하는지 확인"""
    assert YouTubeAPIError is not None
    assert issubclass(YouTubeAPIError, YouTubeTranscriptError)


def test_transcript_not_found_error_exists():
    """자막 찾기 실패 예외 클래스가 존재하는지 확인"""
    assert TranscriptNotFoundError is not None
    assert issubclass(TranscriptNotFoundError, YouTubeTranscriptError)


def test_transcript_processing_error_exists():
    """자막 처리 예외 클래스가 존재하는지 확인"""
    assert TranscriptProcessingError is not None
    assert issubclass(TranscriptProcessingError, YouTubeTranscriptError)


def test_rag_error_exists():
    """RAG 관련 예외 클래스가 존재하는지 확인"""
    assert RAGError is not None
    assert issubclass(RAGError, YouTubeTranscriptError)


def test_vector_search_error_exists():
    """벡터 검색 예외 클래스가 존재하는지 확인"""
    assert VectorSearchError is not None
    assert issubclass(VectorSearchError, RAGError)


def test_database_connection_error_exists():
    """데이터베이스 연결 예외 클래스가 존재하는지 확인"""
    assert DatabaseConnectionError is not None
    assert issubclass(DatabaseConnectionError, YouTubeTranscriptError)


def test_configuration_error_exists():
    """설정 관련 예외 클래스가 존재하는지 확인"""
    assert ConfigurationError is not None
    assert issubclass(ConfigurationError, YouTubeTranscriptError)


def test_exceptions_can_be_raised_with_message():
    """예외들이 메시지와 함께 발생할 수 있는지 확인"""
    test_message = "Test error message"
    
    with pytest.raises(YouTubeTranscriptError, match=test_message):
        raise YouTubeTranscriptError(test_message)
    
    with pytest.raises(YouTubeAPIError, match=test_message):
        raise YouTubeAPIError(test_message)
    
    with pytest.raises(TranscriptNotFoundError, match=test_message):
        raise TranscriptNotFoundError(test_message)


def test_exceptions_can_be_raised_without_message():
    """예외들이 메시지 없이도 발생할 수 있는지 확인"""
    with pytest.raises(YouTubeTranscriptError):
        raise YouTubeTranscriptError()
    
    with pytest.raises(YouTubeAPIError):
        raise YouTubeAPIError()


def test_exception_inheritance_chain():
    """예외 상속 체인이 올바른지 확인"""
    # VectorSearchError는 RAGError를 상속하고, RAGError는 YouTubeTranscriptError를 상속
    error = VectorSearchError("test")
    
    assert isinstance(error, VectorSearchError)
    assert isinstance(error, RAGError)
    assert isinstance(error, YouTubeTranscriptError)
    assert isinstance(error, Exception)


def test_exceptions_have_proper_attributes():
    """예외들이 적절한 속성을 가지는지 확인"""
    message = "Test error"
    error = YouTubeAPIError(message)
    
    assert str(error) == message
    assert error.args == (message,)


def test_youtube_api_error_with_status_code():
    """YouTube API 에러가 상태 코드와 함께 생성될 수 있는지 확인"""
    message = "API quota exceeded"
    status_code = 403
    
    # YouTube API 에러가 추가 정보를 받을 수 있는지 테스트
    error = YouTubeAPIError(message, status_code=status_code)
    
    assert str(error) == message
    assert hasattr(error, 'status_code')
    assert error.status_code == status_code


def test_transcript_not_found_error_with_video_id():
    """자막 찾기 실패 에러가 비디오 ID와 함께 생성될 수 있는지 확인"""
    message = "Transcript not available"
    video_id = "test_video_123"
    
    error = TranscriptNotFoundError(message, video_id=video_id)
    
    assert str(error) == message
    assert hasattr(error, 'video_id')
    assert error.video_id == video_id


def test_database_connection_error_with_details():
    """데이터베이스 연결 에러가 상세 정보와 함께 생성될 수 있는지 확인"""
    message = "Connection failed"
    database_url = "postgresql://localhost/test"
    
    error = DatabaseConnectionError(message, database_url=database_url)
    
    assert str(error) == message
    assert hasattr(error, 'database_url')
    assert error.database_url == database_url 