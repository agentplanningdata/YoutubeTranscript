"""
YouTube API 클라이언트 테스트

YouTube Data API v3 클라이언트의 초기화 및 기본 기능을 테스트합니다.
"""

import pytest
from unittest.mock import patch, MagicMock
import os

from src.services.youtube_api_client import YouTubeAPIClient
from src.utils.exceptions import YouTubeAPIError, ConfigurationError


def test_api_client_initialization():
    """YouTube API 클라이언트가 올바르게 초기화되는지 확인"""
    api_key = "test_api_key_123"
    
    # Mock the Google API client
    with patch('src.services.youtube_api_client.build') as mock_build:
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        
        client = YouTubeAPIClient(api_key=api_key)
        
        # 클라이언트 객체가 올바르게 생성되었는지 확인
        assert client is not None
        assert isinstance(client, YouTubeAPIClient)
        
        # API 키가 올바르게 저장되었는지 확인
        assert client.api_key == api_key
        
        # Google API 클라이언트가 올바른 파라미터로 빌드되었는지 확인
        mock_build.assert_called_once_with(
            'youtube', 'v3', developerKey=api_key
        )
        
        # 서비스 객체가 올바르게 설정되었는지 확인
        assert client.service == mock_service


def test_api_client_initialization_from_env():
    """환경변수에서 API 키를 가져와서 클라이언트가 초기화되는지 확인"""
    test_api_key = "env_api_key_456"
    
    with patch('src.services.youtube_api_client.build') as mock_build:
        with patch.dict(os.environ, {'YOUTUBE_API_KEY': test_api_key}):
            mock_service = MagicMock()
            mock_build.return_value = mock_service
            
            # API 키를 명시하지 않고 클라이언트 생성 (환경변수에서 가져와야 함)
            client = YouTubeAPIClient()
            
            assert client.api_key == test_api_key
            mock_build.assert_called_once_with(
                'youtube', 'v3', developerKey=test_api_key
            )


def test_api_client_initialization_without_api_key():
    """API 키 없이 클라이언트 초기화 시 예외 발생하는지 확인"""
    # 환경변수도 제거한 상태에서 테스트
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ConfigurationError, match="YouTube API key"):
            YouTubeAPIClient()


def test_api_client_initialization_with_empty_api_key():
    """빈 API 키로 클라이언트 초기화 시 예외 발생하는지 확인"""
    with pytest.raises(ConfigurationError, match="YouTube API key"):
        YouTubeAPIClient(api_key="")
    
    with pytest.raises(ConfigurationError, match="YouTube API key"):
        YouTubeAPIClient(api_key=None)


def test_api_client_initialization_with_invalid_credentials():
    """잘못된 API 키로 클라이언트 초기화 시 적절한 처리"""
    invalid_api_key = "invalid_key_123"
    
    # Google API 클라이언트 빌드는 성공하지만 실제 API 호출에서 실패할 수 있음
    with patch('src.services.youtube_api_client.build') as mock_build:
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        
        # 초기화 자체는 성공해야 함 (실제 검증은 API 호출시)
        client = YouTubeAPIClient(api_key=invalid_api_key)
        assert client.api_key == invalid_api_key
        assert client.service == mock_service


def test_invalid_api_key_raises_exception():
    """잘못된 API 키로 실제 API 호출 시 예외 발생하는지 확인"""
    invalid_api_key = "INVALID_API_KEY_123"
    
    with patch('src.services.youtube_api_client.build') as mock_build:
        # HttpError 클래스를 직접 생성 (src.services.youtube_api_client의 HttpError 사용)
        from src.services.youtube_api_client import HttpError
        
        # Mock service 설정
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        
        # HttpError 인스턴스 직접 생성
        http_error = HttpError("403 Forbidden")
        
        # search().list().execute() 호출 시 HttpError 발생하도록 설정  
        mock_search = MagicMock()
        mock_list = MagicMock() 
        mock_execute = MagicMock(side_effect=http_error)
        
        mock_search.list.return_value = mock_list
        mock_list.execute = mock_execute
        mock_service.search.return_value = mock_search
        
        # 클라이언트 생성
        client = YouTubeAPIClient(api_key=invalid_api_key)
        
        # API 호출 시 YouTubeAPIError 예외 발생해야 함
        with pytest.raises(YouTubeAPIError, match="Failed to search channels"):
            client.search_channels("test query")


def test_api_quota_exceeded_handling():
    """API 할당량 초과 시 적절한 예외 처리가 되는지 확인"""
    api_key = "valid_api_key_123"
    
    with patch('src.services.youtube_api_client.build') as mock_build:
        from src.services.youtube_api_client import HttpError
        
        # Mock service 설정
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        
        # Quota exceeded HttpError 생성
        quota_error = HttpError("403 Quota Exceeded")
        
        # search().list().execute() 호출 시 quota error 발생하도록 설정
        mock_search = MagicMock()
        mock_list = MagicMock()
        mock_execute = MagicMock(side_effect=quota_error)
        
        mock_search.list.return_value = mock_list
        mock_list.execute = mock_execute
        mock_service.search.return_value = mock_search
        
        # 클라이언트 생성
        client = YouTubeAPIClient(api_key=api_key)
        
        # API 호출 시 YouTubeAPIError 예외 발생해야 함
        with pytest.raises(YouTubeAPIError, match="Failed to search channels"):
            client.search_channels("test query")
        
        # get_channel_details도 같은 방식으로 처리되어야 함
        mock_channels = MagicMock()
        mock_channels_list = MagicMock()
        mock_channels_execute = MagicMock(side_effect=quota_error)
        
        mock_channels.list.return_value = mock_channels_list
        mock_channels_list.execute = mock_channels_execute
        mock_service.channels.return_value = mock_channels
        
        with pytest.raises(YouTubeAPIError, match="Failed to get channel details"):
            client.get_channel_details("test_channel_id")
        
        # get_channel_videos도 테스트
        with pytest.raises(YouTubeAPIError, match="Failed to get channel videos"):
            client.get_channel_videos("test_channel_id")
        
        # get_video_details도 테스트
        mock_videos = MagicMock()
        mock_videos_list = MagicMock()
        mock_videos_execute = MagicMock(side_effect=quota_error)
        
        mock_videos.list.return_value = mock_videos_list
        mock_videos_list.execute = mock_videos_execute
        mock_service.videos.return_value = mock_videos
        
        with pytest.raises(YouTubeAPIError, match="Failed to get video details"):
            client.get_video_details("test_video_id")


def test_api_client_has_required_methods():
    """API 클라이언트가 필수 메서드들을 가지고 있는지 확인"""
    api_key = "test_api_key_789"
    
    with patch('src.services.youtube_api_client.build') as mock_build:
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        
        client = YouTubeAPIClient(api_key=api_key)
        
        # 필수 메서드들이 존재하는지 확인
        assert hasattr(client, 'search_channels')
        assert hasattr(client, 'get_channel_details')
        assert hasattr(client, 'get_channel_videos')
        assert hasattr(client, 'get_video_details')
        
        # 메서드들이 callable인지 확인
        assert callable(getattr(client, 'search_channels'))
        assert callable(getattr(client, 'get_channel_details'))
        assert callable(getattr(client, 'get_channel_videos'))
        assert callable(getattr(client, 'get_video_details'))


def test_api_client_default_configuration():
    """API 클라이언트의 기본 설정이 올바른지 확인"""
    api_key = "test_api_key_config"
    
    with patch('src.services.youtube_api_client.build') as mock_build:
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        
        client = YouTubeAPIClient(api_key=api_key)
        
        # 기본 설정값들 확인
        assert hasattr(client, 'max_results')
        assert client.max_results > 0
        assert client.max_results <= 50  # YouTube API 기본 제한
        
        assert hasattr(client, 'region_code')
        assert hasattr(client, 'language')


def test_api_client_service_version():
    """올바른 YouTube API 버전을 사용하는지 확인"""
    api_key = "test_api_key_version"
    
    with patch('src.services.youtube_api_client.build') as mock_build:
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        
        YouTubeAPIClient(api_key=api_key)
        
        # YouTube Data API v3를 사용하는지 확인
        mock_build.assert_called_once_with(
            'youtube', 'v3', developerKey=api_key
        ) 