"""
영상 정보 조회 기능 테스트

YouTube API를 통한 영상 정보 조회 기능을 테스트합니다.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.services.video_info import VideoInfoService
from src.utils.exceptions import YouTubeAPIError, VideoProcessingError


def test_get_video_details():
    """영상 상세 정보 조회가 올바르게 작동하는지 확인"""
    
    # Mock YouTube API 응답 데이터
    mock_video_response = {
        'video_id': 'dQw4w9WgXcQ',
        'title': '테스트 영상 제목',
        'description': '테스트 영상의 상세 설명입니다. 이것은 테스트용 영상입니다.',
        'channel_id': 'UCrAOnxB_QYr5N7YKDfKK9fA',
        'channel_title': '테스트 채널',
        'published_at': '2023-01-15T10:30:00Z',
        'duration': 'PT3M45S',  # ISO 8601 duration format
        'view_count': 1000000,
        'like_count': 50000,
        'comment_count': 2500,
        'thumbnail_url': 'https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
        'tags': ['테스트', '영상', 'YouTube', '샘플'],
        'category_id': '22',  # People & Blogs
        'default_language': 'ko',
        'default_audio_language': 'ko'
    }
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        # Mock YouTube API 클라이언트
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.get_video_details.return_value = mock_video_response
        
        # 영상 정보 서비스 생성
        video_service = VideoInfoService()
        
        # 영상 상세 정보 조회 실행
        result = video_service.get_video_details("dQw4w9WgXcQ")
        
        # 결과 검증
        assert result is not None
        assert result['video_id'] == 'dQw4w9WgXcQ'
        assert result['title'] == '테스트 영상 제목'
        assert result['channel_id'] == 'UCrAOnxB_QYr5N7YKDfKK9fA'
        assert result['view_count'] == 1000000
        assert result['like_count'] == 50000
        assert result['comment_count'] == 2500
        assert result['duration'] == 'PT3M45S'
        assert isinstance(result['tags'], list)
        assert '테스트' in result['tags']
        
        # YouTube API 클라이언트가 올바른 매개변수로 호출되었는지 확인
        mock_youtube_client.get_video_details.assert_called_once_with("dQw4w9WgXcQ")


def test_get_video_details_with_invalid_id():
    """유효하지 않은 영상 ID로 조회 시 예외 발생하는지 확인"""
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        video_service = VideoInfoService()
        
        # 빈 영상 ID
        with pytest.raises(ValueError, match="Video ID cannot be empty"):
            video_service.get_video_details("")
        
        with pytest.raises(ValueError, match="Video ID cannot be empty"):
            video_service.get_video_details("   ")
        
        with pytest.raises(ValueError, match="Video ID cannot be empty"):
            video_service.get_video_details(None)


def test_get_video_details_not_found():
    """존재하지 않는 영상 ID로 조회 시 예외 발생하는지 확인"""
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        # 존재하지 않는 영상에 대해 YouTubeAPIError 발생
        mock_youtube_client.get_video_details.side_effect = YouTubeAPIError("Video not found")
        
        video_service = VideoInfoService()
        
        # VideoProcessingError로 변환되어야 함
        with pytest.raises(VideoProcessingError, match="Failed to get video details"):
            video_service.get_video_details("invalid_video_id")


def test_get_video_details_private_or_deleted():
    """비공개 또는 삭제된 영상 조회 시 처리 확인"""
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        # 비공개 영상에 대해 빈 응답 또는 제한된 정보
        mock_youtube_client.get_video_details.return_value = {
            'video_id': 'private_video_123',
            'title': None,  # 비공개 영상은 제목이 null일 수 있음
            'description': None,
            'channel_id': 'UCrAOnxB_QYr5N7YKDfKK9fA',
            'published_at': None,
            'view_count': None,
            'like_count': None
        }
        
        video_service = VideoInfoService()
        result = video_service.get_video_details("private_video_123")
        
        # 기본 필드들은 존재해야 하지만 None 값일 수 있음
        assert result['video_id'] == 'private_video_123'
        assert result['channel_id'] == 'UCrAOnxB_QYr5N7YKDfKK9fA'
        assert result['title'] is None or result['title'] == ""
        assert result['view_count'] is None


def test_get_video_details_with_statistics():
    """영상 통계 정보가 포함된 상세 조회 확인"""
    
    mock_video_response = {
        'video_id': 'stats_test_video',
        'title': '통계 테스트 영상',
        'description': '통계 정보가 포함된 테스트 영상',
        'channel_id': 'UCstatsTest123456789',
        'channel_title': '통계 테스트 채널',
        'published_at': '2023-06-01T15:00:00Z',
        'duration': 'PT10M30S',
        'view_count': 2500000,
        'like_count': 125000,
        'comment_count': 8500,
        'thumbnail_url': 'https://i.ytimg.com/vi/stats_test_video/maxresdefault.jpg',
        'category_id': '28',  # Science & Technology
        'tags': ['통계', '데이터', '분석', '테스트'],
        'definition': 'hd',  # HD 화질
        'caption': 'true'    # 자막 사용 가능
    }
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.get_video_details.return_value = mock_video_response
        
        video_service = VideoInfoService()
        result = video_service.get_video_details("stats_test_video")
        
        # 통계 정보 검증
        assert result['view_count'] == 2500000
        assert result['like_count'] == 125000
        assert result['comment_count'] == 8500
        
        # 추가 메타데이터 검증
        assert result['category_id'] == '28'
        assert result['definition'] == 'hd'
        assert result['caption'] == 'true'
        
        # 태그 검증
        assert isinstance(result['tags'], list)
        assert len(result['tags']) == 4
        assert '통계' in result['tags']
        assert '데이터' in result['tags']


def test_get_video_details_duration_parsing():
    """영상 길이 형식이 올바르게 처리되는지 확인"""
    
    duration_test_cases = [
        ('PT1M30S', '1분 30초'),
        ('PT1H25M10S', '1시간 25분 10초'),
        ('PT45S', '45초'),
        ('PT2H', '2시간'),
        ('PT30M', '30분')
    ]
    
    for iso_duration, expected_readable in duration_test_cases:
        mock_response = {
            'video_id': 'duration_test',
            'title': f'Duration Test {iso_duration}',
            'description': 'Duration test video',
            'channel_id': 'UCdurationTest123',
            'published_at': '2023-01-01T00:00:00Z',
            'duration': iso_duration,
            'view_count': 1000,
            'like_count': 100,
            'thumbnail_url': 'https://i.ytimg.com/vi/duration_test/default.jpg'
        }
        
        with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
            mock_youtube_client = MagicMock()
            mock_client.return_value = mock_youtube_client
            mock_youtube_client.get_video_details.return_value = mock_response
            
            video_service = VideoInfoService()
            result = video_service.get_video_details("duration_test")
            
            # ISO 8601 형식이 그대로 반환되는지 확인
            assert result['duration'] == iso_duration
            
            # 서비스에서 readable format도 제공하는지 확인 (선택사항)
            if 'duration_readable' in result:
                assert isinstance(result['duration_readable'], str)


def test_get_video_details_error_handling():
    """영상 정보 조회 중 다양한 오류 상황 처리 확인"""
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        video_service = VideoInfoService()
        
        # API 할당량 초과
        mock_youtube_client.get_video_details.side_effect = YouTubeAPIError("Quota exceeded")
        
        with pytest.raises(VideoProcessingError, match="Failed to get video details"):
            video_service.get_video_details("quota_test_video")
        
        # 네트워크 오류 시뮬레이션
        mock_youtube_client.get_video_details.side_effect = Exception("Network error")
        
        with pytest.raises(VideoProcessingError, match="Failed to get video details"):
            video_service.get_video_details("network_error_video")


def test_get_video_details_service_initialization():
    """영상 정보 서비스가 올바르게 초기화되는지 확인"""
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        video_service = VideoInfoService()
        
        # 서비스가 올바르게 생성되었는지 확인
        assert video_service is not None
        assert isinstance(video_service, VideoInfoService)
        
        # YouTube API 클라이언트가 초기화되었는지 확인
        mock_client.assert_called_once()


def test_get_video_details_with_api_key():
    """API 키를 지정해서 영상 정보 서비스를 생성할 수 있는지 확인"""
    
    api_key = "test_video_api_key_123"
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        video_service = VideoInfoService(api_key=api_key)
        
        # 지정한 API 키로 YouTube 클라이언트가 생성되었는지 확인
        mock_client.assert_called_once_with(api_key=api_key)


def test_get_channel_videos():
    """채널의 영상 목록 조회가 올바르게 작동하는지 확인"""
    
    # Mock YouTube API 응답 데이터 (채널의 영상 목록)
    mock_channel_videos_response = [
        {
            'video_id': 'channel_video_1',
            'title': '채널 영상 1',
            'description': '첫 번째 채널 영상입니다.',
            'channel_id': 'UCChannelTest123456789',
            'channel_title': '테스트 채널',
            'published_at': '2023-12-01T10:00:00Z',
            'duration': 'PT5M30S',
            'view_count': 15000,
            'like_count': 750,
            'comment_count': 45,
            'thumbnail_url': 'https://i.ytimg.com/vi/channel_video_1/maxresdefault.jpg',
            'tags': ['채널', '영상', '첫번째']
        },
        {
            'video_id': 'channel_video_2',
            'title': '채널 영상 2',
            'description': '두 번째 채널 영상입니다.',
            'channel_id': 'UCChannelTest123456789',
            'channel_title': '테스트 채널',
            'published_at': '2023-11-28T15:30:00Z',
            'duration': 'PT8M15S',
            'view_count': 22000,
            'like_count': 1100,
            'comment_count': 78,
            'thumbnail_url': 'https://i.ytimg.com/vi/channel_video_2/maxresdefault.jpg',
            'tags': ['채널', '영상', '두번째']
        },
        {
            'video_id': 'channel_video_3',
            'title': '채널 영상 3',
            'description': '세 번째 채널 영상입니다.',
            'channel_id': 'UCChannelTest123456789',
            'channel_title': '테스트 채널',
            'published_at': '2023-11-25T09:00:00Z',
            'duration': 'PT12M45S',
            'view_count': 18500,
            'like_count': 920,
            'comment_count': 62,
            'thumbnail_url': 'https://i.ytimg.com/vi/channel_video_3/maxresdefault.jpg',
            'tags': ['채널', '영상', '세번째']
        }
    ]
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        # Mock YouTube API 클라이언트
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.get_channel_videos.return_value = mock_channel_videos_response
        
        # 영상 정보 서비스 생성
        video_service = VideoInfoService()
        
        # 채널 영상 목록 조회 실행
        results = video_service.get_channel_videos("UCChannelTest123456789")
        
        # 결과 검증
        assert results is not None
        assert len(results) == 3
        
        # 첫 번째 영상 검증
        assert results[0]['video_id'] == 'channel_video_1'
        assert results[0]['title'] == '채널 영상 1'
        assert results[0]['channel_id'] == 'UCChannelTest123456789'
        assert results[0]['view_count'] == 15000
        assert results[0]['duration'] == 'PT5M30S'
        
        # 두 번째 영상 검증
        assert results[1]['video_id'] == 'channel_video_2'
        assert results[1]['title'] == '채널 영상 2'
        assert results[1]['view_count'] == 22000
        
        # 세 번째 영상 검증
        assert results[2]['video_id'] == 'channel_video_3'
        assert results[2]['title'] == '채널 영상 3'
        assert results[2]['view_count'] == 18500
        
        # YouTube API 클라이언트가 올바른 매개변수로 호출되었는지 확인
        mock_youtube_client.get_channel_videos.assert_called_once_with(
            channel_id="UCChannelTest123456789", 
            max_results=20
        )


def test_get_channel_videos_with_custom_max_results():
    """채널 영상 목록 조회 시 max_results 파라미터 적용 확인"""
    
    mock_channel_videos_response = [
        {
            'video_id': f'video_{i}',
            'title': f'영상 {i}',
            'description': f'{i}번째 영상',
            'channel_id': 'UCCustomMaxResults123',
            'published_at': f'2023-12-{i:02d}T10:00:00Z',
            'view_count': 1000 * i,
            'thumbnail_url': f'https://i.ytimg.com/vi/video_{i}/default.jpg'
        }
        for i in range(1, 6)  # 5개 영상
    ]
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.get_channel_videos.return_value = mock_channel_videos_response
        
        video_service = VideoInfoService()
        
        # max_results=5로 지정해서 조회
        results = video_service.get_channel_videos("UCCustomMaxResults123", max_results=5)
        
        assert len(results) == 5
        assert results[0]['video_id'] == 'video_1'
        assert results[4]['video_id'] == 'video_5'
        
        # 지정한 max_results로 API가 호출되었는지 확인
        mock_youtube_client.get_channel_videos.assert_called_once_with(
            channel_id="UCCustomMaxResults123", 
            max_results=5
        )


def test_get_channel_videos_empty_channel():
    """영상이 없는 채널 조회 시 빈 리스트 반환 확인"""
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        # 영상이 없는 채널
        mock_youtube_client.get_channel_videos.return_value = []
        
        video_service = VideoInfoService()
        results = video_service.get_channel_videos("UCEmptyChannel123456789")
        
        assert results == []
        assert len(results) == 0
        assert isinstance(results, list)


def test_get_channel_videos_invalid_channel_id():
    """유효하지 않은 채널 ID로 조회 시 예외 발생 확인"""
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        video_service = VideoInfoService()
        
        # 빈 채널 ID
        with pytest.raises(ValueError, match="Channel ID cannot be empty"):
            video_service.get_channel_videos("")
        
        with pytest.raises(ValueError, match="Channel ID cannot be empty"):
            video_service.get_channel_videos("   ")
        
        with pytest.raises(ValueError, match="Channel ID cannot be empty"):
            video_service.get_channel_videos(None)


def test_get_channel_videos_api_error():
    """채널 영상 조회 중 API 오류 시 예외 처리 확인"""
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        # API 오류 발생 시뮬레이션
        mock_youtube_client.get_channel_videos.side_effect = YouTubeAPIError("Channel not found")
        
        video_service = VideoInfoService()
        
        # VideoProcessingError로 변환되어야 함
        with pytest.raises(VideoProcessingError, match="Failed to get videos for channel"):
            video_service.get_channel_videos("UCNonexistentChannel123")


def test_get_channel_videos_with_various_video_types():
    """다양한 타입의 영상(쇼츠, 일반 영상, 라이브 등)이 포함된 채널 조회"""
    
    mock_mixed_videos_response = [
        {
            'video_id': 'regular_video_1',
            'title': '일반 영상',
            'description': '일반적인 긴 영상입니다.',
            'channel_id': 'UCMixedContent123456',
            'published_at': '2023-12-01T10:00:00Z',
            'duration': 'PT15M30S',  # 15분 30초
            'view_count': 50000,
            'like_count': 2500,
            'thumbnail_url': 'https://i.ytimg.com/vi/regular_video_1/maxresdefault.jpg'
        },
        {
            'video_id': 'shorts_video_1',
            'title': '쇼츠 영상',
            'description': '짧은 쇼츠 영상입니다.',
            'channel_id': 'UCMixedContent123456',
            'published_at': '2023-11-30T14:00:00Z',
            'duration': 'PT45S',  # 45초
            'view_count': 125000,
            'like_count': 8500,
            'thumbnail_url': 'https://i.ytimg.com/vi/shorts_video_1/maxresdefault.jpg'
        },
        {
            'video_id': 'live_stream_1',
            'title': '라이브 방송',
            'description': '라이브 스트리밍 영상입니다.',
            'channel_id': 'UCMixedContent123456',
            'published_at': '2023-11-29T19:00:00Z',
            'duration': 'PT2H15M30S',  # 2시간 15분 30초
            'view_count': 35000,
            'like_count': 1800,
            'thumbnail_url': 'https://i.ytimg.com/vi/live_stream_1/maxresdefault.jpg'
        }
    ]
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.get_channel_videos.return_value = mock_mixed_videos_response
        
        video_service = VideoInfoService()
        results = video_service.get_channel_videos("UCMixedContent123456")
        
        assert len(results) == 3
        
        # 일반 영상 확인
        regular_video = results[0]
        assert regular_video['video_id'] == 'regular_video_1'
        assert regular_video['duration'] == 'PT15M30S'
        assert 'duration_readable' in regular_video  # 읽기 쉬운 형태 추가되었는지 확인
        
        # 쇼츠 영상 확인
        shorts_video = results[1]
        assert shorts_video['video_id'] == 'shorts_video_1'
        assert shorts_video['duration'] == 'PT45S'
        assert shorts_video['view_count'] == 125000
        
        # 라이브 방송 확인
        live_video = results[2]
        assert live_video['video_id'] == 'live_stream_1'
        assert live_video['duration'] == 'PT2H15M30S'
        

def test_get_channel_videos_pagination_simulation():
    """채널 영상 목록의 페이징 시나리오 테스트"""
    
    # 첫 페이지 영상들
    first_page_videos = [
        {
            'video_id': f'page1_video_{i}',
            'title': f'첫 페이지 영상 {i}',
            'description': f'첫 페이지의 {i}번째 영상',
            'channel_id': 'UCPaginationTest123',
            'published_at': f'2023-12-{i:02d}T10:00:00Z',
            'view_count': 10000 + i * 1000,
            'thumbnail_url': f'https://i.ytimg.com/vi/page1_video_{i}/default.jpg'
        }
        for i in range(1, 11)  # 10개 영상
    ]
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.get_channel_videos.return_value = first_page_videos
        
        video_service = VideoInfoService()
        
        # 첫 페이지 조회
        results = video_service.get_channel_videos("UCPaginationTest123", max_results=10)
        
        assert len(results) == 10
        assert results[0]['video_id'] == 'page1_video_1'
        assert results[9]['video_id'] == 'page1_video_10'
        
        # 모든 영상이 같은 채널에서 온 것인지 확인
        for video in results:
            assert video['channel_id'] == 'UCPaginationTest123'


def test_video_metadata_validation():
    """영상 메타데이터의 유효성 검증이 올바르게 작동하는지 확인"""
    
    # 완전한 메타데이터를 가진 영상
    complete_video_metadata = {
        'video_id': 'dQw4w9WgXcQ',
        'title': '완전한 메타데이터 영상',
        'description': '이 영상은 모든 메타데이터를 가지고 있습니다.',
        'channel_id': 'UCCompleteMetadata123',
        'channel_title': '완전한 메타데이터 채널',
        'published_at': '2023-06-15T14:30:00Z',
        'duration': 'PT4M30S',
        'view_count': 1500000,
        'like_count': 75000,
        'comment_count': 3200,
        'thumbnail_url': 'https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
        'tags': ['테스트', '메타데이터', '검증', '완전한'],
        'category_id': '22',
        'default_language': 'ko',
        'default_audio_language': 'ko'
    }
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.get_video_details.return_value = complete_video_metadata
        
        video_service = VideoInfoService()
        result = video_service.get_video_details("dQw4w9WgXcQ")
        
        # 필수 메타데이터 필드 존재 확인
        required_fields = [
            'video_id', 'title', 'description', 'channel_id',
            'published_at', 'thumbnail_url'
        ]
        
        for field in required_fields:
            assert field in result, f"Required field '{field}' is missing"
            assert result[field] is not None, f"Field '{field}' should not be None"
        
        # 데이터 타입 검증
        assert isinstance(result['video_id'], str)
        assert isinstance(result['title'], str)
        assert isinstance(result['description'], str)
        assert isinstance(result['channel_id'], str)
        assert isinstance(result['published_at'], str)
        assert isinstance(result['thumbnail_url'], str)
        
        # 선택적 필드 타입 검증 (있는 경우)
        if 'view_count' in result and result['view_count'] is not None:
            assert isinstance(result['view_count'], int)
        
        if 'like_count' in result and result['like_count'] is not None:
            assert isinstance(result['like_count'], int)
        
        if 'comment_count' in result and result['comment_count'] is not None:
            assert isinstance(result['comment_count'], int)
        
        if 'tags' in result:
            assert isinstance(result['tags'], list)
            for tag in result['tags']:
                assert isinstance(tag, str)


def test_video_metadata_validation_with_missing_optional_fields():
    """선택적 메타데이터가 누락된 영상의 검증 처리 확인"""
    
    # 일부 선택적 필드가 누락된 영상
    minimal_video_metadata = {
        'video_id': 'minimal_video_123',
        'title': '최소한의 메타데이터 영상',
        'description': '일부 필드만 있는 영상입니다.',
        'channel_id': 'UCMinimalMetadata123',
        'published_at': '2023-08-20T12:00:00Z',
        'thumbnail_url': 'https://i.ytimg.com/vi/minimal_video_123/default.jpg',
        'view_count': None,  # 조회수 없음
        'like_count': None,  # 좋아요 없음
        'comment_count': None,  # 댓글 수 없음
        'tags': [],  # 태그 없음
        'duration': 'PT0S'  # 길이 정보 없음
    }
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.get_video_details.return_value = minimal_video_metadata
        
        video_service = VideoInfoService()
        result = video_service.get_video_details("minimal_video_123")
        
        # 필수 필드는 여전히 존재해야 함
        assert result['video_id'] == 'minimal_video_123'
        assert result['title'] == '최소한의 메타데이터 영상'
        assert result['description'] == '일부 필드만 있는 영상입니다.'
        assert result['channel_id'] == 'UCMinimalMetadata123'
        
        # 선택적 필드들이 적절히 처리되었는지 확인
        assert 'view_count' in result
        assert 'like_count' in result
        assert 'comment_count' in result
        assert 'tags' in result
        
        # None 값들이 그대로 유지되거나 적절히 처리되는지 확인
        assert result['view_count'] is None
        assert result['like_count'] is None 
        assert result['comment_count'] is None
        assert isinstance(result['tags'], list)
        assert len(result['tags']) == 0


def test_video_metadata_validation_invalid_data_types():
    """잘못된 데이터 타입의 메타데이터 처리 확인"""
    
    # 잘못된 타입의 데이터가 포함된 영상
    invalid_type_metadata = {
        'video_id': 'invalid_types_video',
        'title': '잘못된 타입 테스트',
        'description': '데이터 타입이 올바르지 않은 영상입니다.',
        'channel_id': 'UCInvalidTypes123',
        'published_at': '2023-09-01T16:00:00Z',
        'thumbnail_url': 'https://i.ytimg.com/vi/invalid_types_video/default.jpg',
        'view_count': "150000",  # 문자열 (숫자여야 함)
        'like_count': "7500",    # 문자열 (숫자여야 함)
        'comment_count': "320",  # 문자열 (숫자여야 함)
        'tags': "tag1,tag2,tag3",  # 문자열 (리스트여야 함)
        'duration': 'PT5M45S'
    }
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.get_video_details.return_value = invalid_type_metadata
        
        video_service = VideoInfoService()
        result = video_service.get_video_details("invalid_types_video")
        
        # 기본 필드들은 그대로 유지
        assert result['video_id'] == 'invalid_types_video'
        assert result['title'] == '잘못된 타입 테스트'
        
        # 숫자 필드들이 적절히 변환되었는지 확인
        if 'view_count' in result and result['view_count'] is not None:
            assert isinstance(result['view_count'], int)
            assert result['view_count'] == 150000
        
        if 'like_count' in result and result['like_count'] is not None:
            assert isinstance(result['like_count'], int)
            assert result['like_count'] == 7500
        
        if 'comment_count' in result and result['comment_count'] is not None:
            assert isinstance(result['comment_count'], int)
            assert result['comment_count'] == 320


def test_video_metadata_validation_datetime_formats():
    """다양한 날짜 형식의 메타데이터 검증"""
    
    # 다양한 날짜 형식들
    datetime_test_cases = [
        {
            'video_id': 'datetime_test_1',
            'published_at': '2023-06-15T14:30:00Z'  # 표준 ISO 형식
        },
        {
            'video_id': 'datetime_test_2', 
            'published_at': '2023-06-15T14:30:00.000Z'  # 밀리초 포함
        },
        {
            'video_id': 'datetime_test_3',
            'published_at': '2023-06-15T14:30:00+00:00'  # 시간대 표시
        }
    ]
    
    for test_case in datetime_test_cases:
        base_metadata = {
            'title': '날짜 형식 테스트',
            'description': '날짜 형식 검증용 영상',
            'channel_id': 'UCDateTimeTest123',
            'thumbnail_url': f'https://i.ytimg.com/vi/{test_case["video_id"]}/default.jpg',
            **test_case
        }
        
        with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
            mock_youtube_client = MagicMock()
            mock_client.return_value = mock_youtube_client
            mock_youtube_client.get_video_details.return_value = base_metadata
            
            video_service = VideoInfoService()
            result = video_service.get_video_details(test_case['video_id'])
            
            # 날짜 필드가 문자열로 유지되는지 확인
            assert isinstance(result['published_at'], str)
            assert len(result['published_at']) > 0
            
            # 날짜 형식이 유효한지 확인 (파싱 가능한지)
            from datetime import datetime
            try:
                datetime.fromisoformat(result['published_at'].replace('Z', '+00:00'))
            except ValueError:
                pytest.fail(f"Invalid datetime format: {result['published_at']}")


def test_video_metadata_validation_thumbnail_urls():
    """썸네일 URL 형식의 메타데이터 검증"""
    
    # 다양한 썸네일 URL 형식들
    thumbnail_test_cases = [
        'https://i.ytimg.com/vi/test_video/maxresdefault.jpg',
        'https://i.ytimg.com/vi/test_video/hqdefault.jpg', 
        'https://i.ytimg.com/vi/test_video/mqdefault.jpg',
        'https://i.ytimg.com/vi/test_video/default.jpg'
    ]
    
    for i, thumbnail_url in enumerate(thumbnail_test_cases, 1):
        video_metadata = {
            'video_id': f'thumbnail_test_{i}',
            'title': f'썸네일 테스트 {i}',
            'description': f'썸네일 URL 검증용 영상 {i}',
            'channel_id': 'UCThumbnailTest123',
            'published_at': '2023-07-10T10:00:00Z',
            'thumbnail_url': thumbnail_url
        }
        
        with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
            mock_youtube_client = MagicMock()
            mock_client.return_value = mock_youtube_client
            mock_youtube_client.get_video_details.return_value = video_metadata
            
            video_service = VideoInfoService()
            result = video_service.get_video_details(f'thumbnail_test_{i}')
            
            # 썸네일 URL 검증
            assert isinstance(result['thumbnail_url'], str)
            assert result['thumbnail_url'].startswith('https://')
            assert 'i.ytimg.com' in result['thumbnail_url']
            assert result['thumbnail_url'] == thumbnail_url


def test_video_metadata_validation_edge_cases():
    """메타데이터 검증의 엣지 케이스 처리"""
    
    # 극단적인 케이스들
    edge_case_metadata = {
        'video_id': 'edge_case_test',
        'title': '',  # 빈 제목
        'description': None,  # None 설명
        'channel_id': 'UCEdgeCaseTest123',
        'published_at': '2023-05-01T00:00:00Z',
        'thumbnail_url': 'https://i.ytimg.com/vi/edge_case_test/default.jpg',
        'view_count': 0,  # 조회수 0
        'like_count': -1,  # 음수 (잘못된 값)
        'comment_count': 999999999,  # 매우 큰 수
        'tags': None,  # None 태그
        'duration': 'PT0S'  # 길이 0초
    }
    
    with patch('src.services.video_info.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.get_video_details.return_value = edge_case_metadata
        
        video_service = VideoInfoService()
        result = video_service.get_video_details("edge_case_test")
        
        # 빈 제목이 적절히 처리되는지 확인
        assert 'title' in result
        assert isinstance(result['title'], str)
        
        # None 설명이 빈 문자열로 변환되는지 확인
        assert 'description' in result
        assert isinstance(result['description'], str)
        assert result['description'] == ""
        
        # 태그가 빈 리스트로 처리되는지 확인
        assert 'tags' in result
        assert isinstance(result['tags'], list)
        assert len(result['tags']) == 0
        
        # 숫자 필드들이 적절히 처리되는지 확인
        if 'view_count' in result and result['view_count'] is not None:
            assert isinstance(result['view_count'], int)
            assert result['view_count'] == 0
        
        if 'like_count' in result and result['like_count'] is not None:
            # 음수는 None으로 처리되거나 적절한 기본값으로 설정될 수 있음
            assert isinstance(result['like_count'], int) or result['like_count'] is None 