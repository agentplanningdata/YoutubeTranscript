"""
채널 검색 기능 테스트

YouTube API를 통한 채널 검색 기능을 테스트합니다.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.services.channel_search import ChannelSearchService
from src.utils.exceptions import YouTubeAPIError, ChannelNotFoundError


def test_search_channel_by_name():
    """채널명으로 검색이 올바르게 작동하는지 확인"""
    
    # Mock YouTube API 응답 데이터
    mock_api_response = [
        {
            'channel_id': 'UC_test_channel_1',
            'title': '테스트 채널',
            'description': '테스트용 채널입니다.',
            'thumbnail_url': 'https://example.com/thumbnail1.jpg',
            'published_at': '2020-01-01T00:00:00Z'
        },
        {
            'channel_id': 'UC_test_channel_2', 
            'title': '테스트 채널 2',
            'description': '두번째 테스트 채널입니다.',
            'thumbnail_url': 'https://example.com/thumbnail2.jpg',
            'published_at': '2021-01-01T00:00:00Z'
        }
    ]
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        # Mock YouTube API 클라이언트
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.search_channels.return_value = mock_api_response
        
        # 채널 검색 서비스 생성
        search_service = ChannelSearchService()
        
        # 채널 검색 실행
        results = search_service.search_channels("테스트 채널")
        
        # 결과 검증
        assert len(results) == 2
        assert results[0]['channel_id'] == 'UC_test_channel_1'
        assert results[0]['title'] == '테스트 채널'
        assert results[0]['description'] == '테스트용 채널입니다.'
        
        assert results[1]['channel_id'] == 'UC_test_channel_2'
        assert results[1]['title'] == '테스트 채널 2'
        
        # YouTube API 클라이언트가 올바른 매개변수로 호출되었는지 확인
        mock_youtube_client.search_channels.assert_called_once_with(query="테스트 채널", max_results=10)


def test_search_returns_channel_metadata():
    """채널 검색 결과가 올바른 메타데이터를 반환하는지 확인"""
    
    # 완전한 채널 메타데이터를 포함한 Mock 응답
    mock_api_response = [
        {
            'channel_id': 'UC_metadata_test_channel',
            'title': '메타데이터 테스트 채널',
            'description': '이것은 메타데이터 테스트용 채널입니다. 다양한 정보가 포함되어 있습니다.',
            'thumbnail_url': 'https://yt3.ggpht.com/test_thumbnail.jpg',
            'published_at': '2022-03-15T09:30:00Z'
        }
    ]
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.search_channels.return_value = mock_api_response
        
        search_service = ChannelSearchService()
        results = search_service.search_channels("메타데이터 테스트")
        
        # 결과가 1개인지 확인
        assert len(results) == 1
        channel = results[0]
        
        # 필수 메타데이터 필드들이 존재하는지 확인
        required_fields = [
            'channel_id', 'title', 'description', 
            'thumbnail_url', 'published_at'
        ]
        
        for field in required_fields:
            assert field in channel, f"Required field '{field}' is missing from channel metadata"
            assert channel[field] is not None, f"Field '{field}' should not be None"
            assert channel[field] != "", f"Field '{field}' should not be empty"
        
        # 각 필드의 타입과 내용 검증
        assert isinstance(channel['channel_id'], str)
        assert channel['channel_id'].startswith('UC'), "Channel ID should start with 'UC'"
        assert len(channel['channel_id']) >= 20, "Channel ID should be at least 20 characters"
        
        assert isinstance(channel['title'], str)
        assert len(channel['title']) > 0, "Title should not be empty"
        
        assert isinstance(channel['description'], str)
        
        assert isinstance(channel['thumbnail_url'], str)
        assert channel['thumbnail_url'].startswith('https://'), "Thumbnail URL should start with https://"
        
        assert isinstance(channel['published_at'], str)
        # ISO 8601 날짜 형식 검증
        datetime.fromisoformat(channel['published_at'].replace('Z', '+00:00'))  # 유효한 날짜인지 확인
        
        # 특정 값들 검증
        assert channel['channel_id'] == 'UC_metadata_test_channel'
        assert channel['title'] == '메타데이터 테스트 채널'
        assert '메타데이터 테스트용' in channel['description']
        assert 'yt3.ggpht.com' in channel['thumbnail_url']
        assert channel['published_at'] == '2022-03-15T09:30:00Z'


def test_search_returns_multiple_channels_with_metadata():
    """여러 채널 검색 시 모든 채널이 완전한 메타데이터를 가지는지 확인"""
    
    mock_api_response = [
        {
            'channel_id': 'UC1234567890abcdefghijklmnopqrstuv',
            'title': '첫 번째 채널',
            'description': '첫 번째 채널 설명입니다.',
            'thumbnail_url': 'https://yt3.ggpht.com/channel1.jpg',
            'published_at': '2020-01-01T00:00:00Z'
        },
        {
            'channel_id': 'UCabcdefghij1234567890klmnopqrstuv',
            'title': '두 번째 채널',
            'description': '두 번째 채널 설명입니다.',
            'thumbnail_url': 'https://yt3.ggpht.com/channel2.jpg',
            'published_at': '2021-06-15T12:30:00Z'
        },
        {
            'channel_id': 'UCxyzabcd1234efgh5678ijklmnopqrst',
            'title': '세 번째 채널',
            'description': '',  # 빈 설명 (허용되어야 함)
            'thumbnail_url': 'https://yt3.ggpht.com/channel3.jpg',
            'published_at': '2023-12-01T18:45:30Z'
        }
    ]
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.search_channels.return_value = mock_api_response
        
        search_service = ChannelSearchService()
        results = search_service.search_channels("테스트 쿼리")
        
        assert len(results) == 3
        
        # 모든 채널이 필수 필드를 가지고 있는지 확인
        required_fields = ['channel_id', 'title', 'description', 'thumbnail_url', 'published_at']
        
        for i, channel in enumerate(results):
            for field in required_fields:
                assert field in channel, f"Channel {i} missing field '{field}'"
                
                # description은 빈 문자열이 허용됨
                if field == 'description':
                    assert isinstance(channel[field], str)
                else:
                    assert channel[field] is not None and channel[field] != ""
            
            # 채널 ID 형식 검증
            assert channel['channel_id'].startswith('UC')
            assert len(channel['channel_id']) >= 20
            
            # 썸네일 URL 검증
            assert channel['thumbnail_url'].startswith('https://')
            
            # 날짜 형식 검증
            datetime.fromisoformat(channel['published_at'].replace('Z', '+00:00'))
        
        # 각 채널의 고유성 확인
        channel_ids = [channel['channel_id'] for channel in results]
        assert len(set(channel_ids)) == len(channel_ids), "All channel IDs should be unique"


def test_search_handles_missing_optional_metadata():
    """선택적 메타데이터가 누락되었을 때 적절히 처리하는지 확인"""
    
    # 일부 메타데이터가 누락된 응답
    mock_api_response = [
        {
            'channel_id': 'UC_minimal_channel',
            'title': '미니멀 채널',
            'description': None,  # None 값
            'thumbnail_url': 'https://yt3.ggpht.com/minimal.jpg',
            'published_at': '2023-01-01T00:00:00Z'
        }
    ]
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.search_channels.return_value = mock_api_response
        
        search_service = ChannelSearchService()
        results = search_service.search_channels("미니멀")
        
        assert len(results) == 1
        channel = results[0]
        
        # 기본 필드들은 존재해야 함
        assert 'channel_id' in channel
        assert 'title' in channel
        assert 'thumbnail_url' in channel
        assert 'published_at' in channel
        
        # None 값도 허용되어야 함 (YouTube API에서 가끔 None을 반환할 수 있음)
        assert 'description' in channel
        # description이 None이어도 서비스에서 빈 문자열로 변환할 수 있음


def test_search_nonexistent_channel():
    """존재하지 않는 채널을 검색할 때 적절히 처리하는지 확인"""
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        # 검색 결과 없음
        mock_youtube_client.search_channels.return_value = []
        
        search_service = ChannelSearchService()
        
        # 존재하지 않는 채널 검색
        results = search_service.search_channels("절대존재하지않는채널명12345")
        
        # 빈 리스트 반환되어야 함
        assert results == []
        assert len(results) == 0
        assert isinstance(results, list)


def test_search_nonexistent_channel_with_typo():
    """오타가 있는 채널명으로 검색할 때 처리 확인"""
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        # 오타로 인한 검색 결과 없음
        mock_youtube_client.search_channels.return_value = []
        
        search_service = ChannelSearchService()
        
        # 일반적인 채널명의 오타 버전들
        typo_queries = [
            "유투브",  # 유튜브 오타
            "네이버tv",  # 네이버TV 오타
            "카카오톡",  # 카카오 오타
            "구글tv",   # 구글 TV 오타
        ]
        
        for query in typo_queries:
            results = search_service.search_channels(query)
            
            assert results == []
            assert len(results) == 0
            
            # API가 올바른 쿼리로 호출되었는지 확인
            mock_youtube_client.search_channels.assert_called_with(
                query=query, max_results=10
            )


def test_search_very_specific_nonexistent_channel():
    """매우 구체적이지만 존재하지 않는 채널 검색 처리"""
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        # 매우 구체적인 검색이지만 결과 없음
        mock_youtube_client.search_channels.return_value = []
        
        search_service = ChannelSearchService()
        
        # 매우 구체적이지만 존재하지 않는 채널명들
        specific_queries = [
            "김철수의 요리 채널 2023년 버전",
            "서울 강남 맛집 탐방 채널 공식",
            "파이썬 프로그래밍 튜토리얼 전문 채널",
            "고양이 브이로그 일상 기록 채널"
        ]
        
        for query in specific_queries:
            results = search_service.search_channels(query)
            
            assert results == []
            assert len(results) == 0
            assert isinstance(results, list)


def test_search_nonexistent_channel_with_special_characters():
    """특수문자가 포함된 존재하지 않는 채널명 검색"""
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        mock_youtube_client.search_channels.return_value = []
        
        search_service = ChannelSearchService()
        
        # 특수문자가 포함된 검색어들
        special_char_queries = [
            "채널@#$%^&*()",
            "테스트!@#채널",
            "한글-English-123",
            "채널 이름 (2023) [공식]",
            "음악♪♫♪채널"
        ]
        
        for query in special_char_queries:
            results = search_service.search_channels(query)
            
            assert results == []
            assert len(results) == 0
            
            # YouTube API 클라이언트가 특수문자도 올바르게 전달하는지 확인
            mock_youtube_client.search_channels.assert_called_with(
                query=query, max_results=10
            )


def test_search_nonexistent_channel_different_languages():
    """다양한 언어의 존재하지 않는 채널명 검색"""
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        mock_youtube_client.search_channels.return_value = []
        
        search_service = ChannelSearchService()
        
        # 다양한 언어의 검색어들
        language_queries = [
            "존재하지않는한글채널",     # 한글
            "NonexistentEnglishChannel",  # 영어
            "不存在的中文频道",           # 중국어 간체
            "存在しない日本語チャンネル",    # 일본어
            "CanaleItalianoInesistente", # 이탈리아어
        ]
        
        for query in language_queries:
            results = search_service.search_channels(query)
            
            assert results == []
            assert len(results) == 0
            assert isinstance(results, list)


def test_search_handles_network_timeout_like_no_results():
    """네트워크 타임아웃과 유사한 상황에서 결과 없음 처리"""
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        # 타임아웃으로 인한 빈 결과 (실제로는 YouTubeAPIError가 발생할 수 있지만,
        # 이 테스트에서는 빈 결과 반환 상황을 시뮬레이션)
        mock_youtube_client.search_channels.return_value = []
        
        search_service = ChannelSearchService()
        results = search_service.search_channels("일반적인채널명")
        
        assert results == []
        assert len(results) == 0


def test_find_exact_channel_match_nonexistent():
    """정확 일치 검색에서 존재하지 않는 채널 처리"""
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        # 검색 결과 없음
        mock_youtube_client.search_channels.return_value = []
        
        search_service = ChannelSearchService()
        result = search_service.find_exact_channel_match("존재하지않는정확한채널명")
        
        # None이 반환되어야 함
        assert result is None


def test_find_exact_channel_match_partial_matches_only():
    """정확 일치 검색에서 부분 일치만 있고 정확 일치 없는 경우"""
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        # 부분 일치 결과들만 있음
        mock_api_response = [
            {
                'channel_id': 'UC1234567890abcdefghijklmnopqrstuv',
                'title': '테스트 채널 공식',
                'description': '테스트 채널 공식 계정입니다.',
                'thumbnail_url': 'https://yt3.ggpht.com/test1.jpg',
                'published_at': '2020-01-01T00:00:00Z'
            },
            {
                'channel_id': 'UCabcdefghij1234567890klmnopqrstuv',
                'title': '테스트 채널 비공식',
                'description': '테스트 채널 팬이 운영합니다.',
                'thumbnail_url': 'https://yt3.ggpht.com/test2.jpg',
                'published_at': '2021-01-01T00:00:00Z'
            }
        ]
        
        mock_youtube_client.search_channels.return_value = mock_api_response
        
        search_service = ChannelSearchService()
        
        # "테스트 채널" 검색 (정확히 일치하는 것 없음)
        result = search_service.find_exact_channel_match("테스트 채널")
        
        # 정확 일치가 없으므로 None 반환
        assert result is None


def test_search_result_pagination():
    """검색 결과 페이징이 올바르게 처리되는지 확인"""
    
    # 많은 결과를 시뮬레이션하기 위한 Mock 데이터
    def create_mock_channel(i):
        return {
            'channel_id': f'UC{str(i).zfill(22)}abcdefghijklmnopqr',
            'title': f'테스트 채널 {i}',
            'description': f'테스트 채널 {i}의 설명입니다.',
            'thumbnail_url': f'https://yt3.ggpht.com/channel_{i}.jpg',
            'published_at': '2023-01-01T00:00:00Z'
        }
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        search_service = ChannelSearchService()
        
        # Case 1: max_results=5로 제한
        mock_5_results = [create_mock_channel(i) for i in range(1, 6)]  # 5개
        mock_youtube_client.search_channels.return_value = mock_5_results
        
        results = search_service.search_channels("테스트", max_results=5)
        
        assert len(results) == 5
        assert results[0]['title'] == '테스트 채널 1'
        assert results[4]['title'] == '테스트 채널 5'
        
        # API가 올바른 max_results로 호출되었는지 확인
        mock_youtube_client.search_channels.assert_called_with(
            query="테스트", max_results=5
        )


def test_search_result_pagination_large_numbers():
    """큰 수의 max_results에 대한 페이징 처리"""
    
    def create_mock_channel(i):
        return {
            'channel_id': f'UC{str(i).zfill(22)}abcdefghijklmnopqr',
            'title': f'대용량 채널 {i}',
            'description': f'대용량 테스트 채널 {i}입니다.',
            'thumbnail_url': f'https://yt3.ggpht.com/large_{i}.jpg',
            'published_at': '2023-01-01T00:00:00Z'
        }
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        search_service = ChannelSearchService()
        
        # Case: max_results=50 (큰 수)
        mock_50_results = [create_mock_channel(i) for i in range(1, 51)]  # 50개
        mock_youtube_client.search_channels.return_value = mock_50_results
        
        results = search_service.search_channels("대용량", max_results=50)
        
        assert len(results) == 50
        assert results[0]['title'] == '대용량 채널 1'
        assert results[49]['title'] == '대용량 채널 50'
        
        # API가 올바른 max_results로 호출되었는지 확인
        mock_youtube_client.search_channels.assert_called_with(
            query="대용량", max_results=50
        )


def test_search_result_pagination_zero_and_negative():
    """max_results가 0이나 음수일 때의 처리"""
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        search_service = ChannelSearchService()
        
        # Case 1: max_results=0
        mock_youtube_client.search_channels.return_value = []
        results = search_service.search_channels("테스트", max_results=0)
        
        assert results == []
        assert len(results) == 0
        
        # API가 0으로 호출되었는지 확인
        mock_youtube_client.search_channels.assert_called_with(
            query="테스트", max_results=0
        )
        
        # Case 2: max_results=-1 (음수)
        # 음수는 YouTube API에서 오류를 발생시킬 수 있으므로,
        # 서비스에서 기본값으로 변환하거나 예외를 발생시켜야 함
        try:
            results = search_service.search_channels("테스트", max_results=-1)
            # 만약 서비스에서 음수를 허용한다면 API 호출 확인
            mock_youtube_client.search_channels.assert_called_with(
                query="테스트", max_results=-1
            )
        except (ValueError, YouTubeAPIError):
            # 예외가 발생하는 것도 올바른 처리
            pass


def test_search_result_pagination_boundary_values():
    """경계값들에 대한 페이징 처리 (1, 2, 최대값 근처)"""
    
    def create_mock_channel(i):
        return {
            'channel_id': f'UC{str(i).zfill(22)}boundary_test_value',
            'title': f'경계값 채널 {i}',
            'description': f'경계값 테스트 채널 {i}입니다.',
            'thumbnail_url': f'https://yt3.ggpht.com/boundary_{i}.jpg',
            'published_at': '2023-01-01T00:00:00Z'
        }
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        search_service = ChannelSearchService()
        
        # Case 1: max_results=1
        mock_1_result = [create_mock_channel(1)]
        mock_youtube_client.search_channels.return_value = mock_1_result
        
        results = search_service.search_channels("경계값", max_results=1)
        
        assert len(results) == 1
        assert results[0]['title'] == '경계값 채널 1'
        
        # Case 2: max_results=2
        mock_2_results = [create_mock_channel(i) for i in range(1, 3)]
        mock_youtube_client.search_channels.return_value = mock_2_results
        
        results = search_service.search_channels("경계값", max_results=2)
        
        assert len(results) == 2
        assert results[0]['title'] == '경계값 채널 1'
        assert results[1]['title'] == '경계값 채널 2'


def test_search_result_pagination_actual_less_than_requested():
    """요청한 것보다 실제 결과가 적을 때 처리"""
    
    def create_mock_channel(i):
        return {
            'channel_id': f'UC{str(i).zfill(22)}fewer_results_test',
            'title': f'적은 결과 채널 {i}',
            'description': f'실제 결과가 적은 경우 테스트 {i}',
            'thumbnail_url': f'https://yt3.ggpht.com/fewer_{i}.jpg',
            'published_at': '2023-01-01T00:00:00Z'
        }
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        search_service = ChannelSearchService()
        
        # max_results=10 요청했지만 실제로는 3개만 반환됨
        mock_3_results = [create_mock_channel(i) for i in range(1, 4)]  # 3개만
        mock_youtube_client.search_channels.return_value = mock_3_results
        
        results = search_service.search_channels("적은결과", max_results=10)
        
        # 실제 결과 수만큼 반환되어야 함 (3개)
        assert len(results) == 3
        assert results[0]['title'] == '적은 결과 채널 1'
        assert results[2]['title'] == '적은 결과 채널 3'
        
        # API에는 원래 요청한 max_results로 호출되어야 함
        mock_youtube_client.search_channels.assert_called_with(
            query="적은결과", max_results=10
        )


def test_search_result_pagination_with_detailed_search():
    """상세 검색에서의 페이징 처리"""
    
    def create_mock_channel_basic(i):
        return {
            'channel_id': f'UC{str(i).zfill(22)}detailed_basic_info',
            'title': f'상세검색 채널 {i}',
            'description': f'상세검색 기본 정보 {i}',
            'thumbnail_url': f'https://yt3.ggpht.com/detailed_{i}.jpg',
            'published_at': '2023-01-01T00:00:00Z'
        }
    
    def create_mock_channel_detailed(i):
        return {
            'channel_id': f'UC{str(i).zfill(22)}detailed_basic_info',
            'title': f'상세검색 채널 {i}',
            'description': f'상세검색 기본 정보 {i}',
            'thumbnail_url': f'https://yt3.ggpht.com/detailed_{i}.jpg',
            'published_at': '2023-01-01T00:00:00Z',
            'subscriber_count': 10000 + i * 1000,
            'video_count': 100 + i * 10,
            'view_count': 1000000 + i * 100000
        }
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        # 기본 검색 결과
        mock_basic_results = [create_mock_channel_basic(i) for i in range(1, 4)]
        mock_youtube_client.search_channels.return_value = mock_basic_results
        
        # 상세 정보
        def mock_get_channel_details(channel_id):
            if 'detailed_basic_info' in channel_id:
                # 채널 ID에서 숫자 추출 (예: UC0000000000000000000001detailed_basic_info -> 1)
                if '1detailed_basic_info' in channel_id:
                    return create_mock_channel_detailed(1)
                elif '2detailed_basic_info' in channel_id:
                    return create_mock_channel_detailed(2)
                elif '3detailed_basic_info' in channel_id:
                    return create_mock_channel_detailed(3)
            return {}
        
        mock_youtube_client.get_channel_details.side_effect = mock_get_channel_details
        
        search_service = ChannelSearchService()
        
        # 상세 정보 포함 검색 (max_results=3)
        results = search_service.search_channels_with_details("상세검색", max_results=3)
        
        assert len(results) == 3
        
        # 상세 정보가 포함되어 있는지 확인
        for i, channel in enumerate(results, 1):
            assert 'subscriber_count' in channel
            assert 'video_count' in channel
            assert 'view_count' in channel
            assert channel['subscriber_count'] == 10000 + i * 1000


def test_search_result_pagination_default_behavior():
    """기본 페이징 동작 확인 (max_results 미지정시)"""
    
    def create_mock_channel(i):
        return {
            'channel_id': f'UC{str(i).zfill(22)}default_pagination',
            'title': f'기본 페이징 채널 {i}',
            'description': f'기본 페이징 테스트 {i}',
            'thumbnail_url': f'https://yt3.ggpht.com/default_{i}.jpg',
            'published_at': '2023-01-01T00:00:00Z'
        }
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        # 기본값인 10개를 시뮬레이션
        mock_10_results = [create_mock_channel(i) for i in range(1, 11)]
        mock_youtube_client.search_channels.return_value = mock_10_results
        
        search_service = ChannelSearchService()
        
        # max_results 지정하지 않음 (기본값 사용)
        results = search_service.search_channels("기본페이징")
        
        # 기본값 10개가 반환되어야 함
        assert len(results) == 10
        assert results[0]['title'] == '기본 페이징 채널 1'
        assert results[9]['title'] == '기본 페이징 채널 10'
        
        # API 호출 시 기본값 10이 사용되었는지 확인
        mock_youtube_client.search_channels.assert_called_with(
            query="기본페이징", max_results=10
        )


def test_search_channel_by_name_with_custom_max_results():
    """최대 결과 수를 지정해서 채널 검색이 작동하는지 확인"""
    
    mock_api_response = [
        {
            'channel_id': 'UC_test_channel_1',
            'title': '테스트 채널',
            'description': '테스트용 채널입니다.',
            'thumbnail_url': 'https://example.com/thumbnail1.jpg',
            'published_at': '2020-01-01T00:00:00Z'
        }
    ]
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.search_channels.return_value = mock_api_response
        
        search_service = ChannelSearchService()
        
        # 최대 결과 수를 5로 지정해서 검색
        results = search_service.search_channels("테스트 채널", max_results=5)
        
        assert len(results) == 1
        assert results[0]['channel_id'] == 'UC_test_channel_1'
        
        # 지정한 max_results로 API가 호출되었는지 확인
        mock_youtube_client.search_channels.assert_called_once_with(query="테스트 채널", max_results=5)


def test_search_channel_with_empty_query():
    """빈 검색어로 검색 시 적절한 예외 발생하는지 확인"""
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        search_service = ChannelSearchService()
        
        # 빈 검색어로 검색 시 ValueError 발생해야 함
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            search_service.search_channels("")
        
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            search_service.search_channels("   ")  # 공백만 있는 경우
        
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            search_service.search_channels(None)


def test_search_channel_no_results():
    """검색 결과가 없을 때 빈 리스트 반환하는지 확인"""
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.search_channels.return_value = []  # 빈 결과
        
        search_service = ChannelSearchService()
        
        results = search_service.search_channels("존재하지않는채널명")
        
        assert results == []
        assert len(results) == 0


def test_search_channel_api_error_handling():
    """YouTube API 에러 시 적절한 예외 처리가 되는지 확인"""
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        # YouTube API에서 에러 발생 시뮬레이션
        mock_youtube_client.search_channels.side_effect = YouTubeAPIError("API Error")
        
        search_service = ChannelSearchService()
        
        # YouTubeAPIError가 그대로 전파되어야 함
        with pytest.raises(YouTubeAPIError, match="API Error"):
            search_service.search_channels("테스트 채널")


def test_search_channel_service_initialization():
    """채널 검색 서비스가 올바르게 초기화되는지 확인"""
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        search_service = ChannelSearchService()
        
        # 서비스가 올바르게 생성되었는지 확인
        assert search_service is not None
        assert isinstance(search_service, ChannelSearchService)
        
        # YouTube API 클라이언트가 초기화되었는지 확인
        mock_client.assert_called_once()


def test_search_channel_with_api_key():
    """API 키를 지정해서 채널 검색 서비스를 생성할 수 있는지 확인"""
    
    api_key = "test_api_key_123"
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        
        search_service = ChannelSearchService(api_key=api_key)
        
        # 지정한 API 키로 YouTube 클라이언트가 생성되었는지 확인
        mock_client.assert_called_once_with(api_key=api_key)


def test_search_channel_default_parameters():
    """채널 검색 서비스의 기본 매개변수가 올바른지 확인"""
    
    mock_api_response = [
        {
            'channel_id': 'UC_test_channel_1',
            'title': '테스트 채널',
            'description': '테스트용 채널입니다.',
            'thumbnail_url': 'https://example.com/thumbnail1.jpg',
            'published_at': '2020-01-01T00:00:00Z'
        }
    ]
    
    with patch('src.services.channel_search.YouTubeAPIClient') as mock_client:
        mock_youtube_client = MagicMock()
        mock_client.return_value = mock_youtube_client
        mock_youtube_client.search_channels.return_value = mock_api_response
        
        search_service = ChannelSearchService()
        
        # 기본 매개변수로 검색 (max_results 지정 안함)
        results = search_service.search_channels("테스트 채널")
        
        # 기본값 10으로 API가 호출되어야 함
        mock_youtube_client.search_channels.assert_called_once_with(query="테스트 채널", max_results=10) 