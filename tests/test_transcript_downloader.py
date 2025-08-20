"""
자막 다운로드 기능 테스트

YouTube 영상의 자막 다운로드 기능을 테스트합니다.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.services.transcript_downloader import TranscriptDownloader
from src.utils.exceptions import TranscriptNotFoundError, TranscriptProcessingError


def test_download_transcript_korean():
    """한국어 자막 다운로드가 올바르게 작동하는지 확인"""
    
    # Mock youtube_transcript_api 응답 데이터 (한국어 자막)
    mock_korean_transcript = [
        {
            'text': '안녕하세요, 여러분!',
            'start': 0.0,
            'duration': 2.5
        },
        {
            'text': '오늘은 YouTube API에 대해 알아보겠습니다.',
            'start': 2.5,
            'duration': 3.2
        },
        {
            'text': '먼저 API 키를 발급받아야 합니다.',
            'start': 5.7,
            'duration': 2.8
        },
        {
            'text': '그 다음 Python 라이브러리를 설치하세요.',
            'start': 8.5,
            'duration': 3.1
        },
        {
            'text': '감사합니다!',
            'start': 11.6,
            'duration': 1.4
        }
    ]
    
    with patch('src.services.transcript_downloader.YouTubeTranscriptApi') as mock_api:
        # Mock YouTube Transcript API
        mock_api.get_transcript.return_value = mock_korean_transcript
        
        # 자막 다운로더 생성
        downloader = TranscriptDownloader()
        
        # 한국어 자막 다운로드 실행
        result = downloader.download_transcript("dQw4w9WgXcQ", language="ko")
        
        # 결과 검증
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 5
        
        # 첫 번째 자막 검증
        first_transcript = result[0]
        assert first_transcript['text'] == '안녕하세요, 여러분!'
        assert first_transcript['start'] == 0.0
        assert first_transcript['duration'] == 2.5
        
        # 마지막 자막 검증
        last_transcript = result[-1]
        assert last_transcript['text'] == '감사합니다!'
        assert last_transcript['start'] == 11.6
        assert last_transcript['duration'] == 1.4
        
        # API가 올바른 매개변수로 호출되었는지 확인
        mock_api.get_transcript.assert_called_once_with("dQw4w9WgXcQ", languages=['ko'])


def test_download_transcript_korean_with_fallback():
    """한국어 자막이 없을 때 영어 자막으로 폴백하는 기능 확인"""
    
    # Mock 영어 자막 (한국어가 없는 경우)
    mock_english_transcript = [
        {
            'text': 'Hello everyone!',
            'start': 0.0,
            'duration': 2.0
        },
        {
            'text': 'Today we will learn about YouTube API.',
            'start': 2.0,
            'duration': 3.5
        },
        {
            'text': 'Thank you!',
            'start': 5.5,
            'duration': 1.5
        }
    ]
    
    with patch('src.services.transcript_downloader.YouTubeTranscriptApi') as mock_api, \
         patch('src.services.transcript_downloader.NoTranscriptFound') as mock_no_transcript:
        
        # Mock exception class
        mock_no_transcript_exception = Exception("No transcript found")
        
        # 한국어는 실패, 영어는 성공하도록 설정
        def mock_get_transcript(video_id, languages):
            if languages == ['ko']:
                raise mock_no_transcript_exception
            elif languages == ['en']:
                return mock_english_transcript
            else:
                raise Exception("Unexpected language")
        
        mock_api.get_transcript.side_effect = mock_get_transcript
        
        downloader = TranscriptDownloader()
        
        # 한국어 우선, 영어 폴백으로 다운로드
        result = downloader.download_transcript(
            "english_only_video", 
            language="ko", 
            fallback_languages=['en']
        )
        
        # 영어 자막이 반환되어야 함
        assert result is not None
        assert len(result) == 3
        assert result[0]['text'] == 'Hello everyone!'
        assert result[-1]['text'] == 'Thank you!'
        
        # API가 한국어, 영어 순으로 호출되었는지 확인
        assert mock_api.get_transcript.call_count == 2


def test_download_transcript_korean_invalid_video_id():
    """유효하지 않은 영상 ID로 자막 다운로드 시 예외 발생 확인"""
    
    with patch('src.services.transcript_downloader.YouTubeTranscriptApi') as mock_api, \
         patch('src.services.transcript_downloader.TranscriptsDisabled') as mock_disabled:
        
        # Mock exception 생성
        mock_disabled_exception = Exception("Transcripts disabled")
        mock_api.get_transcript.side_effect = mock_disabled_exception
        
        downloader = TranscriptDownloader()
        
        # TranscriptNotFoundError로 변환되어야 함
        with pytest.raises(TranscriptNotFoundError, match="Transcript not found"):
            downloader.download_transcript("invalid_video_id", language="ko")


def test_download_transcript_korean_empty_video_id():
    """빈 영상 ID로 자막 다운로드 시 예외 발생 확인"""
    
    downloader = TranscriptDownloader()
    
    # 빈 영상 ID
    with pytest.raises(ValueError, match="Video ID cannot be empty"):
        downloader.download_transcript("", language="ko")
    
    with pytest.raises(ValueError, match="Video ID cannot be empty"):
        downloader.download_transcript("   ", language="ko")
    
    with pytest.raises(ValueError, match="Video ID cannot be empty"):
        downloader.download_transcript(None, language="ko")


def test_download_transcript_korean_with_formatting():
    """한국어 자막 다운로드 시 텍스트 포맷팅 적용 확인"""
    
    # 포맷팅이 필요한 자막 데이터
    mock_unformatted_transcript = [
        {
            'text': '  안녕하세요,   여러분!  ',  # 앞뒤 공백
            'start': 0.0,
            'duration': 2.5
        },
        {
            'text': 'YouTube\n\nAPI에\t대해\r알아보겠습니다.',  # 개행, 탭 문자
            'start': 2.5,
            'duration': 3.2
        },
        {
            'text': '감사합니다!!!   ',  # 과도한 느낌표와 공백
            'start': 5.7,
            'duration': 1.8
        }
    ]
    
    with patch('src.services.transcript_downloader.YouTubeTranscriptApi') as mock_api:
        mock_api.get_transcript.return_value = mock_unformatted_transcript
        
        downloader = TranscriptDownloader()
        result = downloader.download_transcript("format_test_video", language="ko")
        
        # 포맷팅이 적용되었는지 확인
        assert result[0]['text'] == '안녕하세요, 여러분!'  # 공백 제거
        assert result[1]['text'] == 'YouTube API에 대해 알아보겠습니다.'  # 개행/탭 정리
        assert result[2]['text'] == '감사합니다!'  # 과도한 문장부호 정리


def test_download_transcript_korean_with_metadata():
    """한국어 자막 다운로드 시 메타데이터 포함 확인"""
    
    mock_transcript = [
        {
            'text': '메타데이터 테스트입니다.',
            'start': 0.0,
            'duration': 2.0
        }
    ]
    
    with patch('src.services.transcript_downloader.YouTubeTranscriptApi') as mock_api:
        mock_api.get_transcript.return_value = mock_transcript
        
        downloader = TranscriptDownloader()
        result = downloader.download_transcript(
            "metadata_test_video", 
            language="ko",
            include_metadata=True
        )
        
        # 메타데이터가 포함되어야 함
        assert isinstance(result, dict)
        assert 'transcript' in result
        assert 'metadata' in result
        
        # 자막 데이터 확인
        transcript_data = result['transcript']
        assert len(transcript_data) == 1
        assert transcript_data[0]['text'] == '메타데이터 테스트입니다.'
        
        # 메타데이터 확인
        metadata = result['metadata']
        assert 'video_id' in metadata
        assert 'language' in metadata
        assert 'download_timestamp' in metadata
        assert metadata['video_id'] == 'metadata_test_video'
        assert metadata['language'] == 'ko'


def test_download_transcript_korean_service_initialization():
    """자막 다운로더 서비스가 올바르게 초기화되는지 확인"""
    
    downloader = TranscriptDownloader()
    
    # 서비스가 올바르게 생성되었는지 확인
    assert downloader is not None
    assert isinstance(downloader, TranscriptDownloader)


def test_download_transcript_korean_large_transcript():
    """큰 용량의 한국어 자막 다운로드 처리 확인"""
    
    # 많은 자막 세그먼트 생성 (100개)
    mock_large_transcript = []
    for i in range(100):
        mock_large_transcript.append({
            'text': f'자막 세그먼트 {i+1}번입니다.',
            'start': i * 2.0,
            'duration': 1.8
        })
    
    with patch('src.services.transcript_downloader.YouTubeTranscriptApi') as mock_api:
        mock_api.get_transcript.return_value = mock_large_transcript
        
        downloader = TranscriptDownloader()
        result = downloader.download_transcript("large_transcript_video", language="ko")
        
        # 모든 세그먼트가 올바르게 처리되었는지 확인
        assert len(result) == 100
        assert result[0]['text'] == '자막 세그먼트 1번입니다.'
        assert result[99]['text'] == '자막 세그먼트 100번입니다.'
        
        # 타임스탬프가 올바른지 확인
        assert result[0]['start'] == 0.0
        assert result[99]['start'] == 198.0  # (99 * 2.0)


def test_download_transcript_korean_error_handling():
    """한국어 자막 다운로드 중 다양한 오류 상황 처리 확인"""
    
    with patch('src.services.transcript_downloader.YouTubeTranscriptApi') as mock_api:
        downloader = TranscriptDownloader()
        
        # 네트워크 오류 시뮬레이션
        mock_api.get_transcript.side_effect = Exception("Network error")
        
        with pytest.raises(TranscriptProcessingError, match="Failed to download transcript"):
            downloader.download_transcript("network_error_video", language="ko")


def test_download_transcript_korean_with_custom_config():
    """커스텀 설정으로 한국어 자막 다운로드 확인"""
    
    mock_transcript = [
        {
            'text': '커스텀 설정 테스트',
            'start': 0.0,
            'duration': 2.0
        }
    ]
    
    with patch('src.services.transcript_downloader.YouTubeTranscriptApi') as mock_api:
        mock_api.get_transcript.return_value = mock_transcript
        
        # 커스텀 설정으로 다운로더 생성
        config = {
            'preserve_formatting': False,
            'include_generated': True,
            'timeout_seconds': 30
        }
        
        downloader = TranscriptDownloader(config=config)
        result = downloader.download_transcript("custom_config_video", language="ko")
        
        assert len(result) == 1
        assert result[0]['text'] == '커스텀 설정 테스트' 