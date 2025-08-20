"""
즉시 처리 시스템 테스트

채널이 와치리스트에 추가될 때 해당 채널의 최신 영상을 자동으로 처리하는 기능을 테스트합니다.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

from src.services.immediate_processing import ImmediateProcessingService
from src.services.models import ChannelInfo, VideoInfo
from src.utils.exceptions import (
    VideoProcessingError, 
    TranscriptNotFoundError,
    DatabaseError
)


@pytest.fixture
def sample_channel_info():
    """테스트용 채널 정보"""
    return ChannelInfo(
        channel_id="UC_test_channel",
        title="테스트 채널",
        description="테스트용 채널입니다",
        thumbnail_url="https://example.com/thumbnail.jpg",
        published_at="2023-01-01T00:00:00Z",
        subscriber_count=10000,
        video_count=100,
        custom_url="@testchannel",
        country="KR",
        default_language="ko"
    )


@pytest.fixture
def sample_latest_video():
    """테스트용 최신 영상 정보"""
    return VideoInfo(
        video_id="test_video_123",
        title="최신 영상",
        description="최신 영상 설명",
        channel_id="UC_test_channel",
        published_at="2023-12-01T10:00:00Z",
        thumbnail_url="https://example.com/video_thumb.jpg",
        channel_title="테스트 채널",
        duration="PT5M30S",
        view_count=1000
    )


@pytest.fixture
def sample_transcript():
    """테스트용 자막 데이터"""
    return [
        {
            'text': '안녕하세요, 오늘의 영상에 오신 것을 환영합니다.',
            'start': 0.0,
            'duration': 3.0
        },
        {
            'text': '오늘은 중요한 주제에 대해 이야기해보겠습니다.',
            'start': 3.0,
            'duration': 4.0
        },
        {
            'text': '먼저 기본 개념부터 설명드리겠습니다.',
            'start': 7.0,
            'duration': 3.5
        }
    ]


@pytest.fixture
def mock_dependencies():
    """모든 의존성 모킹"""
    return {
        'channel_repository': Mock(),
        'video_repository': Mock(),
        'video_info_service': Mock(),
        'transcript_downloader': Mock(),
        'transcript_processor': Mock(),
        'vector_store': Mock()
    }


@pytest.fixture
def processing_service(mock_dependencies):
    """즉시 처리 서비스 인스턴스"""
    return ImmediateProcessingService(
        channel_repository=mock_dependencies['channel_repository'],
        video_repository=mock_dependencies['video_repository'],
        video_info_service=mock_dependencies['video_info_service'],
        transcript_downloader=mock_dependencies['transcript_downloader'],
        transcript_processor=mock_dependencies['transcript_processor'],
        vector_store=mock_dependencies['vector_store']
    )


class TestProcessLatestVideoOnChannelAdd:
    """채널 추가 시 최신 영상 처리 테스트"""
    
    def test_process_latest_video_on_channel_add(
        self, 
        processing_service, 
        mock_dependencies,
        sample_channel_info,
        sample_latest_video,
        sample_transcript
    ):
        """채널 추가 시 최신 영상을 처리하는 테스트"""
        
        # Given: 모든 의존성이 성공적으로 동작하도록 설정
        video_info_service = mock_dependencies['video_info_service']
        transcript_downloader = mock_dependencies['transcript_downloader']
        transcript_processor = mock_dependencies['transcript_processor']
        vector_store = mock_dependencies['vector_store']
        video_repository = mock_dependencies['video_repository']
        
        # 채널의 최신 영상 조회 모킹
        video_info_service.get_channel_videos.return_value = [sample_latest_video.to_dict()]
        
        # 자막 다운로드 모킹
        transcript_downloader.download_transcript.return_value = sample_transcript
        
        # 자막 처리 모킹 (청킹된 Document 객체들)
        mock_chunks = [
            Mock(page_content="안녕하세요, 오늘의 영상에 오신 것을 환영합니다.", metadata={'start': 0.0}),
            Mock(page_content="오늘은 중요한 주제에 대해 이야기해보겠습니다.", metadata={'start': 3.0}),
            Mock(page_content="먼저 기본 개념부터 설명드리겠습니다.", metadata={'start': 7.0})
        ]
        transcript_processor.process_transcript.return_value = mock_chunks
        
        # 벡터 저장 모킹
        vector_store.add_documents.return_value = ["doc_1", "doc_2", "doc_3"]
        
        # 영상 저장 모킹
        video_repository.save_video_info.return_value = True
        video_repository.mark_video_as_processed.return_value = True
        
        # When: 채널 추가와 함께 즉시 처리 실행
        result = processing_service.process_latest_video_on_channel_add(sample_channel_info)
        
        # Then: 전체 처리 파이프라인이 실행되어야 함
        assert result is True, "채널 추가 시 최신 영상 처리가 실패했습니다"
        
        # 1. 채널의 최신 영상을 조회했는지 확인
        video_info_service.get_channel_videos.assert_called_once_with(
            channel_id=sample_channel_info.channel_id,
            max_results=1,
            order='date'
        )
        
        # 2. 영상 정보를 데이터베이스에 저장했는지 확인
        video_repository.save_video_info.assert_called_once()
        saved_video = video_repository.save_video_info.call_args[0][0]
        assert saved_video.video_id == sample_latest_video.video_id
        
        # 3. 자막을 다운로드했는지 확인
        transcript_downloader.download_transcript.assert_called_once_with(
            video_id=sample_latest_video.video_id,
            language="ko",
            fallback_languages=["en"]
        )
        
        # 4. 자막을 처리(청킹)했는지 확인
        transcript_processor.process_transcript.assert_called_once_with(
            transcript_data=sample_transcript,
            video_metadata={
                'video_id': sample_latest_video.video_id,
                'title': sample_latest_video.title,
                'channel_id': sample_latest_video.channel_id,
                'channel_title': sample_latest_video.channel_title
            }
        )
        
        # 5. 벡터 저장소에 저장했는지 확인
        vector_store.add_documents.assert_called_once_with(mock_chunks)
        
        # 6. 영상 처리 완료로 마킹했는지 확인
        video_repository.mark_video_as_processed.assert_called_once_with(
            video_id=sample_latest_video.video_id,
            processing_status='completed'
        )
    
    def test_skip_if_video_already_processed(
        self,
        processing_service,
        mock_dependencies,
        sample_channel_info,
        sample_latest_video
    ):
        """이미 처리된 영상은 스킵하는 테스트"""
        
        # Given: 이미 처리된 영상
        video_info_service = mock_dependencies['video_info_service']
        video_repository = mock_dependencies['video_repository']
        
        # 최신 영상 조회
        video_info_service.get_channel_videos.return_value = [sample_latest_video.to_dict()]
        
        # 이미 처리된 영상으로 설정
        video_repository.get_video_by_id.return_value = Mock(
            video_id=sample_latest_video.video_id,
            processing_status='completed'
        )
        
        # When: 처리 시도
        result = processing_service.process_latest_video_on_channel_add(sample_channel_info)
        
        # Then: 처리를 스킵하고 성공 반환
        assert result is True
        
        # 영상 정보 조회는 했지만, 자막 다운로드는 하지 않았어야 함
        video_info_service.get_channel_videos.assert_called_once()
        mock_dependencies['transcript_downloader'].download_transcript.assert_not_called()
        mock_dependencies['vector_store'].add_documents.assert_not_called()
    
    def test_processing_pipeline_completion(
        self,
        processing_service,
        mock_dependencies,
        sample_channel_info,
        sample_latest_video,
        sample_transcript
    ):
        """전체 처리 파이프라인 완료 확인 테스트"""
        
        # Given: 모든 단계가 성공하도록 설정
        video_info_service = mock_dependencies['video_info_service']
        transcript_downloader = mock_dependencies['transcript_downloader']
        transcript_processor = mock_dependencies['transcript_processor']
        vector_store = mock_dependencies['vector_store']
        video_repository = mock_dependencies['video_repository']
        
        # 각 단계별 모킹
        video_info_service.get_channel_videos.return_value = [sample_latest_video.to_dict()]
        transcript_downloader.download_transcript.return_value = sample_transcript
        
        mock_chunks = [Mock(page_content="test", metadata={})]
        transcript_processor.process_transcript.return_value = mock_chunks
        
        vector_store.add_documents.return_value = ["doc_1"]
        video_repository.save_video_info.return_value = True
        video_repository.mark_video_as_processed.return_value = True
        
        # When: 처리 실행
        result = processing_service.process_latest_video_on_channel_add(sample_channel_info)
        
        # Then: 모든 단계가 순차적으로 실행되어야 함
        assert result is True
        
        # 처리 순서 확인
        assert video_info_service.get_channel_videos.called
        assert transcript_downloader.download_transcript.called
        assert transcript_processor.process_transcript.called
        assert vector_store.add_documents.called
        assert video_repository.mark_video_as_processed.called
        
        # 최종 상태가 'completed'인지 확인
        mark_call_args = video_repository.mark_video_as_processed.call_args
        assert mark_call_args[1]['processing_status'] == 'completed'


class TestImmediateProcessingErrorHandling:
    """즉시 처리 시스템 에러 처리 테스트"""
    
    def test_handle_no_latest_video(
        self,
        processing_service,
        mock_dependencies,
        sample_channel_info
    ):
        """최신 영상이 없는 경우 처리 테스트"""
        
        # Given: 영상이 없는 채널
        video_info_service = mock_dependencies['video_info_service']
        video_info_service.get_channel_videos.return_value = []
        
        # When: 처리 시도
        result = processing_service.process_latest_video_on_channel_add(sample_channel_info)
        
        # Then: 에러 없이 False 반환
        assert result is False
        
        # 자막 다운로드는 시도하지 않았어야 함
        mock_dependencies['transcript_downloader'].download_transcript.assert_not_called()
    
    def test_handle_transcript_not_found(
        self,
        processing_service,
        mock_dependencies,
        sample_channel_info,
        sample_latest_video
    ):
        """자막을 찾을 수 없는 경우 처리 테스트"""
        
        # Given: 자막이 없는 영상
        video_info_service = mock_dependencies['video_info_service']
        transcript_downloader = mock_dependencies['transcript_downloader']
        
        video_info_service.get_channel_videos.return_value = [sample_latest_video.to_dict()]
        transcript_downloader.download_transcript.side_effect = TranscriptNotFoundError(
            "No transcript found for video"
        )
        
        # When: 처리 시도
        result = processing_service.process_latest_video_on_channel_add(sample_channel_info)
        
        # Then: 에러 없이 False 반환 (자막이 없는 영상은 스킵)
        assert result is False
        
        # 벡터 저장은 시도하지 않았어야 함
        mock_dependencies['vector_store'].add_documents.assert_not_called()
    
    def test_handle_processing_error(
        self,
        processing_service,
        mock_dependencies,
        sample_channel_info,
        sample_latest_video
    ):
        """처리 중 일반적인 에러 발생 시 테스트"""
        
        # Given: 벡터 저장 중 에러 발생
        video_info_service = mock_dependencies['video_info_service']
        vector_store = mock_dependencies['vector_store']
        
        video_info_service.get_channel_videos.return_value = [sample_latest_video.to_dict()]
        vector_store.add_documents.side_effect = Exception("Vector store error")
        
        # When & Then: 에러가 발생해야 함
        with pytest.raises(VideoProcessingError) as exc_info:
            processing_service.process_latest_video_on_channel_add(sample_channel_info)
        
        assert "Failed to process latest video" in str(exc_info.value) 