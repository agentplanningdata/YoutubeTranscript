"""
주기적 모니터링 시스템 테스트

와치리스트에 있는 채널들을 주기적으로 확인하여 새 영상을 탐지하고 처리하는 기능을 테스트합니다.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.services.periodic_monitoring import PeriodicMonitoringService
from src.services.models import ChannelInfo, VideoInfo
from src.utils.exceptions import (
    VideoProcessingError, 
    YouTubeAPIError,
    DatabaseError
)


@pytest.fixture
def sample_channel_infos():
    """테스트용 채널 정보 리스트"""
    return [
        ChannelInfo(
            channel_id="UC_channel_1",
            title="테스트 채널 1",
            description="첫 번째 테스트 채널",
            thumbnail_url="https://example.com/thumb1.jpg",
            published_at="2023-01-01T00:00:00Z",
            subscriber_count=50000,
            video_count=200
        ),
        ChannelInfo(
            channel_id="UC_channel_2", 
            title="테스트 채널 2",
            description="두 번째 테스트 채널",
            thumbnail_url="https://example.com/thumb2.jpg",
            published_at="2023-01-01T00:00:00Z",
            subscriber_count=30000,
            video_count=150
        )
    ]


@pytest.fixture
def sample_new_videos():
    """테스트용 새 영상 리스트"""
    return [
        VideoInfo(
            video_id="new_video_1",
            title="새 영상 1",
            description="첫 번째 새 영상",
            channel_id="UC_channel_1",
            published_at="2023-12-15T10:00:00Z",
            thumbnail_url="https://example.com/video1.jpg",
            channel_title="테스트 채널 1"
        ),
        VideoInfo(
            video_id="new_video_2", 
            title="새 영상 2",
            description="두 번째 새 영상",
            channel_id="UC_channel_2",
            published_at="2023-12-15T11:00:00Z",
            thumbnail_url="https://example.com/video2.jpg",
            channel_title="테스트 채널 2"
        )
    ]


@pytest.fixture
def mock_dependencies():
    """모든 의존성 모킹"""
    return {
        'channel_repository': Mock(),
        'video_repository': Mock(),
        'video_info_service': Mock(),
        'immediate_processing_service': Mock(),
        'notification_service': Mock()
    }


@pytest.fixture
def monitoring_service(mock_dependencies):
    """주기적 모니터링 서비스 인스턴스"""
    return PeriodicMonitoringService(
        channel_repository=mock_dependencies['channel_repository'],
        video_repository=mock_dependencies['video_repository'],
        video_info_service=mock_dependencies['video_info_service'],
        immediate_processing_service=mock_dependencies['immediate_processing_service'],
        notification_service=mock_dependencies['notification_service']
    )


class TestScheduledChannelCheck:
    """스케줄된 채널 확인 테스트"""
    
    def test_scheduled_channel_check(
        self,
        monitoring_service,
        mock_dependencies,
        sample_channel_infos,
        sample_new_videos
    ):
        """스케줄된 채널 확인 기본 테스트"""
        
        # Given: 와치리스트에 채널들이 있고, 각 채널에 새 영상이 있음
        channel_repository = mock_dependencies['channel_repository']
        video_info_service = mock_dependencies['video_info_service']
        immediate_processing_service = mock_dependencies['immediate_processing_service']
        
        # 와치리스트 채널 목록 반환
        channel_repository.get_watchlist_channels.return_value = sample_channel_infos
        
        # 각 채널의 최신 영상 반환 (첫 번째 채널은 새 영상, 두 번째는 영상 없음)
        def mock_get_channel_videos(channel_id, max_results=5, order='date', **kwargs):
            if channel_id == "UC_channel_1":
                return [sample_new_videos[0].to_dict()]
            elif channel_id == "UC_channel_2":
                return [sample_new_videos[1].to_dict()]
            return []
        
        video_info_service.get_channel_videos.side_effect = mock_get_channel_videos
        
        # 즉시 처리 서비스 성공 모킹
        immediate_processing_service.process_video.return_value = True
        
        # When: 스케줄된 채널 확인 실행
        result = monitoring_service.check_all_channels()
        
        # Then: 모든 채널을 확인하고 새 영상을 처리해야 함
        assert result.total_channels_checked == 2
        assert result.new_videos_found == 2
        assert result.videos_processed_successfully == 2
        assert result.errors == []
        
        # 와치리스트 조회 확인
        channel_repository.get_watchlist_channels.assert_called_once()
        
        # 각 채널의 최신 영상 조회 확인
        assert video_info_service.get_channel_videos.call_count == 2
        
        # 새 영상 처리 확인
        assert immediate_processing_service.process_video.call_count == 2
        
        # 첫 번째 채널 영상 처리 확인
        first_call = immediate_processing_service.process_video.call_args_list[0]
        assert first_call[0][0].video_id == "new_video_1"
        
        # 두 번째 채널 영상 처리 확인
        second_call = immediate_processing_service.process_video.call_args_list[1]
        assert second_call[0][0].video_id == "new_video_2"
    
    def test_check_channels_with_no_new_videos(
        self,
        monitoring_service,
        mock_dependencies,
        sample_channel_infos
    ):
        """새 영상이 없는 채널들 확인 테스트"""
        
        # Given: 와치리스트 채널들이 있지만 새 영상이 없음
        channel_repository = mock_dependencies['channel_repository']
        video_info_service = mock_dependencies['video_info_service']
        immediate_processing_service = mock_dependencies['immediate_processing_service']
        
        channel_repository.get_watchlist_channels.return_value = sample_channel_infos
        
        # 모든 채널에서 영상 없음
        video_info_service.get_channel_videos.return_value = []
        
        # When: 채널 확인 실행
        result = monitoring_service.check_all_channels()
        
        # Then: 채널은 확인했지만 새 영상이 없어야 함
        assert result.total_channels_checked == 2
        assert result.new_videos_found == 0
        assert result.videos_processed_successfully == 0
        assert result.errors == []
        
        # 즉시 처리는 호출되지 않았어야 함
        immediate_processing_service.process_video.assert_not_called()
    
    def test_check_channels_with_already_processed_videos(
        self,
        monitoring_service,
        mock_dependencies,
        sample_channel_infos,
        sample_new_videos
    ):
        """이미 처리된 영상들이 있는 경우 테스트"""
        
        # Given: 새 영상들이 있지만 이미 처리된 상태
        channel_repository = mock_dependencies['channel_repository']
        video_info_service = mock_dependencies['video_info_service']
        video_repository = mock_dependencies['video_repository']
        immediate_processing_service = mock_dependencies['immediate_processing_service']
        
        channel_repository.get_watchlist_channels.return_value = sample_channel_infos
        
        # 첫 번째 채널에만 새 영상 있음
        def mock_get_channel_videos(channel_id, max_results=5, order='date', **kwargs):
            if channel_id == "UC_channel_1":
                return [sample_new_videos[0].to_dict()]
            return []
        video_info_service.get_channel_videos.side_effect = mock_get_channel_videos
        
        # 영상이 이미 처리된 상태로 설정
        mock_existing_video = Mock(
            video_id="new_video_1",
            processing_status="completed"
        )
        video_repository.get_video_by_id.return_value = mock_existing_video
        
        # When: 채널 확인 실행
        result = monitoring_service.check_all_channels()
        
        # Then: 새 영상을 발견했지만 이미 처리된 것으로 스킵
        assert result.total_channels_checked == 2
        assert result.new_videos_found == 1
        assert result.videos_processed_successfully == 0  # 이미 처리되어 스킵
        assert result.videos_skipped == 1
        
        # 즉시 처리는 호출되지 않았어야 함
        immediate_processing_service.process_video.assert_not_called()
    
    def test_handle_channel_api_error(
        self,
        monitoring_service,
        mock_dependencies,
        sample_channel_infos
    ):
        """채널 API 오류 처리 테스트"""
        
        # Given: 첫 번째 채널에서 API 오류 발생
        channel_repository = mock_dependencies['channel_repository']
        video_info_service = mock_dependencies['video_info_service']
        immediate_processing_service = mock_dependencies['immediate_processing_service']
        
        channel_repository.get_watchlist_channels.return_value = sample_channel_infos
        
        # 첫 번째 채널에서 API 오류, 두 번째는 정상
        def mock_get_channel_videos(channel_id, max_results=5, order='date', **kwargs):
            if channel_id == "UC_channel_1":
                raise YouTubeAPIError("API quota exceeded")
            elif channel_id == "UC_channel_2":
                return []  # 영상 없음
            return []
        
        video_info_service.get_channel_videos.side_effect = mock_get_channel_videos
        
        # When: 채널 확인 실행
        result = monitoring_service.check_all_channels()
        
        # Then: 오류가 있어도 다른 채널은 계속 처리
        assert result.total_channels_checked == 2
        assert result.new_videos_found == 0
        assert result.videos_processed_successfully == 0
        assert len(result.errors) == 1
        assert "UC_channel_1" in result.errors[0]
        assert "API quota exceeded" in result.errors[0]
        
        # 즉시 처리는 호출되지 않았어야 함
        immediate_processing_service.process_video.assert_not_called()


class TestNewVideoDetection:
    """새 영상 탐지 테스트"""
    
    def test_new_video_detection(
        self,
        monitoring_service,
        mock_dependencies,
        sample_channel_infos
    ):
        """새 영상 탐지 기본 테스트"""
        
        # Given: 특정 채널과 마지막 확인 시간
        channel_info = sample_channel_infos[0]
        last_check_time = datetime.now() - timedelta(hours=1)
        
        video_info_service = mock_dependencies['video_info_service']
        video_repository = mock_dependencies['video_repository']
        
        # 새로운 영상들 반환 (마지막 확인 이후)
        new_videos = [
            {
                'video_id': 'new_video_123',
                'title': '새 영상',
                'published_at': (datetime.now() - timedelta(minutes=30)).isoformat() + 'Z',
                'channel_id': channel_info.channel_id,
                'description': '새로 업로드된 영상',
                'thumbnail_url': 'https://example.com/new_video.jpg'
            }
        ]
        video_info_service.get_channel_videos.return_value = new_videos
        
        # 해당 영상이 DB에 없음 (새 영상)
        video_repository.get_video_by_id.return_value = None
        
        # When: 새 영상 탐지
        detected_videos = monitoring_service.detect_new_videos(channel_info, last_check_time)
        
        # Then: 새 영상이 탐지되어야 함
        assert len(detected_videos) == 1
        assert detected_videos[0].video_id == 'new_video_123'
        assert detected_videos[0].title == '새 영상'
        
        # 채널 영상 조회 확인
        video_info_service.get_channel_videos.assert_called_once()
        call_args = video_info_service.get_channel_videos.call_args
        assert call_args[1]['published_after'] == last_check_time
    
    def test_no_new_videos_since_last_check(
        self,
        monitoring_service,
        mock_dependencies,
        sample_channel_infos
    ):
        """마지막 확인 이후 새 영상이 없는 경우 테스트"""
        
        # Given: 마지막 확인 시간 이후 영상이 없음
        channel_info = sample_channel_infos[0]
        last_check_time = datetime.now() - timedelta(hours=1)
        
        video_info_service = mock_dependencies['video_info_service']
        
        # 영상이 없음
        video_info_service.get_channel_videos.return_value = []
        
        # When: 새 영상 탐지
        detected_videos = monitoring_service.detect_new_videos(channel_info, last_check_time)
        
        # Then: 새 영상이 없어야 함
        assert len(detected_videos) == 0


class TestMonitoringIntervalConfiguration:
    """모니터링 간격 설정 테스트"""
    
    def test_monitoring_interval_configuration(
        self,
        mock_dependencies
    ):
        """모니터링 간격 설정 테스트"""
        
        # Given: 사용자 정의 모니터링 설정
        from src.services.periodic_monitoring import MonitoringConfig
        
        custom_config = MonitoringConfig(
            check_interval_minutes=15,  # 15분 간격
            max_videos_per_channel=3,
            process_immediately=True,
            notification_enabled=True,
            error_retry_count=5
        )
        
        # When: 커스텀 설정으로 모니터링 서비스 생성
        monitoring_service = PeriodicMonitoringService(
            channel_repository=mock_dependencies['channel_repository'],
            video_repository=mock_dependencies['video_repository'],
            video_info_service=mock_dependencies['video_info_service'],
            immediate_processing_service=mock_dependencies['immediate_processing_service'],
            notification_service=mock_dependencies['notification_service'],
            config=custom_config
        )
        
        # Then: 설정이 올바르게 적용되어야 함
        assert monitoring_service.config.check_interval_minutes == 15
        assert monitoring_service.config.max_videos_per_channel == 3
        assert monitoring_service.config.process_immediately is True
        assert monitoring_service.config.notification_enabled is True
        assert monitoring_service.config.error_retry_count == 5
    
    def test_default_monitoring_configuration(
        self,
        mock_dependencies
    ):
        """기본 모니터링 설정 테스트"""
        
        # Given & When: 설정 없이 모니터링 서비스 생성
        monitoring_service = PeriodicMonitoringService(
            channel_repository=mock_dependencies['channel_repository'],
            video_repository=mock_dependencies['video_repository'],
            video_info_service=mock_dependencies['video_info_service'],
            immediate_processing_service=mock_dependencies['immediate_processing_service'],
            notification_service=mock_dependencies['notification_service']
            # config 없음 - 기본값 사용
        )
        
        # Then: 기본 설정값들이 적용되어야 함
        assert monitoring_service.config.check_interval_minutes == 30  # 기본값
        assert monitoring_service.config.max_videos_per_channel == 5   # 기본값
        assert monitoring_service.config.process_immediately is True   # 기본값
        assert monitoring_service.config.notification_enabled is True  # 기본값
        assert monitoring_service.config.error_retry_count == 3        # 기본값
    
    def test_configuration_affects_video_fetching(
        self,
        mock_dependencies,
        sample_channel_infos
    ):
        """설정이 영상 조회에 영향을 주는지 테스트"""
        
        # Given: max_videos_per_channel을 2로 제한하는 설정
        from src.services.periodic_monitoring import MonitoringConfig
        
        limited_config = MonitoringConfig(
            max_videos_per_channel=2  # 채널당 최대 2개 영상만
        )
        
        monitoring_service = PeriodicMonitoringService(
            channel_repository=mock_dependencies['channel_repository'],
            video_repository=mock_dependencies['video_repository'],
            video_info_service=mock_dependencies['video_info_service'],
            immediate_processing_service=mock_dependencies['immediate_processing_service'],
            notification_service=mock_dependencies['notification_service'],
            config=limited_config
        )
        
        channel_repository = mock_dependencies['channel_repository']
        video_info_service = mock_dependencies['video_info_service']
        
        # 와치리스트에 채널 하나 설정
        channel_repository.get_watchlist_channels.return_value = [sample_channel_infos[0]]
        
        # 영상 없음으로 설정 (테스트 목적)
        video_info_service.get_channel_videos.return_value = []
        
        # When: 채널 확인 실행
        monitoring_service.check_all_channels()
        
        # Then: max_results=2로 호출되었는지 확인
        video_info_service.get_channel_videos.assert_called_with(
            channel_id=sample_channel_infos[0].channel_id,
            max_results=2,  # 설정된 값
            order='date'
        )
    
    def test_process_immediately_disabled_configuration(
        self,
        mock_dependencies,
        sample_channel_infos,
        sample_new_videos
    ):
        """즉시 처리 비활성화 설정 테스트"""
        
        # Given: 즉시 처리를 비활성화하는 설정
        from src.services.periodic_monitoring import MonitoringConfig
        
        no_immediate_config = MonitoringConfig(
            process_immediately=False  # 즉시 처리 비활성화
        )
        
        monitoring_service = PeriodicMonitoringService(
            channel_repository=mock_dependencies['channel_repository'],
            video_repository=mock_dependencies['video_repository'],
            video_info_service=mock_dependencies['video_info_service'],
            immediate_processing_service=mock_dependencies['immediate_processing_service'],
            notification_service=mock_dependencies['notification_service'],
            config=no_immediate_config
        )
        
        channel_repository = mock_dependencies['channel_repository']
        video_info_service = mock_dependencies['video_info_service']
        video_repository = mock_dependencies['video_repository']
        immediate_processing_service = mock_dependencies['immediate_processing_service']
        
        # 와치리스트와 새 영상 설정
        channel_repository.get_watchlist_channels.return_value = [sample_channel_infos[0]]
        video_info_service.get_channel_videos.return_value = [sample_new_videos[0].to_dict()]
        video_repository.get_video_by_id.return_value = None  # 새 영상
        
        # When: 채널 확인 실행
        result = monitoring_service.check_all_channels()
        
        # Then: 새 영상을 발견했지만 즉시 처리는 하지 않음
        assert result.new_videos_found == 1
        assert result.videos_processed_successfully == 0  # 처리하지 않음
        
        # 즉시 처리 서비스는 호출되지 않았어야 함
        immediate_processing_service.process_video.assert_not_called()


class TestErrorRecoveryInMonitoring:
    """모니터링 중 에러 복구 테스트"""
    
    def test_error_recovery_in_monitoring(
        self,
        mock_dependencies,
        sample_channel_infos
    ):
        """모니터링 중 에러 복구 기본 테스트"""
        
        # Given: 에러 재시도 설정이 있는 모니터링 서비스
        from src.services.periodic_monitoring import MonitoringConfig
        
        retry_config = MonitoringConfig(
            error_retry_count=3,
            error_retry_delay_seconds=0.1  # 테스트용으로 짧게 설정
        )
        
        monitoring_service = PeriodicMonitoringService(
            channel_repository=mock_dependencies['channel_repository'],
            video_repository=mock_dependencies['video_repository'],
            video_info_service=mock_dependencies['video_info_service'],
            immediate_processing_service=mock_dependencies['immediate_processing_service'],
            notification_service=mock_dependencies['notification_service'],
            config=retry_config
        )
        
        channel_repository = mock_dependencies['channel_repository']
        video_info_service = mock_dependencies['video_info_service']
        
        # 와치리스트 설정
        channel_repository.get_watchlist_channels.return_value = sample_channel_infos
        
        # API 호출 시 처음 2번은 실패, 3번째는 성공하도록 설정
        call_count = {'count': 0}
        
        def mock_api_call_with_retry(*args, **kwargs):
            call_count['count'] += 1
            if call_count['count'] <= 2:
                raise YouTubeAPIError("Temporary API error")
            else:
                return []  # 성공
        
        video_info_service.get_channel_videos.side_effect = mock_api_call_with_retry
        
        # When: 에러 복구가 포함된 채널 확인 실행
        result = monitoring_service.check_all_channels_with_retry()
        
        # Then: 재시도를 통해 성공했어야 함
        assert result.total_channels_checked == 2
        assert result.errors == []  # 최종적으로는 에러가 없음
        assert result.retry_attempts > 0  # 재시도가 발생했음
        
        # API 호출이 3번 일어났는지 확인 (2번 실패 + 1번 성공)
        assert call_count['count'] >= 3
    
    def test_max_retry_exceeded(
        self,
        mock_dependencies,
        sample_channel_infos
    ):
        """최대 재시도 횟수 초과 테스트"""
        
        # Given: 재시도 횟수 제한이 있는 설정
        from src.services.periodic_monitoring import MonitoringConfig
        
        limited_retry_config = MonitoringConfig(
            error_retry_count=2,  # 최대 2번만 재시도
            error_retry_delay_seconds=0.1
        )
        
        monitoring_service = PeriodicMonitoringService(
            channel_repository=mock_dependencies['channel_repository'],
            video_repository=mock_dependencies['video_repository'],
            video_info_service=mock_dependencies['video_info_service'],
            immediate_processing_service=mock_dependencies['immediate_processing_service'],
            notification_service=mock_dependencies['notification_service'],
            config=limited_retry_config
        )
        
        channel_repository = mock_dependencies['channel_repository']
        video_info_service = mock_dependencies['video_info_service']
        
        # 하나의 채널로 제한
        channel_repository.get_watchlist_channels.return_value = [sample_channel_infos[0]]
        
        # API 호출이 계속 실패하도록 설정
        video_info_service.get_channel_videos.side_effect = YouTubeAPIError("Persistent API error")
        
        # When: 재시도 제한 초과 상황에서 채널 확인 실행
        result = monitoring_service.check_all_channels_with_retry()
        
        # Then: 최대 재시도 후 에러로 기록되어야 함
        assert result.total_channels_checked == 1
        assert len(result.errors) == 1
        assert "Persistent API error" in result.errors[0]
        assert result.retry_attempts == 2  # 설정된 최대 재시도 횟수
    
    def test_partial_recovery_with_multiple_channels(
        self,
        mock_dependencies,
        sample_channel_infos
    ):
        """여러 채널에서 부분 복구 테스트"""
        
        # Given: 재시도 설정이 있는 모니터링 서비스
        from src.services.periodic_monitoring import MonitoringConfig
        
        retry_config = MonitoringConfig(
            error_retry_count=2,
            error_retry_delay_seconds=0.1
        )
        
        monitoring_service = PeriodicMonitoringService(
            channel_repository=mock_dependencies['channel_repository'],
            video_repository=mock_dependencies['video_repository'],
            video_info_service=mock_dependencies['video_info_service'],
            immediate_processing_service=mock_dependencies['immediate_processing_service'],
            notification_service=mock_dependencies['notification_service'],
            config=retry_config
        )
        
        channel_repository = mock_dependencies['channel_repository']
        video_info_service = mock_dependencies['video_info_service']
        
        # 두 채널 설정
        channel_repository.get_watchlist_channels.return_value = sample_channel_infos
        
        # 첫 번째 채널은 항상 실패, 두 번째 채널은 성공하도록 설정
        def mock_selective_failure(channel_id, *args, **kwargs):
            if channel_id == sample_channel_infos[0].channel_id:
                raise YouTubeAPIError("Channel 1 error")
            else:
                return []  # 채널 2는 성공
        
        video_info_service.get_channel_videos.side_effect = mock_selective_failure
        
        # When: 부분 실패 상황에서 채널 확인 실행
        result = monitoring_service.check_all_channels_with_retry()
        
        # Then: 일부는 성공, 일부는 실패해야 함
        assert result.total_channels_checked == 2
        assert len(result.errors) == 1  # 첫 번째 채널만 실패
        assert "Channel 1 error" in result.errors[0]
        assert result.retry_attempts >= 1  # 재시도가 발생했음
    
    def test_exponential_backoff_delay(
        self,
        mock_dependencies,
        sample_channel_infos
    ):
        """지수 백오프 지연 테스트"""
        
        # Given: 지수 백오프가 활성화된 설정
        from src.services.periodic_monitoring import MonitoringConfig
        
        backoff_config = MonitoringConfig(
            error_retry_count=3,
            error_retry_delay_seconds=0.1,
            use_exponential_backoff=True  # 지수 백오프 활성화
        )
        
        monitoring_service = PeriodicMonitoringService(
            channel_repository=mock_dependencies['channel_repository'],
            video_repository=mock_dependencies['video_repository'],
            video_info_service=mock_dependencies['video_info_service'],
            immediate_processing_service=mock_dependencies['immediate_processing_service'],
            notification_service=mock_dependencies['notification_service'],
            config=backoff_config
        )
        
        channel_repository = mock_dependencies['channel_repository']
        video_info_service = mock_dependencies['video_info_service']
        
        channel_repository.get_watchlist_channels.return_value = [sample_channel_infos[0]]
        
        # API 호출이 처음 2번 실패, 3번째 성공
        call_count = {'count': 0}
        
        def mock_api_with_eventual_success(*args, **kwargs):
            call_count['count'] += 1
            if call_count['count'] <= 2:
                raise YouTubeAPIError("Temporary error")
            return []
        
        video_info_service.get_channel_videos.side_effect = mock_api_with_eventual_success
        
        # When: 지수 백오프를 포함한 재시도 실행
        import time
        start_time = time.time()
        result = monitoring_service.check_all_channels_with_retry()
        elapsed_time = time.time() - start_time
        
        # Then: 재시도가 성공하고, 지연시간이 점진적으로 증가했어야 함
        assert result.total_channels_checked == 1
        assert result.errors == []
        assert result.retry_attempts == 2
        
        # 지수 백오프로 인한 최소 지연시간 확인 (0.1 + 0.2 = 0.3초)
        assert elapsed_time >= 0.3 