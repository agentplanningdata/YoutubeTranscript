"""
주기적 모니터링 서비스

와치리스트에 있는 채널들을 주기적으로 확인하여 새 영상을 탐지하고 처리합니다.
"""

import time
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.services.models import ChannelInfo, VideoInfo
from src.utils.exceptions import VideoProcessingError, YouTubeAPIError, DatabaseError
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)


@dataclass
class MonitoringResult:
    """모니터링 결과"""
    total_channels_checked: int = 0
    new_videos_found: int = 0
    videos_processed_successfully: int = 0
    videos_skipped: int = 0
    errors: List[str] = None
    retry_attempts: int = 0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class MonitoringConfig:
    """모니터링 설정"""
    check_interval_minutes: int = 30
    max_videos_per_channel: int = 5
    process_immediately: bool = True
    notification_enabled: bool = True
    error_retry_count: int = 3
    error_retry_delay_seconds: int = 60
    use_exponential_backoff: bool = False


class PeriodicMonitoringService:
    """
    주기적 모니터링 서비스
    
    와치리스트에 있는 채널들을 주기적으로 확인하여:
    1. 각 채널의 최신 영상들을 조회
    2. 새로 업로드된 영상 탐지
    3. 새 영상을 즉시 처리 (자막 다운로드, 벡터화)
    4. 처리 결과 알림 발송
    """
    
    def __init__(
        self,
        channel_repository=None,
        video_repository=None,
        video_info_service=None,
        immediate_processing_service=None,
        notification_service=None,
        config: Optional[MonitoringConfig] = None
    ):
        """
        주기적 모니터링 서비스를 초기화합니다.
        
        Args:
            channel_repository: 채널 레포지토리
            video_repository: 비디오 레포지토리  
            video_info_service: 비디오 정보 서비스
            immediate_processing_service: 즉시 처리 서비스
            notification_service: 알림 서비스
            config: 모니터링 설정
        """
        self.channel_repository = channel_repository
        self.video_repository = video_repository
        self.video_info_service = video_info_service
        self.immediate_processing_service = immediate_processing_service
        self.notification_service = notification_service
        self.config = config or MonitoringConfig()
        
        # 설정 검증
        self._validate_config()
        
        logger.info(f"Periodic monitoring service initialized with config: {self.config}")
    
    def _validate_config(self) -> None:
        """모니터링 설정을 검증합니다."""
        if self.config.error_retry_count < 0:
            raise ValueError("error_retry_count must be non-negative")
        
        if self.config.error_retry_delay_seconds < 0:
            raise ValueError("error_retry_delay_seconds must be non-negative")
        
        if self.config.max_videos_per_channel < 1:
            raise ValueError("max_videos_per_channel must be at least 1")
    
    def check_all_channels(self) -> MonitoringResult:
        """
        모든 와치리스트 채널들을 확인하여 새 영상을 탐지하고 처리합니다.
        
        Returns:
            모니터링 결과
        """
        result = MonitoringResult()
        
        try:
            logger.info("Starting scheduled channel check")
            
            # 1. 와치리스트 채널들 조회
            channels = self.channel_repository.get_watchlist_channels()
            result.total_channels_checked = len(channels)
            
            if not channels:
                logger.info("No channels in watchlist")
                return result
            
            logger.info(f"Checking {len(channels)} channels in watchlist")
            
            # 2. 각 채널별로 새 영상 확인
            for channel in channels:
                try:
                    self._check_single_channel(channel, result)
                except Exception as e:
                    error_msg = f"Error checking channel {channel.channel_id}: {str(e)}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
            
            logger.info(f"Channel check completed. Found {result.new_videos_found} new videos, processed {result.videos_processed_successfully}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to check all channels: {e}")
            result.errors.append(f"Failed to check all channels: {e}")
            return result
    
    def check_all_channels_with_retry(self) -> MonitoringResult:
        """
        재시도 로직이 포함된 모든 와치리스트 채널 확인
        
        Returns:
            모니터링 결과 (재시도 정보 포함)
        """
        
        result = MonitoringResult()
        
        try:
            logger.info("Starting scheduled channel check with retry logic")
            
            # 1. 와치리스트 채널들 조회
            channels = self.channel_repository.get_watchlist_channels()
            result.total_channels_checked = len(channels)
            
            if not channels:
                logger.info("No channels in watchlist")
                return result
            
            logger.info(f"Checking {len(channels)} channels in watchlist (with retry)")
            
            # 2. 각 채널별로 재시도 로직과 함께 확인
            for channel in channels:
                success = self._check_single_channel_with_retry(channel, result)
                if not success:
                    # 개별 채널 실패는 이미 result.errors에 기록됨
                    pass
            
            logger.info(f"Channel check with retry completed. Found {result.new_videos_found} new videos, processed {result.videos_processed_successfully}, retry attempts: {result.retry_attempts}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to check all channels with retry: {e}")
            result.errors.append(f"Failed to check all channels: {e}")
            return result
    
    def _check_single_channel_with_retry(self, channel: ChannelInfo, result: MonitoringResult) -> bool:
        """재시도 로직이 포함된 단일 채널 확인"""
        
        for attempt in range(self.config.error_retry_count + 1):  # +1 for initial attempt
            # 재시도 카운트 (초기 시도 제외, 성공/실패 무관)
            if attempt > 0:  
                result.retry_attempts += 1
            
            try:
                logger.debug(f"Checking channel: {channel.channel_id} (attempt {attempt + 1})")
                
                # 채널 확인 시도
                self._check_single_channel(channel, result)
                return True  # 성공
                
            except Exception as e:
                if attempt >= self.config.error_retry_count:
                    # 최대 재시도 횟수 초과
                    error_msg = f"Error checking channel {channel.channel_id} after {self.config.error_retry_count} retries: {str(e)}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
                    return False
                
                # 재시도 지연
                delay = self._calculate_retry_delay(attempt)
                logger.warning(f"Channel {channel.channel_id} check failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                time.sleep(delay)
        
        return False
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """재시도 지연 시간을 계산합니다."""
        base_delay = self.config.error_retry_delay_seconds
        
        if self.config.use_exponential_backoff:
            # 지수 백오프: base_delay * 2^attempt
            return base_delay * (2 ** attempt)
        else:
            # 고정 지연
            return base_delay
    
    def _check_single_channel(self, channel: ChannelInfo, result: MonitoringResult) -> None:
        """단일 채널을 확인하여 새 영상을 처리합니다."""
        try:
            logger.debug(f"Checking channel: {channel.channel_id}")
            
            # 1. 채널의 최신 영상들 조회
            latest_videos = self._fetch_latest_videos(channel.channel_id)
            if not latest_videos:
                return
            
            # 2. 각 영상 처리
            for video_data in latest_videos:
                self._process_single_video(video_data, result)
                        
        except YouTubeAPIError as e:
            # YouTube API 오류는 특별히 처리
            raise YouTubeAPIError(f"YouTube API error for channel {channel.channel_id}: {e}")
        except Exception as e:
            raise Exception(f"Error checking channel {channel.channel_id}: {e}")
    
    def _fetch_latest_videos(self, channel_id: str) -> List[Dict[str, Any]]:
        """채널의 최신 영상들을 조회합니다."""
        latest_videos = self.video_info_service.get_channel_videos(
            channel_id=channel_id,
            max_results=self.config.max_videos_per_channel,
            order='date'
        )
        
        if not latest_videos:
            logger.debug(f"No videos found for channel: {channel_id}")
            
        return latest_videos
    
    def _process_single_video(self, video_data: Dict[str, Any], result: MonitoringResult) -> None:
        """단일 영상을 처리합니다."""
        video_info = self._convert_to_video_info(video_data)
        result.new_videos_found += 1
        
        # 이미 처리된 영상인지 확인
        if self._is_video_already_processed(video_info.video_id):
            logger.debug(f"Video {video_info.video_id} already processed, skipping")
            result.videos_skipped += 1
            return
        
        # 새 영상 처리
        if self.config.process_immediately:
            success = self._process_new_video(video_info, result)
            if success:
                result.videos_processed_successfully += 1
    
    def _is_video_already_processed(self, video_id: str) -> bool:
        """영상이 이미 처리되었는지 확인합니다."""
        existing_video = self.video_repository.get_video_by_id(video_id)
        return (existing_video and 
                getattr(existing_video, 'processing_status', None) == 'completed')
    
    def _process_new_video(self, video_info: VideoInfo, result: MonitoringResult) -> bool:
        """새 영상을 처리합니다."""
        try:
            success = self.immediate_processing_service.process_video(video_info)
            if success:
                logger.info(f"Successfully processed video: {video_info.video_id}")
                return True
            else:
                logger.warning(f"Processing returned False for video: {video_info.video_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to process video {video_info.video_id}: {e}")
            result.errors.append(f"Failed to process video {video_info.video_id}: {e}")
            return False
    
    def _convert_to_video_info(self, video_data: Dict[str, Any]) -> VideoInfo:
        """비디오 데이터를 VideoInfo 객체로 변환합니다."""
        return VideoInfo(
            video_id=video_data['video_id'],
            title=video_data['title'],
            description=video_data.get('description', ''),
            channel_id=video_data['channel_id'],
            published_at=video_data['published_at'],
            thumbnail_url=video_data['thumbnail_url'],
            channel_title=video_data.get('channel_title'),
            duration=video_data.get('duration'),
            view_count=video_data.get('view_count')
        )
    
    def detect_new_videos(
        self, 
        channel_info: ChannelInfo, 
        last_check_time: datetime
    ) -> List[VideoInfo]:
        """
        특정 채널에서 마지막 확인 시간 이후의 새 영상들을 탐지합니다.
        
        Args:
            channel_info: 채널 정보
            last_check_time: 마지막 확인 시간
            
        Returns:
            새로 발견된 영상들의 리스트
        """
        try:
            logger.debug(f"Detecting new videos for channel {channel_info.channel_id} since {last_check_time}")
            
            # 마지막 확인 시간 이후 영상들 조회
            recent_videos = self.video_info_service.get_channel_videos(
                channel_id=channel_info.channel_id,
                max_results=self.config.max_videos_per_channel,
                order='date',
                published_after=last_check_time
            )
            
            if not recent_videos:
                logger.debug(f"No new videos found for channel: {channel_info.channel_id}")
                return []
            
            # VideoInfo 객체로 변환하고 실제 새 영상인지 확인
            new_videos = []
            for video_data in recent_videos:
                video_info = self._convert_to_video_info(video_data)
                
                # 데이터베이스에서 해당 영상이 이미 있는지 확인
                existing_video = self.video_repository.get_video_by_id(video_info.video_id)
                if not existing_video:
                    new_videos.append(video_info)
                    logger.debug(f"New video detected: {video_info.video_id}")
            
            logger.info(f"Found {len(new_videos)} new videos for channel {channel_info.channel_id}")
            return new_videos
            
        except Exception as e:
            logger.error(f"Failed to detect new videos for channel {channel_info.channel_id}: {e}")
            return [] 