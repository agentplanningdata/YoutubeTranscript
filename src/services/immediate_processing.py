"""
즉시 처리 시스템 서비스

채널이 와치리스트에 추가될 때 해당 채널의 최신 영상을 자동으로 처리합니다.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from dataclasses import dataclass

from src.services.models import ChannelInfo, VideoInfo
from src.utils.exceptions import VideoProcessingError, TranscriptNotFoundError
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)


@dataclass
class ProcessingConfig:
    """즉시 처리 설정"""
    primary_language: str = "ko"
    fallback_languages: List[str] = None
    max_videos_to_process: int = 1
    skip_if_processed: bool = True
    
    def __post_init__(self):
        if self.fallback_languages is None:
            self.fallback_languages = ["en"]


class ImmediateProcessingService:
    """
    즉시 처리 시스템 서비스
    
    채널이 와치리스트에 추가되면 해당 채널의 최신 영상을 자동으로 처리합니다:
    1. 채널의 최신 영상 조회
    2. 영상 정보를 데이터베이스에 저장
    3. 자막 다운로드
    4. 자막 처리 및 청킹
    5. 벡터 데이터베이스에 저장
    6. 처리 완료 마킹
    """
    
    def __init__(
        self,
        channel_repository=None,
        video_repository=None,
        video_info_service=None,
        transcript_downloader=None,
        transcript_processor=None,
        vector_store=None,
        config: Optional[ProcessingConfig] = None
    ):
        """
        즉시 처리 서비스를 초기화합니다.
        
        Args:
            channel_repository: 채널 레포지토리
            video_repository: 비디오 레포지토리
            video_info_service: 비디오 정보 서비스
            transcript_downloader: 자막 다운로더
            transcript_processor: 자막 처리기
            vector_store: 벡터 저장소
            config: 처리 설정
        """
        self.channel_repository = channel_repository
        self.video_repository = video_repository
        self.video_info_service = video_info_service
        self.transcript_downloader = transcript_downloader
        self.transcript_processor = transcript_processor
        self.vector_store = vector_store
        self.config = config or ProcessingConfig()
        
        logger.info(f"Immediate processing service initialized with config: {self.config}")
    
    def process_latest_video_on_channel_add(self, channel_info: ChannelInfo) -> bool:
        """
        채널 추가 시 최신 영상을 처리합니다.
        
        Args:
            channel_info: 추가된 채널 정보
            
        Returns:
            처리 성공 여부
            
        Raises:
            VideoProcessingError: 처리 중 오류 발생 시
        """
        try:
            logger.info(f"Starting immediate processing for channel: {channel_info.channel_id}")
            
            # 1. 채널의 최신 영상 조회
            latest_video = self._get_latest_video(channel_info.channel_id)
            if not latest_video:
                return False
            
            # 2. 이미 처리된 영상인지 확인
            if self.config.skip_if_processed and self._is_already_processed(latest_video.video_id):
                logger.info(f"Video {latest_video.video_id} already processed, skipping")
                return True
            
            # 3. 영상 정보 저장
            self._save_video_info(latest_video)
            
            # 4. 자막 다운로드 및 처리
            processed_chunks = self._download_and_process_transcript(latest_video)
            if not processed_chunks:
                return False  # 자막이 없는 경우
            
            # 5. 벡터 데이터베이스에 저장
            self._store_in_vector_db(processed_chunks)
            
            # 6. 처리 완료 마킹
            self._mark_as_completed(latest_video.video_id)
            
            logger.info(f"Successfully completed immediate processing for video: {latest_video.video_id}")
            return True
            
        except TranscriptNotFoundError:
            # 자막이 없는 경우는 에러가 아닌 정상적인 상황으로 처리
            return False
            
        except Exception as e:
            logger.error(f"Failed to process latest video for channel {channel_info.channel_id}: {e}")
            raise VideoProcessingError(f"Failed to process latest video: {e}")
    
    def _get_latest_video(self, channel_id: str) -> Optional[VideoInfo]:
        """채널의 최신 영상을 조회합니다."""
        latest_videos = self.video_info_service.get_channel_videos(
            channel_id=channel_id,
            max_results=self.config.max_videos_to_process,
            order='date'
        )
        
        if not latest_videos:
            logger.warning(f"No videos found for channel: {channel_id}")
            return None
        
        latest_video_data = latest_videos[0]
        return VideoInfo(
            video_id=latest_video_data['video_id'],
            title=latest_video_data['title'],
            description=latest_video_data.get('description', ''),
            channel_id=latest_video_data['channel_id'],
            published_at=latest_video_data['published_at'],
            thumbnail_url=latest_video_data['thumbnail_url'],
            channel_title=latest_video_data.get('channel_title'),
            duration=latest_video_data.get('duration'),
            view_count=latest_video_data.get('view_count')
        )
    
    def _is_already_processed(self, video_id: str) -> bool:
        """영상이 이미 처리되었는지 확인합니다."""
        existing_video = self.video_repository.get_video_by_id(video_id)
        return (existing_video and 
                getattr(existing_video, 'processing_status', None) == 'completed')
    
    def _save_video_info(self, video_info: VideoInfo) -> None:
        """영상 정보를 데이터베이스에 저장합니다."""
        self.video_repository.save_video_info(video_info)
        logger.info(f"Saved video info: {video_info.video_id}")
    
    def _download_and_process_transcript(self, video_info: VideoInfo) -> Optional[List[Any]]:
        """자막을 다운로드하고 처리합니다."""
        try:
            # 자막 다운로드
            transcript_data = self.transcript_downloader.download_transcript(
                video_id=video_info.video_id,
                language=self.config.primary_language,
                fallback_languages=self.config.fallback_languages
            )
            logger.info(f"Downloaded transcript for video: {video_info.video_id}")
            
            # 자막 처리 및 청킹
            video_metadata = {
                'video_id': video_info.video_id,
                'title': video_info.title,
                'channel_id': video_info.channel_id,
                'channel_title': video_info.channel_title
            }
            
            processed_chunks = self.transcript_processor.process_transcript(
                transcript_data=transcript_data,
                video_metadata=video_metadata
            )
            logger.info(f"Processed transcript into {len(processed_chunks)} chunks")
            return processed_chunks
            
        except TranscriptNotFoundError as e:
            logger.warning(f"No transcript found for video {video_info.video_id}: {e}")
            return None
    
    def _store_in_vector_db(self, processed_chunks: List[Any]) -> List[str]:
        """처리된 청크들을 벡터 데이터베이스에 저장합니다."""
        document_ids = self.vector_store.add_documents(processed_chunks)
        logger.info(f"Stored {len(document_ids)} chunks in vector database")
        return document_ids
    
    def _mark_as_completed(self, video_id: str) -> None:
        """영상 처리를 완료로 마킹합니다."""
        self.video_repository.mark_video_as_processed(
            video_id=video_id,
            processing_status='completed'
        )
        logger.info(f"Marked video {video_id} as completed") 