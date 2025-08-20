"""
영상 정보 조회 서비스

YouTube API를 통한 영상 정보 조회 기능을 제공합니다.
"""

from typing import Optional, List, Dict, Any
import re

from src.services.youtube_api_client import YouTubeAPIClient
from src.utils.exceptions import YouTubeAPIError, VideoProcessingError
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)


class VideoInfoService:
    """YouTube 영상 정보 조회 서비스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        영상 정보 서비스를 초기화합니다.
        
        Args:
            api_key: YouTube Data API 키. None일 경우 환경변수에서 가져옴
        """
        self.youtube_client = YouTubeAPIClient(api_key=api_key)
        logger.info("Video info service initialized")
    
    def get_video_details(self, video_id: str) -> Dict[str, Any]:
        """
        영상 ID로 상세 정보를 조회합니다.
        
        Args:
            video_id: 조회할 영상의 YouTube ID
            
        Returns:
            영상 상세 정보 딕셔너리
            
        Raises:
            ValueError: 영상 ID가 비어있을 경우
            VideoProcessingError: 영상 정보 조회 실패 시
        """
        # 입력 유효성 검사
        if not video_id or (isinstance(video_id, str) and not video_id.strip()):
            raise ValueError("Video ID cannot be empty")
        
        if video_id is None:
            raise ValueError("Video ID cannot be empty")
        
        try:
            # YouTube API를 통한 영상 정보 조회
            video_info = self.youtube_client.get_video_details(video_id.strip())
            
            # None 값들을 적절히 처리
            processed_info = self._process_video_info(video_info)
            
            logger.info(f"Retrieved video details for: {video_id}")
            return processed_info
            
        except YouTubeAPIError as e:
            logger.error(f"YouTube API error getting video details for {video_id}: {e}")
            raise VideoProcessingError(f"Failed to get video details for {video_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting video details for {video_id}: {e}")
            raise VideoProcessingError(f"Failed to get video details for {video_id}: {e}")
    
    def get_channel_videos(
        self, 
        channel_id: str, 
        max_results: int = 20,
        order: str = 'date'
    ) -> List[Dict[str, Any]]:
        """
        채널의 영상 목록을 조회합니다.
        
        Args:
            channel_id: 채널 ID
            max_results: 최대 결과 수 (기본값: 20)
            order: 정렬 순서 ('date', 'relevance', 'viewCount', 'rating')
            
        Returns:
            영상 정보 리스트
            
        Raises:
            ValueError: 채널 ID가 비어있을 경우
            VideoProcessingError: 영상 목록 조회 실패 시
        """
        # 입력 유효성 검사
        if not channel_id or (isinstance(channel_id, str) and not channel_id.strip()):
            raise ValueError("Channel ID cannot be empty")
        
        if channel_id is None:
            raise ValueError("Channel ID cannot be empty")
        
        try:
            # YouTube API를 통한 채널 영상 목록 조회
            videos = self.youtube_client.get_channel_videos(
                channel_id=channel_id.strip(),
                max_results=max_results
            )
            
            # 각 영상 정보 처리
            processed_videos = []
            for video in videos:
                processed_video = self._process_video_info(video)
                processed_videos.append(processed_video)
            
            logger.info(f"Retrieved {len(processed_videos)} videos for channel: {channel_id}")
            return processed_videos
            
        except YouTubeAPIError as e:
            logger.error(f"YouTube API error getting videos for channel {channel_id}: {e}")
            raise VideoProcessingError(f"Failed to get videos for channel {channel_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting videos for channel {channel_id}: {e}")
            raise VideoProcessingError(f"Failed to get videos for channel {channel_id}: {e}")
    
    def get_multiple_video_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """
        여러 영상의 상세 정보를 한 번에 조회합니다.
        
        Args:
            video_ids: 영상 ID 리스트
            
        Returns:
            영상 정보 리스트
            
        Raises:
            ValueError: 영상 ID 리스트가 비어있을 경우
            VideoProcessingError: 영상 정보 조회 실패 시
        """
        if not video_ids:
            raise ValueError("Video IDs list cannot be empty")
        
        # 빈 ID 제거
        valid_video_ids = [vid for vid in video_ids if vid and vid.strip()]
        
        if not valid_video_ids:
            raise ValueError("No valid video IDs provided")
        
        try:
            results = []
            
            # YouTube API는 한 번에 여러 영상을 조회할 수 있지만,
            # 여기서는 개별 조회로 구현 (에러 처리를 위해)
            for video_id in valid_video_ids:
                try:
                    video_info = self.get_video_details(video_id)
                    results.append(video_info)
                except VideoProcessingError as e:
                    logger.warning(f"Skipping video {video_id}: {e}")
                    # 개별 영상 오류는 전체를 실패시키지 않음
                    continue
            
            logger.info(f"Retrieved details for {len(results)}/{len(valid_video_ids)} videos")
            return results
            
        except Exception as e:
            logger.error(f"Unexpected error getting multiple video details: {e}")
            raise VideoProcessingError(f"Failed to get multiple video details: {e}")
    
    def search_videos(
        self, 
        query: str, 
        max_results: int = 20,
        order: str = 'relevance'
    ) -> List[Dict[str, Any]]:
        """
        키워드로 영상을 검색합니다.
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            order: 정렬 순서 ('relevance', 'date', 'viewCount', 'rating')
            
        Returns:
            검색된 영상 정보 리스트
            
        Raises:
            ValueError: 검색 쿼리가 비어있을 경우
            VideoProcessingError: 영상 검색 실패 시
        """
        if not query or (isinstance(query, str) and not query.strip()):
            raise ValueError("Search query cannot be empty")
        
        try:
            # YouTube API를 통한 영상 검색
            # 실제 구현에서는 search().list() API를 사용
            # 여기서는 기본 구조만 구현
            
            logger.info(f"Searching videos with query: {query}")
            
            # TODO: 실제 YouTube 검색 API 호출 구현
            # 현재는 빈 리스트 반환
            return []
            
        except Exception as e:
            logger.error(f"Unexpected error searching videos: {e}")
            raise VideoProcessingError(f"Failed to search videos: {e}")
    
    def _process_video_info(self, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        YouTube API 응답 데이터를 처리하고 정규화합니다.
        
        Args:
            video_info: 원본 영상 정보
            
        Returns:
            처리된 영상 정보
        """
        if not video_info:
            return video_info
        
        # 기본 복사
        processed = video_info.copy()
        
        # None 값 처리
        if processed.get('title') is None:
            processed['title'] = ""
        
        if processed.get('description') is None:
            processed['description'] = ""
        
        # 태그 처리
        if 'tags' in processed and processed['tags'] is None:
            processed['tags'] = []
        elif 'tags' not in processed:
            processed['tags'] = []
        
        # 통계 정보 정수 변환
        stat_fields = ['view_count', 'like_count', 'comment_count', 'subscriber_count']
        for field in stat_fields:
            if field in processed and processed[field] is not None:
                try:
                    processed[field] = int(processed[field])
                except (ValueError, TypeError):
                    processed[field] = None
        
        # Duration 처리 (추가 기능)
        if 'duration' in processed and processed['duration']:
            readable_duration = self._parse_duration(processed['duration'])
            if readable_duration:
                processed['duration_readable'] = readable_duration
        
        return processed
    
    def _parse_duration(self, iso_duration: str) -> Optional[str]:
        """
        ISO 8601 duration을 읽기 쉬운 형태로 변환합니다.
        
        Args:
            iso_duration: ISO 8601 형식 duration (예: PT3M45S)
            
        Returns:
            읽기 쉬운 형태의 duration (예: 3분 45초)
        """
        if not iso_duration or not iso_duration.startswith('PT'):
            return None
        
        try:
            # PT3M45S -> 3분 45초
            pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
            match = re.match(pattern, iso_duration)
            
            if not match:
                return None
            
            hours, minutes, seconds = match.groups()
            
            parts = []
            
            if hours:
                parts.append(f"{hours}시간")
            
            if minutes:
                parts.append(f"{minutes}분")
            
            if seconds:
                parts.append(f"{seconds}초")
            
            return " ".join(parts) if parts else None
            
        except Exception:
            return None
    
    def get_video_thumbnail_urls(self, video_id: str) -> Dict[str, str]:
        """
        영상의 다양한 크기 썸네일 URL을 반환합니다.
        
        Args:
            video_id: 영상 ID
            
        Returns:
            썸네일 URL 딕셔너리 (key: 크기, value: URL)
        """
        base_url = f"https://i.ytimg.com/vi/{video_id}"
        
        return {
            'default': f"{base_url}/default.jpg",      # 120x90
            'medium': f"{base_url}/mqdefault.jpg",     # 320x180
            'high': f"{base_url}/hqdefault.jpg",       # 480x360
            'standard': f"{base_url}/sddefault.jpg",   # 640x480
            'maxres': f"{base_url}/maxresdefault.jpg", # 1280x720
        }
    
    def is_video_available(self, video_id: str) -> bool:
        """
        영상이 사용 가능한지 확인합니다.
        
        Args:
            video_id: 확인할 영상 ID
            
        Returns:
            영상 사용 가능 여부
        """
        try:
            video_info = self.get_video_details(video_id)
            return video_info is not None and video_info.get('title') is not None
        except VideoProcessingError:
            return False
        except Exception:
            return False 