"""
YouTube API 클라이언트

YouTube Data API v3를 사용하여 채널 검색, 영상 정보 조회 등의 기능을 제공합니다.
"""

import os
from typing import Optional, List, Dict, Any

# Google API 클라이언트 import 처리
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    # 테스트 환경에서는 mock을 사용할 수 있도록
    def build(*args, **kwargs):
        raise ImportError("Google API client not available")
    
    class HttpError(Exception):
        pass
    
    GOOGLE_API_AVAILABLE = False

from src.utils.exceptions import YouTubeAPIError, ConfigurationError
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)


class YouTubeAPIClient:
    """YouTube Data API v3 클라이언트"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        YouTube API 클라이언트를 초기화합니다.
        
        Args:
            api_key: YouTube Data API 키. None일 경우 환경변수에서 가져옴
            
        Raises:
            ConfigurationError: API 키가 없거나 유효하지 않은 경우
        """
        # API 키 설정
        if api_key is None:
            api_key = os.getenv('YOUTUBE_API_KEY')
        
        if not api_key:
            raise ConfigurationError("YouTube API key is required. Set YOUTUBE_API_KEY environment variable or provide api_key parameter.")
        
        if api_key == "":
            raise ConfigurationError("YouTube API key cannot be empty.")
        
        self.api_key = api_key
        
        # 기본 설정
        self.max_results = 25
        self.region_code = 'KR'
        self.language = 'ko'
        
        # Google API 클라이언트 빌드
        try:
            self.service = build('youtube', 'v3', developerKey=self.api_key)
            logger.info(f"YouTube API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YouTube API client: {e}")
            raise YouTubeAPIError(f"Failed to initialize YouTube API client: {e}")
    
    def search_channels(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        채널을 검색합니다.
        
        Args:
            query: 검색할 채널명
            max_results: 최대 결과 수 (기본값: self.max_results)
            
        Returns:
            검색된 채널 정보 리스트
            
        Raises:
            YouTubeAPIError: API 호출 실패 시
        """
        if max_results is None:
            max_results = self.max_results
            
        try:
            request = self.service.search().list(
                part='snippet',
                q=query,
                type='channel',
                maxResults=max_results,
                regionCode=self.region_code
            )
            response = request.execute()
            
            channels = []
            for item in response.get('items', []):
                channel_info = {
                    'channel_id': item['id']['channelId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'thumbnail_url': item['snippet']['thumbnails']['default']['url'],
                    'published_at': item['snippet']['publishedAt']
                }
                channels.append(channel_info)
                
            logger.info(f"Found {len(channels)} channels for query: {query}")
            return channels
            
        except HttpError as e:
            logger.error(f"YouTube API error in search_channels: {e}")
            raise YouTubeAPIError(f"Failed to search channels: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in search_channels: {e}")
            raise YouTubeAPIError(f"Unexpected error in search channels: {e}")
    
    def get_channel_details(self, channel_id: str) -> Dict[str, Any]:
        """
        채널의 상세 정보를 조회합니다.
        
        Args:
            channel_id: 채널 ID
            
        Returns:
            채널 상세 정보
            
        Raises:
            YouTubeAPIError: API 호출 실패 시
        """
        try:
            request = self.service.channels().list(
                part='snippet,statistics,brandingSettings',
                id=channel_id
            )
            response = request.execute()
            
            if not response.get('items'):
                raise YouTubeAPIError(f"Channel not found: {channel_id}")
            
            item = response['items'][0]
            channel_details = {
                'channel_id': item['id'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'subscriber_count': int(item['statistics'].get('subscriberCount', 0)),
                'video_count': int(item['statistics'].get('videoCount', 0)),
                'view_count': int(item['statistics'].get('viewCount', 0)),
                'thumbnail_url': item['snippet']['thumbnails']['default']['url'],
                'created_at': item['snippet']['publishedAt']
            }
            
            logger.info(f"Retrieved details for channel: {channel_details['title']}")
            return channel_details
            
        except HttpError as e:
            logger.error(f"YouTube API error in get_channel_details: {e}")
            raise YouTubeAPIError(f"Failed to get channel details: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in get_channel_details: {e}")
            raise YouTubeAPIError(f"Unexpected error in get channel details: {e}")
    
    def get_channel_videos(self, channel_id: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        채널의 최신 영상 목록을 조회합니다.
        
        Args:
            channel_id: 채널 ID
            max_results: 최대 결과 수 (기본값: self.max_results)
            
        Returns:
            영상 정보 리스트
            
        Raises:
            YouTubeAPIError: API 호출 실패 시
        """
        if max_results is None:
            max_results = self.max_results
            
        try:
            request = self.service.search().list(
                part='snippet',
                channelId=channel_id,
                type='video',
                order='date',
                maxResults=max_results
            )
            response = request.execute()
            
            videos = []
            for item in response.get('items', []):
                video_info = {
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail_url': item['snippet']['thumbnails']['default']['url']
                }
                videos.append(video_info)
            
            logger.info(f"Retrieved {len(videos)} videos for channel: {channel_id}")
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube API error in get_channel_videos: {e}")
            raise YouTubeAPIError(f"Failed to get channel videos: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in get_channel_videos: {e}")
            raise YouTubeAPIError(f"Unexpected error in get channel videos: {e}")
    
    def get_video_details(self, video_id: str) -> Dict[str, Any]:
        """
        특정 영상의 상세 정보를 조회합니다.
        
        Args:
            video_id: 영상 ID
            
        Returns:
            영상 상세 정보
            
        Raises:
            YouTubeAPIError: API 호출 실패 시
        """
        try:
            request = self.service.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            )
            response = request.execute()
            
            if not response.get('items'):
                raise YouTubeAPIError(f"Video not found: {video_id}")
            
            item = response['items'][0]
            video_details = {
                'video_id': item['id'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'channel_id': item['snippet']['channelId'],
                'channel_title': item['snippet']['channelTitle'],
                'published_at': item['snippet']['publishedAt'],
                'duration': item['contentDetails']['duration'],
                'view_count': int(item['statistics'].get('viewCount', 0)),
                'like_count': int(item['statistics'].get('likeCount', 0)),
                'comment_count': int(item['statistics'].get('commentCount', 0)),
                'thumbnail_url': item['snippet']['thumbnails']['default']['url']
            }
            
            logger.info(f"Retrieved details for video: {video_details['title']}")
            return video_details
            
        except HttpError as e:
            logger.error(f"YouTube API error in get_video_details: {e}")
            raise YouTubeAPIError(f"Failed to get video details: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in get_video_details: {e}")
            raise YouTubeAPIError(f"Unexpected error in get video details: {e}") 