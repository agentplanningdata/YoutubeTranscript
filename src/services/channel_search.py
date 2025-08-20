"""
채널 검색 서비스

YouTube API를 통한 채널 검색 기능을 제공합니다.
"""

from typing import Optional, List, Dict, Any

from src.services.youtube_api_client import YouTubeAPIClient
from src.services.models import ChannelInfo, ChannelSearchResult
from src.utils.exceptions import YouTubeAPIError, ChannelNotFoundError
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)


class ChannelSearchService:
    """YouTube 채널 검색 서비스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        채널 검색 서비스를 초기화합니다.
        
        Args:
            api_key: YouTube Data API 키. None일 경우 환경변수에서 가져옴
        """
        self.youtube_client = YouTubeAPIClient(api_key=api_key)
        logger.info("Channel search service initialized")
    
    def search_channels(
        self, 
        query: str, 
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        채널명으로 채널을 검색합니다.
        
        Args:
            query: 검색할 채널명
            max_results: 최대 결과 수 (기본값: 10)
            
        Returns:
            검색된 채널 정보 리스트 (딕셔너리 형태)
            
        Raises:
            ValueError: 검색어가 비어있을 경우
            YouTubeAPIError: YouTube API 호출 실패 시
        """
        # 입력 유효성 검사
        if not query or (isinstance(query, str) and not query.strip()):
            raise ValueError("Search query cannot be empty")
        
        if query is None:
            raise ValueError("Search query cannot be empty")
        
        try:
            # YouTube API를 통한 채널 검색
            raw_channels = self.youtube_client.search_channels(
                query=query.strip(), 
                max_results=max_results
            )
            
            # 데이터 모델을 사용하여 검증 및 정규화
            channel_objects = []
            for raw_channel in raw_channels:
                try:
                    channel_info = ChannelInfo.from_dict(raw_channel)
                    channel_objects.append(channel_info)
                except ValueError as ve:
                    logger.warning(f"Invalid channel data skipped: {ve}")
                    # 유효하지 않은 데이터는 건너뛰고 계속 진행
                    continue
            
            logger.info(f"Found {len(channel_objects)} valid channels for query: {query}")
            
            # 호환성을 위해 딕셔너리로 변환하여 반환
            return [channel.to_dict() for channel in channel_objects]
            
        except YouTubeAPIError as e:
            logger.error(f"YouTube API error in channel search: {e}")
            # YouTubeAPIError는 그대로 전파
            raise
        except Exception as e:
            logger.error(f"Unexpected error in channel search: {e}")
            raise YouTubeAPIError(f"Unexpected error in channel search: {e}")
    
    def search_channels_with_details(
        self, 
        query: str, 
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        채널을 검색하고 각 채널의 상세 정보도 함께 가져옵니다.
        
        Args:
            query: 검색할 채널명
            max_results: 최대 결과 수 (기본값: 10)
            
        Returns:
            상세 정보가 포함된 채널 정보 리스트
            
        Raises:
            ValueError: 검색어가 비어있을 경우
            YouTubeAPIError: YouTube API 호출 실패 시
        """
        # 기본 검색 결과 가져오기
        channels = self.search_channels(query, max_results)
        
        # 각 채널의 상세 정보 가져오기
        detailed_channels = []
        for channel in channels:
            try:
                channel_details = self.youtube_client.get_channel_details(
                    channel['channel_id']
                )
                
                # 기본 정보와 상세 정보를 병합
                merged_channel = {**channel, **channel_details}
                detailed_channels.append(merged_channel)
                
            except YouTubeAPIError as e:
                logger.warning(f"Failed to get details for channel {channel['channel_id']}: {e}")
                # 상세 정보를 가져올 수 없어도 기본 정보는 포함
                detailed_channels.append(channel)
        
        logger.info(f"Retrieved detailed information for {len(detailed_channels)} channels")
        return detailed_channels
    
    def find_exact_channel_match(
        self, 
        query: str
    ) -> Optional[Dict[str, Any]]:
        """
        정확히 일치하는 채널을 찾습니다.
        
        Args:
            query: 검색할 채널명
            
        Returns:
            정확히 일치하는 채널 정보 (없으면 None)
            
        Raises:
            ValueError: 검색어가 비어있을 경우
            YouTubeAPIError: YouTube API 호출 실패 시
        """
        channels = self.search_channels(query, max_results=20)
        
        # 정확히 일치하는 채널 찾기
        for channel in channels:
            if channel['title'].lower() == query.lower().strip():
                logger.info(f"Found exact match for query: {query}")
                return channel
        
        logger.info(f"No exact match found for query: {query}")
        return None
    
    def search_channels_by_keywords(
        self, 
        keywords: List[str], 
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        여러 키워드로 채널을 검색합니다.
        
        Args:
            keywords: 검색할 키워드 리스트
            max_results: 최대 결과 수 (기본값: 10)
            
        Returns:
            검색된 채널 정보 리스트
            
        Raises:
            ValueError: 키워드 리스트가 비어있을 경우
            YouTubeAPIError: YouTube API 호출 실패 시
        """
        if not keywords:
            raise ValueError("Keywords list cannot be empty")
        
        # 키워드들을 공백으로 연결하여 검색
        query = " ".join(keywords)
        return self.search_channels(query, max_results) 