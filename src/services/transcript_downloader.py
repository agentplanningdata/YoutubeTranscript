"""
자막 다운로드 서비스

YouTube 영상의 자막을 다운로드하고 전처리하는 기능을 제공합니다.
"""

from typing import Optional, List, Dict, Any, Union
import re
from datetime import datetime

from src.utils.exceptions import TranscriptNotFoundError, TranscriptProcessingError
from src.utils.logger import get_project_logger

# youtube_transcript_api import 처리
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    # 테스트 환경에서는 Mock으로 대체
    class YouTubeTranscriptApi:
        @staticmethod
        def get_transcript(video_id, languages=None):
            raise ImportError("youtube_transcript_api not available")
    
    class NoTranscriptFound(Exception):
        pass
    
    class TranscriptsDisabled(Exception):
        pass
    
    class VideoUnavailable(Exception):
        pass
    
    YOUTUBE_TRANSCRIPT_AVAILABLE = False

logger = get_project_logger(__name__)


class TranscriptDownloader:
    """YouTube 영상 자막 다운로드 서비스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        자막 다운로더를 초기화합니다.
        
        Args:
            config: 다운로더 설정 딕셔너리
        """
        self.config = config or {}
        
        # 기본 설정
        self.default_config = {
            'preserve_formatting': True,
            'include_generated': True,
            'timeout_seconds': 30,
            'max_retries': 3,
            'clean_text': True,
            'include_timestamps': True
        }
        
        # 설정 병합
        self.settings = {**self.default_config, **self.config}
        
        logger.info("Transcript downloader initialized")
    
    def download_transcript(
        self, 
        video_id: str, 
        language: str = "ko",
        fallback_languages: Optional[List[str]] = None,
        include_metadata: bool = False
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        YouTube 영상의 자막을 다운로드합니다.
        
        Args:
            video_id: YouTube 영상 ID
            language: 원하는 자막 언어 코드 (기본: "ko")
            fallback_languages: 원하는 언어가 없을 때 대체 언어들
            include_metadata: 메타데이터 포함 여부
            
        Returns:
            자막 데이터 리스트 또는 메타데이터가 포함된 딕셔너리
            
        Raises:
            ValueError: 영상 ID가 비어있을 경우
            TranscriptNotFoundError: 자막을 찾을 수 없을 경우
            TranscriptProcessingError: 자막 처리 중 오류 발생 시
        """
        # 입력 유효성 검사
        if not video_id or (isinstance(video_id, str) and not video_id.strip()):
            raise ValueError("Video ID cannot be empty")
        
        if video_id is None:
            raise ValueError("Video ID cannot be empty")
        
        video_id = video_id.strip()
        
        try:
            # 자막 다운로드 시도
            transcript_data = self._download_with_fallback(
                video_id, language, fallback_languages or []
            )
            
            # 자막 전처리
            processed_transcript = self._process_transcript(transcript_data)
            
            logger.info(f"Successfully downloaded transcript for video {video_id} in language {language}")
            
            if include_metadata:
                return {
                    'transcript': processed_transcript,
                    'metadata': {
                        'video_id': video_id,
                        'language': language,
                        'download_timestamp': datetime.now().isoformat(),
                        'total_segments': len(processed_transcript),
                        'total_duration': self._calculate_total_duration(processed_transcript),
                        'fallback_used': fallback_languages is not None
                    }
                }
            
            return processed_transcript
            
        except TranscriptNotFoundError:
            # TranscriptNotFoundError는 그대로 전파
            raise
        except Exception as e:
            # youtube_transcript_api 관련 예외들을 TranscriptNotFoundError로 변환
            error_message = str(e).lower()
            keywords = ['transcript found', 'not found', 'disabled', 'unavailable']
            if any(keyword in error_message for keyword in keywords):
                logger.error(f"Transcript not found for video {video_id}: {e}")
                raise TranscriptNotFoundError(f"Transcript not found for video {video_id}: {e}", video_id=video_id)
            else:
                logger.error(f"Failed to download transcript for video {video_id}: {e}")
                raise TranscriptProcessingError(f"Failed to download transcript for video {video_id}: {e}")
    
    def _download_with_fallback(
        self, 
        video_id: str, 
        primary_language: str,
        fallback_languages: List[str]
    ) -> List[Dict[str, Any]]:
        """
        주 언어로 자막 다운로드를 시도하고, 실패 시 폴백 언어들을 순서대로 시도합니다.
        
        Args:
            video_id: YouTube 영상 ID
            primary_language: 주 언어 코드
            fallback_languages: 폴백 언어 코드 리스트
            
        Returns:
            자막 데이터 리스트
        """
        languages_to_try = [primary_language] + fallback_languages
        
        for lang in languages_to_try:
            try:
                logger.debug(f"Attempting to download transcript in {lang} for video {video_id}")
                return YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
            except Exception as e:
                # youtube_transcript_api 예외 확인
                error_message = str(e)
                error_lower = error_message.lower()
                keywords = ['transcript found', 'not found', 'disabled', 'unavailable']
                if any(keyword in error_lower for keyword in keywords):
                    if lang == languages_to_try[-1]:  # 마지막 시도인 경우
                        raise
                    continue
                else:
                    # 다른 예외는 즉시 발생
                    raise
        
        # 모든 언어에서 실패한 경우
        raise TranscriptNotFoundError(f"No transcript available in any of the requested languages: {languages_to_try}", video_id=video_id)
    
    def _process_transcript(self, raw_transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        원본 자막 데이터를 처리하고 정제합니다.
        
        Args:
            raw_transcript: 원본 자막 데이터
            
        Returns:
            처리된 자막 데이터
        """
        processed = []
        
        for segment in raw_transcript:
            processed_segment = segment.copy()
            
            # 텍스트 정제
            if self.settings.get('clean_text', True):
                processed_segment['text'] = self._clean_text(segment['text'])
            
            # 타임스탬프 정보 보존
            if self.settings.get('include_timestamps', True):
                processed_segment['start'] = float(segment.get('start', 0.0))
                processed_segment['duration'] = float(segment.get('duration', 0.0))
                processed_segment['end'] = processed_segment['start'] + processed_segment['duration']
            
            processed.append(processed_segment)
        
        return processed
    
    def _clean_text(self, text: str) -> str:
        """
        자막 텍스트를 정제합니다.
        
        Args:
            text: 원본 텍스트
            
        Returns:
            정제된 텍스트
        """
        if not text:
            return text
        
        # 앞뒤 공백 제거
        cleaned = text.strip()
        
        # 연속된 공백을 단일 공백으로 변경
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 개행 문자, 탭 문자를 공백으로 변경
        cleaned = re.sub(r'[\n\r\t]', ' ', cleaned)
        
        # 다시 연속 공백 제거
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 과도한 문장부호 정리 (예: !!! -> !)
        cleaned = re.sub(r'([!?.])\1{2,}', r'\1', cleaned)
        
        # 최종 공백 제거
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _calculate_total_duration(self, transcript: List[Dict[str, Any]]) -> float:
        """
        전체 자막의 총 길이를 계산합니다.
        
        Args:
            transcript: 자막 데이터
            
        Returns:
            총 길이 (초)
        """
        if not transcript:
            return 0.0
        
        try:
            last_segment = transcript[-1]
            return last_segment.get('start', 0.0) + last_segment.get('duration', 0.0)
        except (KeyError, IndexError, TypeError):
            return 0.0
    
    def get_available_languages(self, video_id: str) -> List[str]:
        """
        영상에서 사용 가능한 자막 언어 목록을 조회합니다.
        
        Args:
            video_id: YouTube 영상 ID
            
        Returns:
            사용 가능한 언어 코드 리스트
            
        Raises:
            ValueError: 영상 ID가 비어있을 경우
            TranscriptProcessingError: 언어 목록 조회 실패 시
        """
        if not video_id or (isinstance(video_id, str) and not video_id.strip()):
            raise ValueError("Video ID cannot be empty")
        
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id.strip())
            languages = []
            
            for transcript in transcript_list:
                languages.append(transcript.language_code)
            
            logger.info(f"Found {len(languages)} available languages for video {video_id}")
            return languages
            
        except Exception as e:
            logger.error(f"Failed to get available languages for video {video_id}: {e}")
            raise TranscriptProcessingError(f"Failed to get available languages: {e}")
    
    def is_transcript_available(self, video_id: str, language: str = "ko") -> bool:
        """
        특정 언어의 자막이 사용 가능한지 확인합니다.
        
        Args:
            video_id: YouTube 영상 ID
            language: 확인할 언어 코드
            
        Returns:
            자막 사용 가능 여부
        """
        try:
            available_languages = self.get_available_languages(video_id)
            return language in available_languages
        except Exception:
            return False
    
    def download_multiple_languages(
        self, 
        video_id: str, 
        languages: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        여러 언어의 자막을 한 번에 다운로드합니다.
        
        Args:
            video_id: YouTube 영상 ID
            languages: 다운로드할 언어 코드 리스트
            
        Returns:
            언어별 자막 데이터 딕셔너리
        """
        results = {}
        
        for language in languages:
            try:
                transcript = self.download_transcript(video_id, language)
                results[language] = transcript
                logger.info(f"Successfully downloaded {language} transcript for video {video_id}")
            except (TranscriptNotFoundError, TranscriptProcessingError) as e:
                logger.warning(f"Failed to download {language} transcript for video {video_id}: {e}")
                results[language] = None
        
        return results
    
    def get_transcript_stats(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        자막의 통계 정보를 반환합니다.
        
        Args:
            transcript: 자막 데이터
            
        Returns:
            통계 정보 딕셔너리
        """
        if not transcript:
            return {
                'total_segments': 0,
                'total_duration': 0.0,
                'total_words': 0,
                'average_segment_duration': 0.0,
                'words_per_minute': 0.0
            }
        
        total_segments = len(transcript)
        total_duration = self._calculate_total_duration(transcript)
        
        # 총 단어 수 계산
        total_words = sum(len(segment.get('text', '').split()) for segment in transcript)
        
        # 평균 세그먼트 길이
        average_duration = total_duration / total_segments if total_segments > 0 else 0.0
        
        # 분당 단어 수
        words_per_minute = (total_words / (total_duration / 60)) if total_duration > 0 else 0.0
        
        return {
            'total_segments': total_segments,
            'total_duration': round(total_duration, 2),
            'total_words': total_words,
            'average_segment_duration': round(average_duration, 2),
            'words_per_minute': round(words_per_minute, 1)
        } 