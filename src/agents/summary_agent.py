"""
요약 에이전트

YouTube 자막을 받아서 한국어로 똑똑한 요약을 생성하는 특화된 에이전트입니다.
BaseAgent를 상속받아 LangGraph 기반으로 구현됩니다.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.agents.base_agent import BaseAgent, AgentConfig, AgentState
from src.services.models import TranscriptSegment, VideoInfo
from src.utils.exceptions import ConfigurationError, ProcessingError
from src.utils.logger import get_project_logger
from src.utils.langfuse_utils import trace_agent_execution

logger = get_project_logger(__name__)


@dataclass
class SummaryAgentConfig(AgentConfig):
    """
    요약 에이전트 설정
    
    BaseAgent의 AgentConfig를 확장하여 요약 관련 설정을 추가합니다.
    """
    
    # BaseAgent 기본 설정 오버라이드
    enable_memory: bool = False  # 기본적으로 메모리 비활성화
    
    # 요약 관련 설정
    max_summary_length: int = 300
    include_timestamps: bool = True
    summary_language: str = "korean"
    summary_style: str = "balanced"  # brief, balanced, detailed
    
    # 타임스탬프 관련 설정
    key_moments_count: int = 3
    timestamp_format: str = "[MM:SS]"  # [MM:SS], [HH:MM:SS], MM분 SS초
    identify_key_moments: bool = True
    key_moments_criteria: List[str] = field(default_factory=lambda: [
        "new_topic", "important_concept", "conclusion"
    ])
    
    # 고급 설정
    use_video_context: bool = True
    include_video_metadata: bool = True
    preserve_key_terms: bool = True
    include_examples: bool = False
    adaptive_length: bool = True
    
    # 빈 자막 처리
    handle_empty_transcript: bool = True
    fallback_to_video_info: bool = True
    min_transcript_length: int = 10
    handle_short_transcript: bool = True
    
    def __post_init__(self):
        """설정 검증"""
        super().__post_init__()
        
        # 요약 길이 검증
        if self.max_summary_length <= 0:
            raise ConfigurationError("Max summary length must be positive")
        
        # 스타일 검증
        valid_styles = ["brief", "balanced", "detailed"]
        if self.summary_style not in valid_styles:
            raise ConfigurationError(f"Summary style must be one of {valid_styles}")
        
        # 언어 검증
        valid_languages = ["korean", "english", "auto"]
        if self.summary_language not in valid_languages:
            raise ConfigurationError(f"Summary language must be one of {valid_languages}")
        
        # key_moments_count 검증
        if self.key_moments_count < 0 or self.key_moments_count > 10:
            raise ConfigurationError("Key moments count must be between 0 and 10")
        
        # 기본 system_prompt 설정 (없는 경우)
        if not self.system_prompt:
            self.system_prompt = self._create_default_system_prompt()
    
    def _create_default_system_prompt(self) -> str:
        """기본 시스템 프롬프트 생성"""
        language_instruction = {
            "korean": "한국어로 답변하세요.",
            "english": "Answer in English.",
            "auto": "Use the same language as the input content."
        }.get(self.summary_language, "한국어로 답변하세요.")
        
        style_instruction = {
            "brief": "간결하고 핵심적인 요약을 제공하세요.",
            "balanced": "적절한 길이의 균형잡힌 요약을 제공하세요.",
            "detailed": "상세하고 포괄적인 요약을 제공하세요."
        }.get(self.summary_style, "적절한 길이의 균형잡힌 요약을 제공하세요.")
        
        return f"""
        당신은 YouTube 영상 자막을 분석하여 똑똑한 요약을 생성하는 전문 AI 어시스턴트입니다.

        주요 역할:
        1. YouTube 영상의 자막과 메타데이터를 분석하여 핵심 내용을 파악합니다.
        2. 사용자가 이해하기 쉽고 유용한 요약을 생성합니다.
        3. 중요한 타임스탬프와 핵심 순간들을 식별합니다.
        4. 영상의 주제와 맥락을 고려하여 맞춤형 요약을 제공합니다.

        요약 가이드라인:
        - {language_instruction}
        - {style_instruction}
        - 최대 {self.max_summary_length}자 이내로 작성하세요.
        - 핵심 개념과 중요한 용어는 보존하세요.
        - 논리적이고 구조화된 흐름을 유지하세요.
        - 타임스탬프를 포함하여 특정 순간을 참조할 수 있도록 하세요.
        
        출력 형식:
        - 명확하고 읽기 쉬운 문단 형태
        - 주요 포인트별로 구분
        - 필요시 타임스탬프 표시: {self.timestamp_format}
        
        항상 정확하고 유용하며 사용자 친화적인 요약을 제공하세요.
        """.strip()


class SummaryAgent(BaseAgent):
    """
    요약 에이전트
    
    YouTube 자막을 받아서 한국어로 똑똑한 요약을 생성하는 특화된 에이전트입니다.
    BaseAgent를 상속받아 LangGraph 기반으로 구현됩니다.
    """
    
    def __init__(self, config: SummaryAgentConfig):
        """
        요약 에이전트 초기화
        
        Args:
            config: SummaryAgentConfig 설정
        """
        super().__init__(config)
        self.config: SummaryAgentConfig = config
        
        # 요약 도구 추가
        self.tools.extend([
            self._create_format_timestamp_tool(),
            self._create_extract_key_moments_tool(),
            self._create_analyze_video_context_tool()
        ])
        
        logger.info(f"SummaryAgent '{config.agent_name}' initialized with style: {config.summary_style}")
    
    def _create_format_timestamp_tool(self):
        """타임스탬프 포맷팅 도구 생성"""
        
        @tool
        def format_timestamp(seconds: float, format_type: str = None) -> str:
            """
            초를 포맷된 타임스탬프 문자열로 변환합니다.
            
            Args:
                seconds: 변환할 초 단위 시간
                format_type: 포맷 타입 (기본값은 config에서 가져옴)
                
            Returns:
                포맷된 타임스탬프 문자열
            """
            if format_type is None:
                format_type = self.config.timestamp_format
            
            if format_type == "[MM:SS]":
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                return f"[{minutes:02d}:{secs:02d}]"
            elif format_type == "[HH:MM:SS]":
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
            elif format_type == "MM분 SS초":
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                if minutes > 0:
                    return f"{minutes}분 {secs}초"
                else:
                    return f"{secs}초"
            else:
                return str(seconds)
        
        return format_timestamp
    
    def _create_extract_key_moments_tool(self):
        """핵심 순간 추출 도구 생성"""
        
        @tool
        def extract_key_moments(
            transcript_segments: List[Dict], 
            max_moments: int = 3
        ) -> List[Dict[str, Any]]:
            """
            자막 세그먼트에서 핵심 순간들을 추출합니다.
            
            Args:
                transcript_segments: 자막 세그먼트 리스트
                max_moments: 최대 추출할 순간 수
                
            Returns:
                핵심 순간들의 리스트
            """
            if not transcript_segments:
                return []
            
            # 단순한 키워드 기반 중요도 계산
            important_keywords = [
                "결론", "요약", "중요", "핵심", "포인트", "결과",
                "따라서", "그러므로", "결국", "마지막으로",
                "first", "second", "finally", "important", "key"
            ]
            
            scored_segments = []
            for segment in transcript_segments:
                text = segment.get('text', '')
                score = 0.0
                
                # 키워드 점수
                for keyword in important_keywords:
                    if keyword in text.lower():
                        score += 1.0
                
                # 길이 점수 (적당한 길이가 좋음)
                text_length = len(text)
                if 50 <= text_length <= 200:
                    score += 0.5
                
                # 위치 점수 (시작과 끝 부분이 중요)
                total_segments = len(transcript_segments)
                segment_index = transcript_segments.index(segment)
                if segment_index < total_segments * 0.2 or segment_index > total_segments * 0.8:
                    score += 0.3
                
                scored_segments.append({
                    'time': segment.get('start_time', 0.0),
                    'text': text,
                    'importance_score': min(score, 1.0),  # 1.0으로 제한
                    'segment_index': segment_index,
                    'moment_type': 'important_concept'  # 기본값
                })
            
            # 점수순으로 정렬하고 상위 항목만 반환
            scored_segments.sort(key=lambda x: x['importance_score'], reverse=True)
            return scored_segments[:max_moments]
        
        return extract_key_moments
    
    def _create_analyze_video_context_tool(self):
        """영상 컨텍스트 분석 도구 생성"""
        
        @tool
        def analyze_video_context(video_info: Dict[str, Any]) -> Dict[str, Any]:
            """
            영상 정보를 분석하여 컨텍스트를 제공합니다.
            
            Args:
                video_info: VideoInfo 딕셔너리
                
            Returns:
                분석된 컨텍스트 정보
            """
            context = {
                'title': video_info.get('title', ''),
                'description': video_info.get('description', ''),
                'tags': video_info.get('tags', []),
                'category': 'unknown',
                'estimated_topic': 'general',
                'content_type': 'educational'  # 기본값
            }
            
            # 태그에서 주제 추정
            tags = [tag.lower() for tag in context['tags']]
            tech_keywords = ['ai', '인공지능', '기술', 'tech', 'programming', '프로그래밍']
            edu_keywords = ['교육', 'education', '강의', 'tutorial', '튜토리얼']
            
            if any(keyword in ' '.join(tags) for keyword in tech_keywords):
                context['estimated_topic'] = 'technology'
            elif any(keyword in ' '.join(tags) for keyword in edu_keywords):
                context['content_type'] = 'educational'
            
            return context
        
        return analyze_video_context
    
    def generate_summary(
        self,
        video_info: VideoInfo,
        transcript_segments: List[TranscriptSegment]
    ) -> Dict[str, Any]:
        """
        비동기가 아닌 일반 요약 생성
        
        Args:
            video_info: 영상 정보
            transcript_segments: 자막 세그먼트들
            
        Returns:
            생성된 요약 결과
        """
        start_time = time.time()
        
        try:
            # 빈 자막 처리
            if not transcript_segments:
                return self._handle_empty_transcript(video_info)
            
            # 매우 짧은 자막 처리
            if len(transcript_segments) == 1 and self.config.handle_short_transcript:
                return self._handle_short_transcript(video_info, transcript_segments[0])
            
            # 요약 생성 메시지 준비
            messages = self._prepare_summary_messages(video_info, transcript_segments)
            
            # BaseAgent의 invoke 사용
            agent_input = {
                'messages': messages
            }
            
            result = self.invoke(agent_input)
            
            # 결과 후처리
            summary_result = self._process_summary_result(
                result, video_info, transcript_segments, start_time
            )
            
            logger.info(f"Summary generated successfully for video {video_info.video_id}")
            return summary_result
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            raise ProcessingError(f"Failed to generate summary: {e}")
    
    async def agenerate_summary(
        self,
        video_info: VideoInfo,
        transcript_segments: List[TranscriptSegment]
    ) -> Dict[str, Any]:
        """
        비동기 요약 생성
        
        Args:
            video_info: 영상 정보
            transcript_segments: 자막 세그먼트들
            
        Returns:
            생성된 요약 결과
        """
        start_time = time.time()
        
        try:
            # 빈 자막 처리
            if not transcript_segments:
                return self._handle_empty_transcript(video_info)
            
            # 요약 생성 메시지 준비
            messages = self._prepare_summary_messages(video_info, transcript_segments)
            
            # BaseAgent의 ainvoke 사용
            agent_input = {
                'messages': messages
            }
            
            result = await self.ainvoke(agent_input)
            
            # 결과 후처리
            summary_result = self._process_summary_result(
                result, video_info, transcript_segments, start_time, execution_type="async"
            )
            
            logger.info(f"Async summary generated successfully for video {video_info.video_id}")
            return summary_result
            
        except Exception as e:
            logger.error(f"Failed to generate async summary: {e}")
            raise ProcessingError(f"Failed to generate async summary: {e}")
    
    def generate_summary_with_memory(
        self,
        video_info: VideoInfo,
        transcript_segments: List[TranscriptSegment],
        thread_id: str
    ) -> Dict[str, Any]:
        """
        메모리를 사용한 요약 생성
        
        Args:
            video_info: 영상 정보
            transcript_segments: 자막 세그먼트들
            thread_id: 스레드 ID
            
        Returns:
            생성된 요약 결과
        """
        start_time = time.time()
        
        try:
            # 요약 생성 메시지 준비
            messages = self._prepare_summary_messages(video_info, transcript_segments)
            
            # BaseAgent의 invoke_with_thread 사용
            agent_input = {
                'messages': messages
            }
            
            result = self.invoke_with_thread(agent_input, thread_id)
            
            # 결과 후처리
            summary_result = self._process_summary_result(
                result, video_info, transcript_segments, start_time
            )
            
            # 메모리 관련 메타데이터 추가
            summary_result['metadata']['thread_id'] = thread_id
            summary_result['metadata']['previous_summary_available'] = True
            
            logger.info(f"Memory-based summary generated for video {video_info.video_id}")
            return summary_result
            
        except Exception as e:
            logger.error(f"Failed to generate memory-based summary: {e}")
            raise ProcessingError(f"Failed to generate memory-based summary: {e}")
    
    def _prepare_summary_messages(
        self,
        video_info: VideoInfo,
        transcript_segments: List[TranscriptSegment]
    ) -> List[Any]:
        """요약 생성을 위한 메시지 준비"""
        
        # 자막 텍스트 결합
        transcript_text = "\n".join([
            f"[{seg.get_formatted_time()}] {seg.text}"
            for seg in transcript_segments
        ])
        
        # 영상 컨텍스트 정보 준비
        context_info = ""
        if self.config.use_video_context:
            context_info = f"""
영상 정보:
- 제목: {video_info.title}
- 설명: {video_info.description[:200]}...
- 태그: {', '.join(video_info.tags[:5]) if video_info.tags else '없음'}
- 길이: {video_info.duration_readable or '알 수 없음'}
- 조회수: {video_info.get_formatted_stats().get('views', '알 수 없음')}
"""
        
        # 요약 요청 메시지
        user_message = f"""
다음 YouTube 영상의 자막을 분석하여 요약해주세요:

{context_info}

자막 내용:
{transcript_text}

요청 사항:
- 요약 스타일: {self.config.summary_style}
- 최대 길이: {self.config.max_summary_length}자
- 언어: {self.config.summary_language}
- 타임스탬프 포함: {'예' if self.config.include_timestamps else '아니오'}
- 핵심 순간 수: {self.config.key_moments_count}개

핵심 내용을 놓치지 않으면서도 이해하기 쉬운 요약을 생성해주세요.
""".strip()
        
        return [HumanMessage(content=user_message)]
    
    def _handle_empty_transcript(self, video_info: VideoInfo) -> Dict[str, Any]:
        """빈 자막 처리"""
        
        if not self.config.handle_empty_transcript:
            raise ProcessingError("Transcript is empty and empty handling is disabled")
        
        if self.config.fallback_to_video_info:
            # 영상 정보만으로 기본 요약 생성
            fallback_summary = f"""
이 영상은 '{video_info.title}'라는 제목의 콘텐츠입니다.
{video_info.description[:150] if video_info.description else '상세 설명이 제공되지 않았습니다.'}

주요 정보:
- 채널: {video_info.channel_title or '알 수 없음'}
- 게시일: {video_info.published_at[:10] if video_info.published_at else '알 수 없음'}
- 길이: {video_info.duration_readable or '알 수 없음'}
- 태그: {', '.join(video_info.tags[:5]) if video_info.tags else '없음'}

※ 이 요약은 자막이 제공되지 않아 영상 메타데이터만을 기반으로 생성되었습니다.
            """.strip()
            
            return {
                'summary': fallback_summary,
                'video_info': video_info.to_dict(),
                'key_timestamps': [],
                'timestamp_segments': [],
                'warning': '자막이 제공되지 않았습니다. 영상 정보만을 기반으로 요약이 생성되었습니다.',
                'metadata': {
                    'agent_name': self.config.agent_name,
                    'summary_language': self.config.summary_language,
                    'is_fallback_summary': True,
                    'original_segments_count': 0,
                    'processing_time': 0.0
                }
            }
        else:
            raise ProcessingError("Empty transcript and fallback is disabled")
    
    def _handle_short_transcript(
        self, 
        video_info: VideoInfo, 
        segment: TranscriptSegment
    ) -> Dict[str, Any]:
        """매우 짧은 자막 처리"""
        
        short_summary = f"""
'{video_info.title}' 영상의 짧은 내용입니다:

{segment.text}

이 영상은 약 {segment.get_formatted_time('korean')}에 걸쳐 간단한 메시지를 전달합니다.
        """.strip()
        
        return {
            'summary': short_summary,
            'video_info': video_info.to_dict(),
            'key_timestamps': [{
                'time': segment.start_time,
                'text': segment.text,
                'importance_score': 1.0,
                'moment_type': 'complete_content'
            }],
            'timestamp_segments': [segment.to_dict()],
            'metadata': {
                'agent_name': self.config.agent_name,
                'summary_language': self.config.summary_language,
                'is_short_transcript': True,
                'original_segments_count': 1,
                'processing_time': 0.0
            }
        }
    
    def _process_summary_result(
        self,
        agent_result: Dict[str, Any],
        video_info: VideoInfo,
        transcript_segments: List[TranscriptSegment],
        start_time: float,
        execution_type: str = "sync"
    ) -> Dict[str, Any]:
        """에이전트 결과를 요약 결과로 변환"""
        
        processing_time = time.time() - start_time
        
        # 에이전트 응답에서 실제 요약 텍스트 추출
        messages = agent_result.get('messages', [])
        summary_text = ""
        
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                summary_text = last_message.content
            elif isinstance(last_message, dict):
                summary_text = last_message.get('content', '')
        
        # 핵심 타임스탬프 추출 (설정에 따라)
        key_timestamps = []
        if self.config.include_timestamps and self.config.key_moments_count > 0:
            key_timestamps = self._extract_key_timestamps(transcript_segments)
        
        # 결과 구성
        result = {
            'summary': summary_text,
            'video_info': video_info.to_dict(),
            'key_timestamps': key_timestamps,
            'timestamp_segments': [seg.to_dict() for seg in transcript_segments],
            'metadata': {
                'agent_name': self.config.agent_name,
                'model_used': self.config.model_name,
                'summary_language': self.config.summary_language,
                'summary_style': self.config.summary_style,
                'max_length': self.config.max_summary_length,
                'include_timestamps': self.config.include_timestamps,
                'key_moments_count': len(key_timestamps),
                'original_segments_count': len(transcript_segments),
                'processing_time': round(processing_time, 2),
                'execution_type': execution_type,
                'timestamp_format': self.config.timestamp_format
            }
        }
        
        # 영상 메타데이터 추가 (설정에 따라)
        if self.config.include_video_metadata:
            result['metadata'].update({
                'video_title': video_info.title,
                'video_duration': video_info.duration,
                'channel_id': video_info.channel_id,
                'published_at': video_info.published_at
            })
        
        return result
    
    def _extract_key_timestamps(
        self, 
        transcript_segments: List[TranscriptSegment]
    ) -> List[Dict[str, Any]]:
        """핵심 타임스탬프 추출"""
        
        # 단순한 추출 로직 (추후 개선 가능)
        key_moments = []
        segment_count = len(transcript_segments)
        
        if segment_count <= self.config.key_moments_count:
            # 세그먼트가 적으면 모두 포함
            key_moments = transcript_segments
        else:
            # 균등하게 분포된 핵심 순간들 선택
            step = segment_count // self.config.key_moments_count
            for i in range(0, segment_count, step):
                if len(key_moments) < self.config.key_moments_count:
                    key_moments.append(transcript_segments[i])
        
        return [{
            'time': segment.start_time,
            'text': segment.text,
            'importance_score': 0.8,  # 기본 점수
            'moment_type': 'important_concept'
        } for segment in key_moments] 