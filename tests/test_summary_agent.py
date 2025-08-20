"""
요약 에이전트 테스트

YouTube 자막을 받아서 한국어로 똑똑한 요약을 생성하는 
SummaryAgent의 기능을 테스트합니다.
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.agents.base_agent import BaseAgent, AgentConfig
from src.services.models import TranscriptSegment, VideoInfo
from src.utils.exceptions import ConfigurationError
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)


@pytest.fixture
def sample_video_info():
    """테스트용 영상 정보"""
    return VideoInfo(
        video_id="dQw4w9WgXcQ",
        title="테스트 영상: AI 기술 설명",
        description="인공지능 기술에 대한 상세한 설명 영상입니다.",
        channel_id="UC1234567890",
        published_at="2023-01-15T12:00:00Z",
        thumbnail_url="https://example.com/thumbnail.jpg",
        duration="PT15M30S",
        view_count=50000,
        like_count=1000,
        tags=["AI", "기술", "설명", "교육"]
    )


@pytest.fixture
def sample_transcript_segments():
    """테스트용 자막 세그먼트들"""
    return [
        TranscriptSegment(
            text="안녕하세요. 오늘은 인공지능에 대해 말씀드리겠습니다.",
            start_time=0.0,
            end_time=3.5,
            segment_id="seg_001"
        ),
        TranscriptSegment(
            text="인공지능은 컴퓨터가 인간처럼 학습하고 판단할 수 있는 기술입니다.",
            start_time=3.5,
            end_time=8.2,
            segment_id="seg_002"
        ),
        TranscriptSegment(
            text="머신러닝과 딥러닝이 핵심 기술이라고 할 수 있습니다.",
            start_time=8.2,
            end_time=12.1,
            segment_id="seg_003"
        ),
        TranscriptSegment(
            text="특히 자연어 처리와 컴퓨터 비전 분야에서 큰 발전을 보이고 있습니다.",
            start_time=12.1,
            end_time=17.8,
            segment_id="seg_004"
        ),
        TranscriptSegment(
            text="앞으로 AI 기술은 더욱 발전하여 우리 생활 전반에 영향을 미칠 것입니다.",
            start_time=17.8,
            end_time=23.5,
            segment_id="seg_005"
        )
    ]


@pytest.fixture
def empty_transcript_segments():
    """빈 자막 세그먼트"""
    return []


@pytest.fixture
def short_transcript_segments():
    """매우 짧은 자막 세그먼트"""
    return [
        TranscriptSegment(
            text="안녕하세요.",
            start_time=0.0,
            end_time=1.5,
            segment_id="short_001"
        )
    ]


class TestSummaryAgentInitialization:
    """요약 에이전트 초기화 테스트"""
    
    def test_summary_agent_initialization(self):
        """요약 에이전트 기본 초기화 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="test_summary_agent",
            model_name="gpt-4o-mini",
            max_summary_length=300,
            include_timestamps=True,
            summary_language="korean"
        )
        
        agent = SummaryAgent(config)
        
        assert agent.config.agent_name == "test_summary_agent"
        assert agent.config.max_summary_length == 300
        assert agent.config.include_timestamps is True
        assert agent.config.summary_language == "korean"
        assert isinstance(agent, BaseAgent)
        assert agent.state_graph is not None
    
    def test_summary_agent_with_minimal_config(self):
        """최소 설정으로 요약 에이전트 초기화"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="minimal_summary_agent"
        )
        
        agent = SummaryAgent(config)
        
        assert agent.config.agent_name == "minimal_summary_agent"
        assert agent.config.max_summary_length > 0  # 기본값 존재
        assert agent.config.summary_language is not None  # 기본값 존재
        assert agent.state_graph is not None
    
    def test_summary_agent_invalid_config(self):
        """잘못된 설정으로 초기화 실패 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        with pytest.raises(ConfigurationError):
            config = SummaryAgentConfig(
                agent_name="",  # 빈 이름
                max_summary_length=-100  # 음수 길이
            )
            SummaryAgent(config)


class TestGenerateSummaryFromTranscript:
    """자막 기반 요약 생성 테스트"""
    
    @patch('src.agents.base_agent.ChatGoogleGenerativeAI')
    def test_generate_summary_from_transcript(self, mock_gemini, sample_video_info, sample_transcript_segments):
        """자막에서 요약을 생성하는 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        from langchain_core.messages import AIMessage
        
        # Mock LLM 응답 설정
        mock_llm_instance = Mock()
        mock_gemini.return_value = mock_llm_instance
        
        # 한국어 요약 응답 Mock
        mock_summary = """
        이 영상은 인공지능 기술에 대한 포괄적인 설명을 다룹니다. 
        
        [00:00] 인공지능 소개 - 컴퓨터가 인간처럼 학습하고 판단하는 기술을 설명합니다.
        [00:08] 핵심 기술 - 머신러닝과 딥러닝이 AI의 핵심 기술임을 강조합니다.  
        [00:12] 응용 분야 - 자연어 처리와 컴퓨터 비전 분야의 발전을 소개합니다.
        [00:17] 미래 전망 - AI 기술이 우리 생활 전반에 미칠 영향을 논의합니다.
        
        영상은 AI 기술의 현재와 미래를 균형있게 다루며, 기술적 내용을 이해하기 쉽게 설명합니다.
        """.strip()
        
        # BaseAgent의 invoke 메서드 Mock
        mock_invoke_result = {
            'messages': [AIMessage(content=mock_summary)]
        }
        
        config = SummaryAgentConfig(
            agent_name="transcript_summary_agent",
            model_name="gpt-4o-mini",
            max_summary_length=200,
            include_timestamps=True,
            summary_language="korean"
        )
        
        agent = SummaryAgent(config)
        
        # invoke 메서드를 Mock으로 패치
        with patch.object(agent, 'invoke', return_value=mock_invoke_result) as mock_invoke:
            # 요약 생성
            result = agent.generate_summary(
                video_info=sample_video_info,
                transcript_segments=sample_transcript_segments
            )
            
            # invoke가 호출되었는지 확인
            mock_invoke.assert_called_once()
            
            # 호출된 인자 확인
            call_args = mock_invoke.call_args[0][0]
            assert 'messages' in call_args
            assert len(call_args['messages']) > 0
        
        # 결과 검증
        assert "summary" in result
        assert "video_info" in result
        assert "metadata" in result
        assert "key_timestamps" in result
        
        summary = result["summary"]
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "인공지능" in summary or "AI" in summary  # 핵심 주제 포함
        
        metadata = result["metadata"]
        assert metadata["agent_name"] == "transcript_summary_agent"
        assert metadata["summary_language"] == "korean"
        assert metadata["original_segments_count"] == 5
        assert metadata["execution_type"] == "sync"
    
    def test_generate_summary_with_video_context(self, sample_video_info, sample_transcript_segments):
        """영상 정보를 활용한 컨텍스트 포함 요약 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="context_summary_agent",
            use_video_context=True,
            include_video_metadata=True
        )
        
        agent = SummaryAgent(config)
        agent.compile_graph()
        
        result = agent.generate_summary(
            video_info=sample_video_info,
            transcript_segments=sample_transcript_segments
        )
        
        # 영상 정보가 요약에 반영되었는지 확인
        summary = result["summary"]
        metadata = result["metadata"]
        
        assert "video_title" in metadata
        assert metadata["video_title"] == "테스트 영상: AI 기술 설명"
        assert metadata["video_duration"] == "PT15M30S"
        assert "channel_id" in metadata
    
    def test_generate_summary_preserves_key_information(self, sample_video_info, sample_transcript_segments):
        """요약에서 핵심 정보 보존 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="key_info_summary_agent",
            preserve_key_terms=True,
            max_summary_length=150
        )
        
        agent = SummaryAgent(config)
        agent.compile_graph()
        
        result = agent.generate_summary(
            video_info=sample_video_info,
            transcript_segments=sample_transcript_segments
        )
        
        summary = result["summary"]
        
        # 핵심 기술 용어들이 보존되었는지 확인
        key_terms = ["인공지능", "머신러닝", "딥러닝", "자연어 처리", "컴퓨터 비전"]
        found_terms = [term for term in key_terms if term in summary]
        
        assert len(found_terms) >= 2  # 최소 2개 이상의 핵심 용어 포함


class TestSummaryLengthControl:
    """요약 길이 제어 테스트"""
    
    def test_summary_length_control_short(self, sample_video_info, sample_transcript_segments):
        """짧은 요약 생성 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="short_summary_agent",
            max_summary_length=100,  # 매우 짧게 설정
            summary_style="brief"
        )
        
        agent = SummaryAgent(config)
        agent.compile_graph()
        
        result = agent.generate_summary(
            video_info=sample_video_info,
            transcript_segments=sample_transcript_segments
        )
        
        summary = result["summary"]
        
        # 길이 제한 준수 확인 (약간의 여유 허용)
        assert len(summary) <= 150  # 약간의 마진 허용
        assert len(summary) > 20   # 최소 길이는 확보
        
        metadata = result["metadata"]
        assert metadata["summary_style"] == "brief"
        assert metadata["max_length"] == 100
    
    def test_summary_length_control_detailed(self, sample_video_info, sample_transcript_segments):
        """상세한 요약 생성 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="detailed_summary_agent",
            max_summary_length=500,  # 길게 설정
            summary_style="detailed",
            include_examples=True
        )
        
        agent = SummaryAgent(config)
        agent.compile_graph()
        
        result = agent.generate_summary(
            video_info=sample_video_info,
            transcript_segments=sample_transcript_segments
        )
        
        summary = result["summary"]
        
        # 상세한 요약 검증
        assert len(summary) >= 150  # 최소 길이
        assert len(summary) <= 650  # 최대 길이 (마진 포함)
        
        metadata = result["metadata"]
        assert metadata["summary_style"] == "detailed"
        assert metadata["include_examples"] is True
    
    def test_adaptive_summary_length(self, sample_video_info, short_transcript_segments):
        """자막 길이에 따른 적응적 요약 길이 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="adaptive_summary_agent",
            adaptive_length=True,
            max_summary_length=300
        )
        
        agent = SummaryAgent(config)
        agent.compile_graph()
        
        result = agent.generate_summary(
            video_info=sample_video_info,
            transcript_segments=short_transcript_segments
        )
        
        summary = result["summary"]
        metadata = result["metadata"]
        
        # 짧은 자막에 대해서는 더 짧은 요약 생성
        assert len(summary) <= 150  # 원본이 짧으면 요약도 짧게
        assert metadata["adaptive_length"] is True
        assert metadata["original_length"] == len(short_transcript_segments)


class TestSummaryWithKeyTimestamps:
    """주요 타임스탬프 포함 요약 테스트"""
    
    def test_summary_with_key_timestamps(self, sample_video_info, sample_transcript_segments):
        """주요 타임스탬프를 포함한 요약 생성 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="timestamp_summary_agent",
            include_timestamps=True,
            key_moments_count=3,
            timestamp_format="[MM:SS]"
        )
        
        agent = SummaryAgent(config)
        agent.compile_graph()
        
        result = agent.generate_summary(
            video_info=sample_video_info,
            transcript_segments=sample_transcript_segments
        )
        
        # 결과 검증
        assert "summary" in result
        assert "key_timestamps" in result
        assert "timestamp_segments" in result
        
        key_timestamps = result["key_timestamps"]
        assert isinstance(key_timestamps, list)
        assert len(key_timestamps) <= 3  # key_moments_count 준수
        
        # 각 타임스탬프 항목 검증
        for timestamp_item in key_timestamps:
            assert "time" in timestamp_item
            assert "text" in timestamp_item
            assert "importance_score" in timestamp_item
            assert isinstance(timestamp_item["time"], (int, float))
            assert isinstance(timestamp_item["text"], str)
            assert 0 <= timestamp_item["importance_score"] <= 1.0
    
    def test_timestamp_format_variations(self, sample_video_info, sample_transcript_segments):
        """다양한 타임스탬프 형식 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        test_formats = ["[MM:SS]", "[HH:MM:SS]", "MM분 SS초"]
        
        for format_style in test_formats:
            config = SummaryAgentConfig(
                agent_name=f"timestamp_{format_style.replace(':', '_')}_agent",
                include_timestamps=True,
                timestamp_format=format_style
            )
            
            agent = SummaryAgent(config)
            agent.compile_graph()
            
            result = agent.generate_summary(
                video_info=sample_video_info,
                transcript_segments=sample_transcript_segments
            )
            
            summary = result["summary"]
            metadata = result["metadata"]
            
            assert metadata["timestamp_format"] == format_style
            
            # 타임스탬프가 요약에 포함되었는지 확인
            if "[" in format_style:
                assert "[" in summary and "]" in summary
            elif "분" in format_style:
                assert "분" in summary and "초" in summary
    
    def test_key_moments_identification(self, sample_video_info, sample_transcript_segments):
        """핵심 순간 식별 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="key_moments_agent",
            include_timestamps=True,
            identify_key_moments=True,
            key_moments_criteria=["new_topic", "important_concept", "conclusion"]
        )
        
        agent = SummaryAgent(config)
        agent.compile_graph()
        
        result = agent.generate_summary(
            video_info=sample_video_info,
            transcript_segments=sample_transcript_segments
        )
        
        key_timestamps = result["key_timestamps"]
        metadata = result["metadata"]
        
        # 핵심 순간 식별 메타데이터 확인
        assert metadata["identify_key_moments"] is True
        assert "key_moments_criteria" in metadata
        
        # 각 타임스탬프가 식별된 기준을 포함하는지 확인
        for timestamp_item in key_timestamps:
            assert "moment_type" in timestamp_item
            assert timestamp_item["moment_type"] in ["new_topic", "important_concept", "conclusion", "other"]


class TestEmptyTranscriptHandling:
    """빈 자막 처리 테스트"""
    
    def test_empty_transcript_handling(self, sample_video_info, empty_transcript_segments):
        """빈 자막에 대한 처리 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="empty_transcript_agent",
            handle_empty_transcript=True,
            fallback_to_video_info=True
        )
        
        agent = SummaryAgent(config)
        agent.compile_graph()
        
        result = agent.generate_summary(
            video_info=sample_video_info,
            transcript_segments=empty_transcript_segments
        )
        
        # 빈 자막에 대한 적절한 처리 확인
        assert "summary" in result
        assert "warning" in result
        
        summary = result["summary"]
        warning = result["warning"]
        
        # 영상 정보를 활용한 fallback 요약이 생성되었는지 확인
        assert len(summary) > 0
        assert "자막이 제공되지 않았습니다" in warning or "transcript not available" in warning.lower()
        
        # 영상 메타데이터를 활용한 요약인지 확인
        assert sample_video_info.title in summary or "AI" in summary or "기술" in summary
    
    def test_empty_transcript_without_fallback(self, sample_video_info, empty_transcript_segments):
        """fallback 없이 빈 자막 처리 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        from src.utils.exceptions import ProcessingError
        
        config = SummaryAgentConfig(
            agent_name="no_fallback_agent",
            handle_empty_transcript=False,
            fallback_to_video_info=False
        )
        
        agent = SummaryAgent(config)
        agent.compile_graph()
        
        # 빈 자막에서 fallback 없이는 에러 발생해야 함
        with pytest.raises((ProcessingError, ValueError)):
            agent.generate_summary(
                video_info=sample_video_info,
                transcript_segments=empty_transcript_segments
            )
    
    def test_very_short_transcript_handling(self, sample_video_info, short_transcript_segments):
        """매우 짧은 자막 처리 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="short_transcript_agent",
            min_transcript_length=10,  # 최소 길이 설정
            handle_short_transcript=True
        )
        
        agent = SummaryAgent(config)
        agent.compile_graph()
        
        result = agent.generate_summary(
            video_info=sample_video_info,
            transcript_segments=short_transcript_segments
        )
        
        summary = result["summary"]
        metadata = result["metadata"]
        
        # 짧은 자막에 대한 적절한 처리 확인
        assert len(summary) > 0
        assert metadata["is_short_transcript"] is True
        assert metadata["original_segments_count"] == 1
        
        # 짧은 자막이지만 유의미한 요약이 생성되었는지 확인
        assert "안녕하세요" in summary or len(summary) >= 10


class TestSummaryAgentIntegration:
    """요약 에이전트 통합 테스트"""
    
    def test_complete_summary_workflow(self, sample_video_info, sample_transcript_segments):
        """완전한 요약 워크플로우 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="complete_workflow_agent",
            max_summary_length=250,
            include_timestamps=True,
            summary_language="korean",
            use_video_context=True,
            key_moments_count=2
        )
        
        agent = SummaryAgent(config)
        agent.compile_graph()
        
        # 전체 워크플로우 실행
        result = agent.generate_summary(
            video_info=sample_video_info,
            transcript_segments=sample_transcript_segments
        )
        
        # 모든 구성 요소가 포함되었는지 확인
        required_keys = ["summary", "key_timestamps", "video_info", "metadata"]
        for key in required_keys:
            assert key in result
        
        # 요약 품질 검증
        summary = result["summary"]
        assert 50 <= len(summary) <= 300  # 적절한 길이
        assert any(term in summary for term in ["인공지능", "AI", "기술", "머신러닝"])  # 핵심 주제 포함
        
        # 타임스탬프 검증
        key_timestamps = result["key_timestamps"]
        assert len(key_timestamps) <= 2  # key_moments_count 준수
        
        # 메타데이터 완성도 검증
        metadata = result["metadata"]
        essential_metadata = ["agent_name", "summary_language", "processing_time", "model_used"]
        for meta_key in essential_metadata:
            assert meta_key in metadata
    
    def test_summary_agent_memory_usage(self, sample_video_info, sample_transcript_segments):
        """요약 에이전트의 메모리 사용 테스트"""
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="memory_test_agent",
            enable_memory=True
        )
        
        agent = SummaryAgent(config)
        agent.compile_graph()
        
        thread_id = "summary_memory_test"
        
        # 첫 번째 요약 생성
        result1 = agent.generate_summary_with_memory(
            video_info=sample_video_info,
            transcript_segments=sample_transcript_segments,
            thread_id=thread_id
        )
        
        # 두 번째 요약 생성 (같은 스레드)
        result2 = agent.generate_summary_with_memory(
            video_info=sample_video_info,
            transcript_segments=sample_transcript_segments[:3],  # 일부만 사용
            thread_id=thread_id
        )
        
        # 메모리를 활용한 연관성 확인
        assert result1["summary"] != result2["summary"]  # 다른 결과
        assert "metadata" in result1 and "metadata" in result2
        
        # 메모리 사용 메타데이터 확인
        metadata2 = result2["metadata"]
        assert "previous_summary_available" in metadata2
        assert metadata2["thread_id"] == thread_id
    
    def test_summary_agent_async_execution(self, sample_video_info, sample_transcript_segments):
        """요약 에이전트 비동기 실행 테스트"""
        import asyncio
        from src.agents.summary_agent import SummaryAgent, SummaryAgentConfig
        
        config = SummaryAgentConfig(
            agent_name="async_summary_agent",
            enable_memory=False
        )
        
        agent = SummaryAgent(config)
        agent.compile_graph()
        
        async def async_summary_test():
            result = await agent.agenerate_summary(
                video_info=sample_video_info,
                transcript_segments=sample_transcript_segments
            )
            return result
        
        # 비동기 실행
        result = asyncio.run(async_summary_test())
        
        # 비동기 실행 결과 검증
        assert "summary" in result
        assert "metadata" in result
        assert len(result["summary"]) > 0
        
        metadata = result["metadata"]
        assert metadata["execution_type"] == "async" 