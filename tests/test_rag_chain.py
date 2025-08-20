"""
LCEL RAG 체인 테스트

LangChain Expression Language 기반 RAG 체인의
기능을 테스트합니다.
"""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.agents.rag_chain import RAGChain, RAGConfig, create_rag_chain
from src.services.similarity_search_service import SimilaritySearchService
from src.utils.exceptions import ProcessingError
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)


@pytest.fixture
def mock_similarity_service():
    """Mock 검색 서비스"""
    service = Mock(spec=SimilaritySearchService)
    
    # Mock retriever
    mock_retriever = Mock()
    
    # Mock 검색 결과
    mock_docs = [
        Document(
            page_content="안녕하세요. 오늘은 인공지능에 대해 말씀드리겠습니다.",
            metadata={
                "video_id": "dQw4w9WgXcQ",
                "start_time": 0.0,
                "end_time": 3.5,
                "chunk_index": 0
            }
        ),
        Document(
            page_content="AI 기술은 머신러닝과 딥러닝으로 구성됩니다.",
            metadata={
                "video_id": "dQw4w9WgXcQ", 
                "start_time": 10.0,
                "end_time": 15.0,
                "chunk_index": 1
            }
        )
    ]
    
    mock_retriever.invoke.return_value = mock_docs
    mock_retriever.ainvoke.return_value = mock_docs
    service.retriever = mock_retriever
    
    return service


@pytest.fixture
def rag_config():
    """RAG 설정"""
    return RAGConfig(
        model_name="gemini-2.0-flash",
        search_type="hybrid",
        max_docs=5,
        include_timestamps=True,
        include_sources=True,
        language="korean"
    )


class TestRAGChainInitialization:
    """RAG 체인 초기화 테스트"""
    
    def test_rag_chain_initialization(self, mock_similarity_service, rag_config):
        """RAG 체인 기본 초기화 테스트"""
        with patch('src.agents.rag_chain.ChatGoogleGenerativeAI') as mock_llm:
            rag_chain = RAGChain(mock_similarity_service, rag_config)
            
            assert rag_chain.similarity_service == mock_similarity_service
            assert rag_chain.config == rag_config
            assert rag_chain.chain is not None
            assert rag_chain.prompt_template is not None
            
            # LLM 초기화 확인
            mock_llm.assert_called_once_with(
                model="gemini-2.0-flash",
                temperature=0.7,
                max_tokens=1000
            )
    
    def test_rag_chain_with_default_config(self, mock_similarity_service):
        """기본 설정으로 RAG 체인 초기화"""
        with patch('src.agents.rag_chain.ChatGoogleGenerativeAI'):
            rag_chain = RAGChain(mock_similarity_service)
            
            assert isinstance(rag_chain.config, RAGConfig)
            assert rag_chain.config.model_name == "gemini-2.0-flash"
            assert rag_chain.config.search_type == "hybrid"
            assert rag_chain.config.language == "korean"


class TestRAGChainInvoke:
    """RAG 체인 invoke 테스트"""
    
    def test_rag_chain_format_context(self, mock_similarity_service):
        """컨텍스트 포맷팅 기능 테스트 (Mock 문제 해결을 위한 단순 테스트)"""
        config = RAGConfig(include_timestamps=True)
        
        with patch('src.agents.rag_chain.ChatGoogleGenerativeAI'):
            rag_chain = RAGChain(mock_similarity_service, config)
            
            # Document 리스트 직접 생성
            from langchain_core.documents import Document
            docs = [
                Document(
                    page_content="인공지능은 컴퓨터가 학습하는 기술입니다.",
                    metadata={"start_time": 10.0, "end_time": 15.0}
                ),
                Document(
                    page_content="머신러닝과 딥러닝이 핵심입니다.",
                    metadata={"start_time": 20.0, "end_time": 25.0}  
                )
            ]
            
            # 컨텍스트 포맷팅 테스트
            context = rag_chain._format_context(docs)
            
            # 검증
            assert "[00:10] 인공지능은 컴퓨터가 학습하는 기술입니다." in context
            assert "[00:20] 머신러닝과 딥러닝이 핵심입니다." in context
    
    def test_rag_chain_format_sources(self, mock_similarity_service):
        """출처 정보 포맷팅 테스트"""
        config = RAGConfig(include_sources=True)
        
        with patch('src.agents.rag_chain.ChatGoogleGenerativeAI'):
            rag_chain = RAGChain(mock_similarity_service, config)
            
            # Document 리스트 직접 생성
            from langchain_core.documents import Document
            docs = [
                Document(
                    page_content="테스트 내용",
                    metadata={
                        "video_id": "ABC123",
                        "start_time": 30.0,
                        "end_time": 35.0
                    }
                )
            ]
            
            # 출처 포맷팅 테스트
            sources = rag_chain._format_sources(docs)
            
            # 검증
            assert "출처 1: 영상 ABC123 (00:30)" in sources


class TestRAGChainAsync:
    """RAG 체인 비동기 테스트"""
    
    @patch('src.agents.rag_chain.ChatGoogleGenerativeAI')
    @patch('src.utils.langfuse_utils.get_langfuse_callbacks')
    def test_rag_chain_ainvoke(self, mock_callbacks, mock_llm_class, mock_similarity_service, rag_config):
        """RAG 체인 비동기 invoke 테스트"""
        mock_callbacks.return_value = []
        
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        mock_answer = "AI는 미래 기술의 핵심입니다."
        
        rag_chain = RAGChain(mock_similarity_service, rag_config)
        
        async def async_test():
            with patch.object(rag_chain.chain, 'ainvoke', return_value=mock_answer) as mock_chain_ainvoke:
                result = await rag_chain.ainvoke("AI의 미래는?")
                
                mock_chain_ainvoke.assert_called_once_with("AI의 미래는?", config={"callbacks": []})
                
                assert result["answer"] == mock_answer
                assert result["question"] == "AI의 미래는?"
                assert result["metadata"]["execution_type"] == "async"
                return result
        
        # 비동기 테스트 실행
        result = asyncio.run(async_test())
        assert result is not None


class TestRAGChainStreaming:
    """RAG 체인 스트리밍 테스트"""
    
    @patch('src.agents.rag_chain.ChatGoogleGenerativeAI')
    @patch('src.utils.langfuse_utils.get_langfuse_callbacks')
    def test_rag_chain_stream(self, mock_callbacks, mock_llm_class, mock_similarity_service, rag_config):
        """RAG 체인 스트리밍 테스트"""
        mock_callbacks.return_value = []
        
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        # Mock 스트리밍 응답
        mock_chunks = ["인공", "지능은 ", "컴퓨터가 ", "학습하는 ", "기술입니다."]
        
        rag_chain = RAGChain(mock_similarity_service, rag_config)
        
        with patch.object(rag_chain.chain, 'stream', return_value=iter(mock_chunks)) as mock_chain_stream:
            chunks = list(rag_chain.stream("AI란 무엇인가?"))
            
            mock_chain_stream.assert_called_once_with("AI란 무엇인가?", config={"callbacks": []})
            assert chunks == mock_chunks


class TestRAGChainBatch:
    """RAG 체인 배치 테스트"""
    
    @patch('src.agents.rag_chain.ChatGoogleGenerativeAI')
    @patch('src.utils.langfuse_utils.get_langfuse_callbacks')
    def test_rag_chain_batch(self, mock_callbacks, mock_llm_class, mock_similarity_service, rag_config):
        """RAG 체인 배치 처리 테스트"""
        mock_callbacks.return_value = []
        
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        questions = ["AI란 무엇인가?", "머신러닝이란?", "딥러닝의 특징은?"]
        mock_answers = [
            "AI는 인공지능입니다.", 
            "머신러닝은 학습 기술입니다.", 
            "딥러닝은 신경망 기술입니다."
        ]
        
        rag_chain = RAGChain(mock_similarity_service, rag_config)
        
        with patch.object(rag_chain.chain, 'batch', return_value=mock_answers) as mock_chain_batch:
            results = rag_chain.batch(questions)
            
            mock_chain_batch.assert_called_once_with(questions, config={"callbacks": []})
            
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result["answer"] == mock_answers[i]
                assert result["question"] == questions[i]
                assert result["metadata"]["execution_type"] == "batch"


class TestRAGChainContextFormatting:
    """RAG 체인 컨텍스트 포맷팅 테스트"""
    
    def test_format_context_with_timestamps(self, mock_similarity_service):
        """타임스탬프 포함 컨텍스트 포맷팅 테스트"""
        config = RAGConfig(include_timestamps=True)
        
        with patch('src.agents.rag_chain.ChatGoogleGenerativeAI'):
            rag_chain = RAGChain(mock_similarity_service, config)
            
            docs = [
                Document(
                    page_content="첫 번째 내용입니다.",
                    metadata={"start_time": 30.0, "end_time": 35.0}
                ),
                Document(
                    page_content="두 번째 내용입니다.",
                    metadata={"start_time": 90.5, "end_time": 95.0}
                )
            ]
            
            context = rag_chain._format_context(docs)
            
            assert "[00:30] 첫 번째 내용입니다." in context
            assert "[01:30] 두 번째 내용입니다." in context
    
    def test_format_context_without_timestamps(self, mock_similarity_service):
        """타임스탬프 제외 컨텍스트 포맷팅 테스트"""
        config = RAGConfig(include_timestamps=False)
        
        with patch('src.agents.rag_chain.ChatGoogleGenerativeAI'):
            rag_chain = RAGChain(mock_similarity_service, config)
            
            docs = [
                Document(page_content="내용 1", metadata={"start_time": 10.0}),
                Document(page_content="내용 2", metadata={"start_time": 20.0})
            ]
            
            context = rag_chain._format_context(docs)
            
            assert "[" not in context  # 타임스탬프 없음
            assert "내용 1" in context
            assert "내용 2" in context
    
    def test_format_sources(self, mock_similarity_service):
        """출처 정보 포맷팅 테스트"""
        config = RAGConfig(include_sources=True)
        
        with patch('src.agents.rag_chain.ChatGoogleGenerativeAI'):
            rag_chain = RAGChain(mock_similarity_service, config)
            
            docs = [
                Document(
                    page_content="내용",
                    metadata={
                        "video_id": "ABC123",
                        "start_time": 45.0,
                        "end_time": 50.0
                    }
                )
            ]
            
            sources = rag_chain._format_sources(docs)
            
            assert "출처 1: 영상 ABC123 (00:45)" in sources


class TestRAGChainErrorHandling:
    """RAG 체인 에러 처리 테스트"""
    
    @patch('src.agents.rag_chain.ChatGoogleGenerativeAI')
    def test_invoke_error_handling(self, mock_llm_class, mock_similarity_service, rag_config):
        """invoke 에러 처리 테스트"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        rag_chain = RAGChain(mock_similarity_service, rag_config)
        
        # Chain invoke에서 에러 발생
        with patch.object(rag_chain.chain, 'invoke', side_effect=Exception("LLM Error")):
            with pytest.raises(ProcessingError, match="RAG chain failed"):
                rag_chain.invoke("테스트 질문")
    
    @patch('src.agents.rag_chain.ChatGoogleGenerativeAI')
    def test_stream_error_handling(self, mock_llm_class, mock_similarity_service, rag_config):
        """스트리밍 에러 처리 테스트"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        rag_chain = RAGChain(mock_similarity_service, rag_config)
        
        # Chain stream에서 에러 발생
        with patch.object(rag_chain.chain, 'stream', side_effect=Exception("Stream Error")):
            with pytest.raises(ProcessingError, match="RAG streaming failed"):
                list(rag_chain.stream("테스트 질문"))


class TestRAGChainFactory:
    """RAG 체인 팩토리 테스트"""
    
    @patch('src.agents.rag_chain.create_service_factory')
    @patch('src.services.config_manager.create_configuration_manager')
    def test_create_rag_chain_default(self, mock_create_config, mock_create_factory):
        """기본 설정으로 RAG 체인 생성 테스트"""
        # Mock 설정
        mock_config_manager = Mock()
        mock_create_config.return_value = mock_config_manager
        
        mock_service_factory = Mock()
        mock_similarity_service = Mock()
        mock_service_factory.get_similarity_search_service.return_value = mock_similarity_service
        mock_create_factory.return_value = mock_service_factory
        
        with patch('src.agents.rag_chain.ChatGoogleGenerativeAI'):
            rag_chain = create_rag_chain()
            
            assert isinstance(rag_chain, RAGChain)
            assert rag_chain.similarity_service == mock_similarity_service
            
            # Factory 함수들이 호출되었는지 확인
            mock_create_config.assert_called_once()
            mock_create_factory.assert_called_once_with(mock_config_manager)
            mock_service_factory.get_similarity_search_service.assert_called_once()
    
    @patch('src.agents.rag_chain.create_service_factory')
    def test_create_rag_chain_with_config(self, mock_create_factory):
        """설정 관리자를 제공하여 RAG 체인 생성 테스트"""
        mock_config_manager = Mock()
        
        mock_service_factory = Mock()
        mock_similarity_service = Mock()
        mock_service_factory.get_similarity_search_service.return_value = mock_similarity_service
        mock_create_factory.return_value = mock_service_factory
        
        custom_rag_config = RAGConfig(model_name="custom-model")
        
        with patch('src.agents.rag_chain.ChatGoogleGenerativeAI'):
            rag_chain = create_rag_chain(mock_config_manager, custom_rag_config)
            
            assert rag_chain.config.model_name == "custom-model"
            mock_create_factory.assert_called_once_with(mock_config_manager) 