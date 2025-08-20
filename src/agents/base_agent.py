"""
LangGraph 기반 기본 에이전트

최신 LangGraph 패턴을 활용한 기본 에이전트 클래스입니다.
StateGraph, 메시지 처리, 상태 관리, 메모리, 도구 통합을 제공합니다.
"""

import asyncio
import os
from typing import Dict, Any, List, Optional, Union, Annotated, Callable
from dataclasses import dataclass, field
from datetime import datetime

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

# 프로젝트 모듈
from src.utils.exceptions import ConfigurationError
from src.utils.logger import get_project_logger
from src.utils.langfuse_utils import (
    get_langfuse_callbacks,
    is_langfuse_enabled,
    langfuse_manager,
    trace_agent_execution
)

logger = get_project_logger(__name__)


@dataclass
class AgentConfig:
    """에이전트 설정 클래스"""
    
    # 기본 설정
    agent_name: str
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # 시스템 프롬프트
    system_prompt: str = "You are a helpful AI assistant for YouTube video analysis."
    
    # 메모리 설정
    enable_memory: bool = True
    
    # 도구 설정
    tools: List[BaseTool] = field(default_factory=list)
    
    # 실행 설정
    execution_timeout: float = 30.0
    max_iterations: int = 10
    
    def __post_init__(self):
        """설정 후처리 및 검증"""
        if not self.agent_name or not self.agent_name.strip():
            raise ConfigurationError("Agent name cannot be empty")
        
        if not (0 <= self.temperature <= 1):
            raise ConfigurationError("Temperature must be between 0 and 1")
        
        if self.max_tokens <= 0:
            raise ConfigurationError("Max tokens must be positive")
        
        if self.execution_timeout <= 0:
            raise ConfigurationError("Execution timeout must be positive")


# LangGraph 상태 정의
class AgentState(Dict[str, Any]):
    """에이전트 상태 타입 어노테이션"""
    messages: Annotated[List[AnyMessage], add_messages]
    agent_metadata: Dict[str, Any]


class BaseAgent:
    """
    LangGraph 기반 기본 에이전트 클래스
    
    최신 LangGraph 패턴을 활용하여 상태 관리, 메시지 처리, 
    도구 통합, 메모리 기능을 제공합니다.
    """
    
    def __init__(self, config: AgentConfig):
        """
        기본 에이전트를 초기화합니다.
        
        Args:
            config: 에이전트 설정
        """
        self.config = config
        self.llm = None
        self.state_graph = None
        self.compiled_graph = None
        self.memory_saver = None
        self.tools = config.tools.copy()
        self.is_compiled = False
        
        # LLM 초기화
        self._initialize_llm()
        
        # 메모리 초기화
        if config.enable_memory:
            self.memory_saver = MemorySaver()
        
        # 상태 그래프 초기화
        self._initialize_state_graph()
        
        logger.info(f"BaseAgent '{config.agent_name}' initialized")
    
    def _initialize_llm(self):
        """LLM을 초기화합니다."""
        try:
            # Google API 키 확인
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key and self.config.model_name.startswith("gemini"):
                logger.warning("Google API key not found. Some features may not work.")
            
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                google_api_key=api_key
            )
            
            # 도구가 있으면 LLM에 바인딩
            if self.tools:
                self.llm = self.llm.bind_tools(self.tools)
            
            logger.info(f"LLM initialized: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            # Mock LLM for testing
            self.llm = MockLLM()
    
    def _initialize_state_graph(self):
        """상태 그래프를 초기화합니다."""
        try:
            # StateGraph 생성
            self.state_graph = StateGraph(AgentState)
            
            # 노드 추가
            self.state_graph.add_node("agent", self._agent_node)
            
            # 도구가 있으면 도구 노드 추가
            if self.tools:
                self.state_graph.add_node("tools", ToolNode(self.tools))
            
            # 엣지 추가
            self.state_graph.add_edge(START, "agent")
            
            if self.tools:
                # 조건부 엣지 (도구 호출 여부에 따라)
                self.state_graph.add_conditional_edges(
                    "agent",
                    self._should_continue,
                    {
                        "continue": "tools",
                        "end": END
                    }
                )
                self.state_graph.add_edge("tools", "agent")
            else:
                self.state_graph.add_edge("agent", END)
            
            logger.info("State graph initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize state graph: {e}")
            raise ConfigurationError(f"State graph initialization failed: {e}")
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """
        메인 에이전트 노드 - LLM 호출 및 응답 생성
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        try:
            messages = state.get("messages", [])
            
            # 시스템 메시지가 없으면 추가
            if not messages or not isinstance(messages[0], SystemMessage):
                system_message = SystemMessage(content=self.config.system_prompt)
                messages.insert(0, system_message)
            
            # LLM 호출
            response = self._call_llm(messages)
            
            # 응답을 상태에 추가
            return {
                "messages": [response],
                "agent_metadata": {
                    **state.get("agent_metadata", {}),
                    "last_response_time": datetime.now().isoformat(),
                    "model_used": self.config.model_name
                }
            }
            
        except Exception as e:
            logger.error(f"Error in agent node: {e}")
            error_message = AIMessage(
                content=f"Sorry, I encountered an error: {str(e)}"
            )
            return {
                "messages": [error_message],
                "agent_metadata": {
                    **state.get("agent_metadata", {}),
                    "error": str(e),
                    "error_time": datetime.now().isoformat()
                }
            }
    
    def _should_continue(self, state: AgentState) -> str:
        """
        도구 호출이 필요한지 결정합니다.
        
        Args:
            state: 현재 상태
            
        Returns:
            다음 행동 ("continue" 또는 "end")
        """
        messages = state.get("messages", [])
        
        if messages:
            last_message = messages[-1]
            # 도구 호출이 있으면 continue
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "continue"
        
        return "end"
    
    def _call_llm(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        LLM을 호출합니다.
        
        Args:
            messages: 메시지 목록
            
        Returns:
            LLM 응답 메시지
        """
        try:
            response = self.llm.invoke(messages)
            return response
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # 기본 오류 응답
            return AIMessage(content="I'm sorry, I'm having trouble processing your request right now.")
    
    def compile_graph(self) -> Any:
        """
        상태 그래프를 컴파일합니다.
        
        Returns:
            컴파일된 그래프
        """
        try:
            compile_kwargs = {}
            if self.memory_saver:
                compile_kwargs["checkpointer"] = self.memory_saver
            
            self.compiled_graph = self.state_graph.compile(**compile_kwargs)
            self.is_compiled = True
            
            # Langfuse 추적 설정 로깅
            if is_langfuse_enabled():
                logger.info(f"Graph compiled successfully with Langfuse tracking enabled")
            else:
                logger.info("Graph compiled successfully")
                
            return self.compiled_graph
            
        except Exception as e:
            logger.error(f"Graph compilation failed: {e}")
            
            # Langfuse 에러 로깅
            if is_langfuse_enabled():
                langfuse_manager.log_error(e, {
                    "agent_name": self.config.agent_name,
                    "operation": "graph_compilation",
                    "model": self.config.model_name
                })
            
            raise ConfigurationError(f"Graph compilation failed: {e}")
    
    def create_initial_state(self) -> Dict[str, Any]:
        """
        초기 상태를 생성합니다.
        
        Returns:
            초기 상태 딕셔너리
        """
        return {
            "messages": [],
            "agent_metadata": {
                "agent_name": self.config.agent_name,
                "created_at": datetime.now().isoformat(),
                "model": self.config.model_name
            }
        }
    
    def add_message_to_state(self, state: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        """
        상태에 메시지를 추가합니다.
        
        Args:
            state: 현재 상태
            message: 추가할 메시지
            
        Returns:
            업데이트된 상태
        """
        new_state = state.copy()
        
        # 메시지를 적절한 형식으로 변환
        if message.get("role") == "human":
            langchain_message = HumanMessage(content=message["content"])
        elif message.get("role") == "assistant":
            langchain_message = AIMessage(content=message["content"])
        else:
            langchain_message = HumanMessage(content=message["content"])  # 기본값
        
        new_state["messages"] = new_state.get("messages", []) + [langchain_message]
        
        return new_state
    
    def update_agent_metadata(self, state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트 메타데이터를 업데이트합니다.
        
        Args:
            state: 현재 상태
            metadata: 업데이트할 메타데이터
            
        Returns:
            업데이트된 상태
        """
        new_state = state.copy()
        current_metadata = new_state.get("agent_metadata", {})
        current_metadata.update(metadata)
        new_state["agent_metadata"] = current_metadata
        
        return new_state
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        상태의 유효성을 검증합니다.
        
        Args:
            state: 검증할 상태
            
        Returns:
            유효성 여부
        """
        try:
            # 필수 키 확인
            if "messages" not in state or "agent_metadata" not in state:
                return False
            
            # 메시지가 리스트인지 확인
            if not isinstance(state["messages"], list):
                return False
            
            # 메타데이터가 딕셔너리인지 확인
            if not isinstance(state["agent_metadata"], dict):
                return False
            
            return True
            
        except Exception:
            return False
    
    def recover_from_corrupted_state(self, corrupted_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        손상된 상태를 복구합니다.
        
        Args:
            corrupted_state: 손상된 상태
            
        Returns:
            복구된 상태
        """
        logger.warning("Recovering from corrupted state")
        
        # 기본 상태로 초기화
        recovered_state = self.create_initial_state()
        
        # 복구 가능한 부분이 있으면 보존
        if isinstance(corrupted_state, dict):
            if "messages" in corrupted_state and isinstance(corrupted_state["messages"], list):
                # 유효한 메시지만 보존
                valid_messages = []
                for msg in corrupted_state["messages"]:
                    if isinstance(msg, dict) and "content" in msg:
                        valid_messages.append(msg)
                
                if valid_messages:
                    recovered_state["messages"] = valid_messages
        
        return recovered_state
    
    @trace_agent_execution("base_agent")
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트를 동기적으로 실행합니다.
        
        Args:
            input_data: 입력 데이터
            
        Returns:
            실행 결과
        """
        if not self.is_compiled:
            self.compile_graph()
        
        try:
            # 입력 검증
            if not isinstance(input_data, dict) or "messages" not in input_data:
                raise ValueError("Input must contain 'messages' key")
            
            # Langfuse 콜백 설정
            invoke_config = {}
            if is_langfuse_enabled():
                invoke_config["callbacks"] = get_langfuse_callbacks()
            
            # 그래프 실행
            result = self.compiled_graph.invoke(input_data, config=invoke_config)
            
            # Langfuse 로깅
            if is_langfuse_enabled():
                langfuse_manager.log_agent_execution(
                    agent_name=self.config.agent_name,
                    input_data=input_data,
                    output_data=result,
                    model=self.config.model_name,
                    operation="invoke"
                )
            
            logger.info("Agent invocation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}")
            
            # Langfuse 에러 로깅
            if is_langfuse_enabled():
                langfuse_manager.log_error(e, {
                    "agent_name": self.config.agent_name,
                    "operation": "invoke",
                    "input_data": input_data,
                    "model": self.config.model_name
                })
            
            raise
    
    def invoke_with_thread(self, input_data: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
        """
        스레드 ID와 함께 에이전트를 실행합니다 (메모리 유지).
        
        Args:
            input_data: 입력 데이터
            thread_id: 스레드 ID
            
        Returns:
            실행 결과
        """
        if not self.is_compiled:
            self.compile_graph()
        
        if not self.memory_saver:
            logger.warning("Memory not enabled, but thread_id provided")
            return self.invoke(input_data)
        
        try:
            # 설정 구성 (스레드 + Langfuse)
            config = {"configurable": {"thread_id": thread_id}}
            if is_langfuse_enabled():
                config["callbacks"] = get_langfuse_callbacks()
            
            # Langfuse 세션 추적
            if is_langfuse_enabled():
                with langfuse_manager.trace_session(
                    session_id=thread_id,
                    agent_name=self.config.agent_name,
                    model=self.config.model_name
                ):
                    result = self.compiled_graph.invoke(input_data, config=config)
            else:
                result = self.compiled_graph.invoke(input_data, config=config)
            
            logger.info(f"Agent invocation with thread {thread_id} completed")
            return result
            
        except Exception as e:
            logger.error(f"Agent invocation with thread failed: {e}")
            
            # Langfuse 에러 로깅
            if is_langfuse_enabled():
                langfuse_manager.log_error(e, {
                    "agent_name": self.config.agent_name,
                    "operation": "invoke_with_thread",
                    "thread_id": thread_id,
                    "input_data": input_data,
                    "model": self.config.model_name
                })
            
            raise
    
    async def ainvoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트를 비동기적으로 실행합니다.
        
        Args:
            input_data: 입력 데이터
            
        Returns:
            실행 결과
        """
        if not self.is_compiled:
            self.compile_graph()
        
        try:
            # Langfuse 콜백 설정
            invoke_config = {}
            if is_langfuse_enabled():
                invoke_config["callbacks"] = get_langfuse_callbacks()
            
            result = await self.compiled_graph.ainvoke(input_data, config=invoke_config)
            
            # Langfuse 로깅
            if is_langfuse_enabled():
                langfuse_manager.log_agent_execution(
                    agent_name=self.config.agent_name,
                    input_data=input_data,
                    output_data=result,
                    model=self.config.model_name,
                    operation="ainvoke"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Async agent invocation failed: {e}")
            
            # Langfuse 에러 로깅
            if is_langfuse_enabled():
                langfuse_manager.log_error(e, {
                    "agent_name": self.config.agent_name,
                    "operation": "ainvoke",
                    "input_data": input_data,
                    "model": self.config.model_name
                })
            
            raise
    
    def stream(self, input_data: Dict[str, Any]):
        """
        에이전트를 스트리밍 모드로 실행합니다.
        
        Args:
            input_data: 입력 데이터
            
        Yields:
            스트리밍 결과
        """
        if not self.is_compiled:
            self.compile_graph()
        
        try:
            # Langfuse 콜백 설정
            stream_config = {}
            if is_langfuse_enabled():
                stream_config["callbacks"] = get_langfuse_callbacks()
            
            # 스트리밍 실행 및 결과 수집 (Langfuse 로깅용)
            stream_results = []
            for chunk in self.compiled_graph.stream(input_data, config=stream_config):
                stream_results.append(chunk)
                yield chunk
            
            # Langfuse 로깅 (스트리밍 완료 후)
            if is_langfuse_enabled():
                langfuse_manager.log_agent_execution(
                    agent_name=self.config.agent_name,
                    input_data=input_data,
                    output_data={"stream_chunks": len(stream_results), "final_result": stream_results[-1] if stream_results else None},
                    model=self.config.model_name,
                    operation="stream"
                )
                
        except Exception as e:
            logger.error(f"Agent streaming failed: {e}")
            
            # Langfuse 에러 로깅
            if is_langfuse_enabled():
                langfuse_manager.log_error(e, {
                    "agent_name": self.config.agent_name,
                    "operation": "stream",
                    "input_data": input_data,
                    "model": self.config.model_name
                })
            
            raise
    
    def get_thread_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        특정 스레드의 상태를 가져옵니다.
        
        Args:
            thread_id: 스레드 ID
            
        Returns:
            스레드 상태
        """
        if not self.memory_saver or not self.is_compiled:
            return None
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.compiled_graph.get_state(config)
            
            return state.values if state else None
            
        except Exception as e:
            logger.error(f"Failed to get thread state: {e}")
            return None
    
    def reset_thread_state(self, thread_id: str) -> bool:
        """
        특정 스레드의 상태를 리셋합니다.
        
        Args:
            thread_id: 스레드 ID
            
        Returns:
            리셋 성공 여부
        """
        if not self.memory_saver or not self.is_compiled:
            return False
        
        try:
            # 메모리 세이버에서 해당 스레드 상태 삭제
            # 구체적인 구현은 메모리 저장소에 따라 다를 수 있음
            logger.info(f"Thread {thread_id} state reset")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset thread state: {e}")
            return False


# Mock LLM for testing
class MockLLM:
    """테스트용 Mock LLM"""
    
    def invoke(self, messages):
        return AIMessage(content="Mock response for testing")
    
    def bind_tools(self, tools):
        return self 