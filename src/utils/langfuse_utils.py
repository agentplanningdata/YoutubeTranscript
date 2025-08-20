"""
Langfuse 통합 유틸리티

LangChain/LangGraph 애플리케이션의 실행을 추적하고 모니터링하기 위한
Langfuse 통합 기능을 제공합니다.
"""

import os
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from src.config import config
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)

# Langfuse 관련 임포트 (선택적)
try:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    Langfuse = None
    CallbackHandler = None
    LANGFUSE_AVAILABLE = False
    logger.warning("Langfuse not available. Install with 'pip install langfuse'")


class LangfuseManager:
    """
    Langfuse 추적 관리 클래스
    
    LangChain/LangGraph 애플리케이션의 실행을 추적하고
    성능 지표를 모니터링하는 기능을 제공합니다.
    """
    
    def __init__(self):
        """Langfuse 매니저를 초기화합니다."""
        self.client = None
        self.callback_handler = None
        self.enabled = False
        
        if LANGFUSE_AVAILABLE and config.is_langfuse_enabled():
            self._initialize_langfuse()
        else:
            logger.info("Langfuse tracking disabled (missing config or package)")
    
    def _initialize_langfuse(self):
        """Langfuse 클라이언트를 초기화합니다."""
        try:
            self.client = Langfuse(
                secret_key=config.langfuse_secret_key,
                public_key=config.langfuse_public_key,
                host=config.langfuse_host
            )
            
            # 연결 테스트
            self.client.auth_check()
            
            # Langfuse 3.3.0+ 공식 방법: 환경변수를 자동으로 읽음
            self.callback_handler = CallbackHandler()
            
            self.enabled = True
            logger.info(f"Langfuse initialized successfully (host: {config.langfuse_host})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self.enabled = False
    
    def is_enabled(self) -> bool:
        """Langfuse 추적이 활성화되어 있는지 확인합니다."""
        return self.enabled
    
    def get_callback_handler(self) -> Optional[CallbackHandler]:
        """LangChain 콜백 핸들러를 반환합니다."""
        return self.callback_handler if self.enabled else None
    
    def get_callbacks(self) -> List[Any]:
        """LangChain callbacks 리스트를 반환합니다."""
        if self.enabled and self.callback_handler:
            return [self.callback_handler]
        return []
    
    @contextmanager
    def trace_session(self, session_id: str, user_id: Optional[str] = None, **metadata):
        """
        세션 기반 추적 컨텍스트 매니저
        
        Args:
            session_id: 세션 식별자
            user_id: 사용자 식별자 (선택적)
            **metadata: 추가 메타데이터
        """
        if not self.enabled:
            yield None
            return
        
        try:
            # 세션 메타데이터 설정
            session_metadata = {
                "session_id": session_id,
                **({"user_id": user_id} if user_id else {}),
                **metadata
            }
            
            # 콜백 핸들러에 세션 정보 설정
            if self.callback_handler:
                self.callback_handler.trace_id = session_id
            
            logger.debug(f"Starting Langfuse trace session: {session_id}")
            yield session_metadata
            
        except Exception as e:
            logger.error(f"Error in Langfuse trace session: {e}")
            yield None
        finally:
            if self.enabled:
                try:
                    self.client.flush()
                    logger.debug(f"Finished Langfuse trace session: {session_id}")
                except Exception as e:
                    logger.error(f"Error flushing Langfuse session: {e}")
    
    def create_trace(self, name: str, **metadata) -> Optional[Any]:
        """
        새로운 이벤트를 생성합니다. (Langfuse 3.3.0+ 호환)
        
        Args:
            name: 이벤트 이름
            **metadata: 추가 메타데이터
            
        Returns:
            Langfuse event 객체
        """
        if not self.enabled:
            return None
        
        try:
            # Langfuse 3.3.0에서는 create_event 사용
            event = self.client.create_event(
                name=name,
                metadata=metadata
            )
            logger.debug(f"Created Langfuse event: {name}")
            return event
        except Exception as e:
            logger.error(f"Failed to create Langfuse event: {e}")
            return None
    
    def log_agent_execution(self, agent_name: str, input_data: Dict[str, Any], 
                          output_data: Dict[str, Any], **metadata):
        """
        에이전트 실행을 로깅합니다. (Langfuse 3.3.0+ 호환)
        
        Args:
            agent_name: 에이전트 이름
            input_data: 입력 데이터
            output_data: 출력 데이터
            **metadata: 추가 메타데이터
        """
        if not self.enabled:
            return
        
        try:
            event_metadata = {
                "agent_name": agent_name,
                "input_size": len(str(input_data)),
                "output_size": len(str(output_data)),
                **metadata
            }
            
            # Langfuse 3.3.0에서는 create_event 사용
            event = self.client.create_event(
                name=f"agent_execution_{agent_name}",
                input=input_data,
                output=output_data,
                metadata=event_metadata
            )
            
            logger.debug(f"Logged agent execution for {agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to log agent execution: {e}")
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """
        에러를 추적합니다.
        
        Args:
            error: 발생한 에러
            context: 에러 발생 컨텍스트
        """
        if not self.enabled:
            return
        
        try:
            self.client.event(
                name="error",
                input=context,
                level="ERROR",
                status_message=str(error),
                metadata={
                    "error_type": type(error).__name__,
                    "error_message": str(error)
                }
            )
            
            logger.debug(f"Logged error: {type(error).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to log error to Langfuse: {e}")
    
    def flush(self):
        """모든 추적 데이터를 Langfuse 서버로 전송합니다."""
        if self.enabled and self.client:
            try:
                self.client.flush()
                logger.debug("Langfuse data flushed successfully")
            except Exception as e:
                logger.error(f"Failed to flush Langfuse data: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Langfuse 통계 정보를 반환합니다."""
        if not self.enabled:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "host": config.langfuse_host,
            "has_callback_handler": self.callback_handler is not None,
            "client_initialized": self.client is not None
        }


# 전역 Langfuse 매니저 인스턴스
langfuse_manager = LangfuseManager()


def get_langfuse_callbacks() -> List[Any]:
    """
    Langfuse 콜백 리스트를 반환하는 편의 함수
    
    Returns:
        LangChain에서 사용할 수 있는 콜백 리스트
    """
    return langfuse_manager.get_callbacks()


def is_langfuse_enabled() -> bool:
    """
    Langfuse 추적이 활성화되어 있는지 확인하는 편의 함수
    
    Returns:
        Langfuse 활성화 여부
    """
    return langfuse_manager.is_enabled()


def trace_agent_execution(agent_name: str):
    """
    에이전트 실행 추적을 위한 데코레이터
    
    Args:
        agent_name: 에이전트 이름
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_langfuse_enabled():
                return func(*args, **kwargs)
            
            try:
                # 함수 실행
                result = func(*args, **kwargs)
                
                # 실행 결과 로깅
                langfuse_manager.log_agent_execution(
                    agent_name=agent_name,
                    input_data={"args": str(args), "kwargs": str(kwargs)},
                    output_data={"result": str(result)},
                    function_name=func.__name__
                )
                
                return result
                
            except Exception as e:
                # 에러 로깅
                langfuse_manager.log_error(e, {
                    "agent_name": agent_name,
                    "function_name": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                })
                raise
        
        return wrapper
    return decorator


def create_langfuse_config_for_agent() -> Dict[str, Any]:
    """
    에이전트용 Langfuse 설정을 생성합니다.
    
    Returns:
        LangGraph compile시 사용할 수 있는 설정 딕셔너리
    """
    if is_langfuse_enabled():
        return {
            "callbacks": get_langfuse_callbacks(),
            "metadata": {
                "langfuse_enabled": True,
                "langfuse_host": config.langfuse_host
            }
        }
    else:
        return {
            "callbacks": [],
            "metadata": {
                "langfuse_enabled": False
            }
        } 