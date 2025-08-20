"""
로깅 유틸리티

프로젝트 전체에서 사용할 로거 설정 및 관리 기능을 제공합니다.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str, 
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    지정된 이름과 설정으로 로거를 생성하고 설정합니다.
    
    Args:
        name: 로거 이름
        level: 로그 레벨 (기본값: INFO)
        format_string: 로그 포맷 문자열
        
    Returns:
        설정된 Logger 객체
    """
    logger = logging.getLogger(name)
    
    # 기존 핸들러가 있으면 중복 방지를 위해 제거
    if logger.handlers:
        logger.handlers.clear()
    
    # 로그 레벨 설정
    logger.setLevel(level)
    
    # 포맷터 설정
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # 상위 로거로의 전파 방지 (중복 출력 방지)
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    지정된 이름의 로거를 가져옵니다. 존재하지 않으면 기본 설정으로 생성합니다.
    
    Args:
        name: 로거 이름
        
    Returns:
        Logger 객체
    """
    logger = logging.getLogger(name)
    
    # 로거가 아직 설정되지 않았다면 기본 설정으로 설정
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


def set_log_level(logger: logging.Logger, level: int) -> None:
    """
    로거의 로그 레벨을 변경합니다.
    
    Args:
        logger: 로거 객체
        level: 새로운 로그 레벨
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def get_project_logger(module_name: str) -> logging.Logger:
    """
    프로젝트 표준 로거를 가져옵니다.
    
    Args:
        module_name: 모듈 이름 (보통 __name__)
        
    Returns:
        프로젝트 표준 설정이 적용된 Logger 객체
    """
    # 프로젝트 루트 네임스페이스 추가
    if not module_name.startswith('youtube_transcript'):
        if module_name.startswith('src.'):
            module_name = f"youtube_transcript.{module_name[4:]}"
        else:
            module_name = f"youtube_transcript.{module_name}"
    
    return get_logger(module_name) 