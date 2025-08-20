"""
로거 설정 테스트

프로젝트의 로깅 시스템이 올바르게 설정되고 동작하는지 확인합니다.
"""

import logging
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

from src.utils.logger import setup_logger, get_logger


def test_setup_logger_function_exists():
    """setup_logger 함수가 존재하는지 확인"""
    assert setup_logger is not None
    assert callable(setup_logger)


def test_get_logger_function_exists():
    """get_logger 함수가 존재하는지 확인"""
    assert get_logger is not None
    assert callable(get_logger)


def test_setup_logger_returns_logger():
    """setup_logger가 Logger 객체를 반환하는지 확인"""
    logger = setup_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_logger_has_correct_default_level():
    """로거가 올바른 기본 레벨을 가지는지 확인"""
    logger = setup_logger("test_default_level")
    # 기본적으로 INFO 레벨이어야 함
    assert logger.level == logging.INFO


def test_logger_can_set_different_levels():
    """로거가 다양한 레벨로 설정 가능한지 확인"""
    # DEBUG 레벨 테스트
    debug_logger = setup_logger("test_debug", level=logging.DEBUG)
    assert debug_logger.level == logging.DEBUG
    
    # WARNING 레벨 테스트
    warning_logger = setup_logger("test_warning", level=logging.WARNING)
    assert warning_logger.level == logging.WARNING
    
    # ERROR 레벨 테스트
    error_logger = setup_logger("test_error", level=logging.ERROR)
    assert error_logger.level == logging.ERROR


def test_logger_writes_to_correct_levels():
    """로거가 올바른 레벨에서 메시지를 출력하는지 확인"""
    # StringIO를 사용하여 로그 출력을 캡처
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    
    logger = setup_logger("test_levels", level=logging.DEBUG)
    logger.handlers.clear()  # 기존 핸들러 제거
    logger.addHandler(handler)
    
    # 각 레벨별로 메시지 로깅
    logger.debug("Debug message")
    logger.info("Info message") 
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    log_output = log_capture.getvalue()
    
    assert "Debug message" in log_output
    assert "Info message" in log_output
    assert "Warning message" in log_output
    assert "Error message" in log_output
    assert "Critical message" in log_output


def test_logger_respects_level_filtering():
    """로거가 설정된 레벨에 따라 메시지를 필터링하는지 확인"""
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    
    # WARNING 레벨로 설정
    logger = setup_logger("test_filtering", level=logging.WARNING)
    logger.handlers.clear()
    logger.addHandler(handler)
    
    logger.debug("Should not appear")
    logger.info("Should not appear") 
    logger.warning("Should appear")
    logger.error("Should appear")
    
    log_output = log_capture.getvalue()
    
    assert "Should not appear" not in log_output
    assert "Should appear" in log_output


def test_logger_has_proper_format():
    """로거가 적절한 형식으로 메시지를 출력하는지 확인"""
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = setup_logger("test_format", level=logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)
    
    logger.info("Test message")
    
    log_output = log_capture.getvalue()
    
    # 로그 출력에 시간, 로거명, 레벨, 메시지가 포함되어야 함
    assert "INFO" in log_output
    assert "test_format" in log_output  
    assert "Test message" in log_output


def test_get_logger_returns_same_instance():
    """get_logger가 같은 이름에 대해 동일한 인스턴스를 반환하는지 확인"""
    logger1 = get_logger("same_name")
    logger2 = get_logger("same_name")
    
    assert logger1 is logger2


def test_logger_can_log_with_extra_fields():
    """로거가 추가 필드와 함께 로깅할 수 있는지 확인"""
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    
    logger = setup_logger("test_extra", level=logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)
    
    # 추가 정보와 함께 로깅
    logger.info("Test message with extra", extra={"user_id": 123, "action": "test"})
    
    log_output = log_capture.getvalue()
    assert "Test message with extra" in log_output 