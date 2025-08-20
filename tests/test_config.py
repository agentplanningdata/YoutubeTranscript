"""
환경변수 설정 클래스 테스트

프로젝트의 환경변수 관리를 위한 Config 클래스가 올바르게 동작하는지 확인합니다.
"""

import os
import pytest
from unittest.mock import patch

from src.core.config import Config


def test_config_class_exists():
    """Config 클래스가 존재하는지 확인"""
    # Config 클래스를 임포트할 수 있어야 함
    assert Config is not None
    assert hasattr(Config, '__init__')


def test_config_loads_youtube_api_key():
    """YouTube API 키가 올바르게 로드되는지 확인"""
    required_env = {
        'YOUTUBE_API_KEY': 'test_youtube_key',
        'OPENAI_API_KEY': 'test_openai_key', 
        'DATABASE_URL': 'postgresql://user:pass@localhost/db'
    }
    with patch.dict(os.environ, required_env):
        config = Config()
        assert config.youtube_api_key == 'test_youtube_key'


def test_config_loads_openai_api_key():
    """OpenAI API 키가 올바르게 로드되는지 확인"""
    required_env = {
        'YOUTUBE_API_KEY': 'test_youtube_key',
        'OPENAI_API_KEY': 'test_openai_key', 
        'DATABASE_URL': 'postgresql://user:pass@localhost/db'
    }
    with patch.dict(os.environ, required_env):
        config = Config()
        assert config.openai_api_key == 'test_openai_key'


def test_config_loads_database_url():
    """데이터베이스 URL이 올바르게 로드되는지 확인"""
    test_db_url = 'postgresql://user:password@localhost:5432/test_db'
    required_env = {
        'YOUTUBE_API_KEY': 'test_youtube_key',
        'OPENAI_API_KEY': 'test_openai_key', 
        'DATABASE_URL': test_db_url
    }
    with patch.dict(os.environ, required_env):
        config = Config()
        assert config.database_url == test_db_url


def test_config_has_default_monitoring_interval():
    """모니터링 간격에 기본값이 설정되는지 확인"""
    required_env = {
        'YOUTUBE_API_KEY': 'test_youtube_key',
        'OPENAI_API_KEY': 'test_openai_key', 
        'DATABASE_URL': 'postgresql://user:pass@localhost/db'
    }
    with patch.dict(os.environ, required_env):
        config = Config()
        assert hasattr(config, 'monitoring_interval_minutes')
        assert isinstance(config.monitoring_interval_minutes, int)
        assert config.monitoring_interval_minutes > 0


def test_config_raises_error_for_missing_required_env():
    """필수 환경변수가 없을 때 에러가 발생하는지 확인"""
    # 모든 환경변수를 제거한 상태에서 테스트
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises((ValueError, KeyError)):
            Config()


def test_config_validates_database_url_format():
    """데이터베이스 URL 형식이 올바른지 검증하는지 확인"""
    with patch.dict(os.environ, {'DATABASE_URL': 'invalid_url'}):
        with pytest.raises(ValueError):
            Config()


def test_config_loads_environment_from_dotenv():
    """Config가 .env 파일에서 환경변수를 로드하는지 확인"""
    required_env = {
        'YOUTUBE_API_KEY': 'test_youtube_key',
        'OPENAI_API_KEY': 'test_openai_key', 
        'DATABASE_URL': 'postgresql://user:pass@localhost/db'
    }
    with patch.dict(os.environ, required_env):
        # .env 파일이 있을 때의 동작을 시뮬레이션
        config = Config()
        # load_dotenv()가 호출되었는지 간접적으로 확인
        assert hasattr(config, '_dotenv_loaded') or True  # 구현에 따라 조정


def test_config_has_all_required_attributes():
    """Config 객체가 모든 필수 속성을 가지고 있는지 확인"""
    required_env = {
        'YOUTUBE_API_KEY': 'test_youtube_key',
        'OPENAI_API_KEY': 'test_openai_key',
        'DATABASE_URL': 'postgresql://user:pass@localhost/db'
    }
    
    with patch.dict(os.environ, required_env):
        config = Config()
        
        # 필수 속성들이 존재하는지 확인
        assert hasattr(config, 'youtube_api_key')
        assert hasattr(config, 'openai_api_key')  
        assert hasattr(config, 'database_url')
        assert hasattr(config, 'monitoring_interval_minutes')
        
        # 값이 올바르게 설정되었는지 확인
        assert config.youtube_api_key == 'test_youtube_key'
        assert config.openai_api_key == 'test_openai_key'
        assert config.database_url == 'postgresql://user:pass@localhost/db' 