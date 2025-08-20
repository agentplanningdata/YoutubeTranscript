"""
환경변수 설정 관리

프로젝트의 환경변수들을 관리하는 Config 클래스입니다.
"""

import os
from typing import Optional

# dotenv가 설치되어 있을 경우에만 사용
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class Config:
    """환경변수 기반 설정 클래스"""
    
    def __init__(self):
        """환경변수를 로드하고 설정을 초기화합니다."""
        # .env 파일에서 환경변수 로드 (가능한 경우에만)
        if DOTENV_AVAILABLE:
            load_dotenv()
        
        # 필수 환경변수들
        self.youtube_api_key = self._get_required_env('YOUTUBE_API_KEY')
        self.openai_api_key = self._get_required_env('OPENAI_API_KEY')
        self.database_url = self._get_required_env('DATABASE_URL')
        
        # 데이터베이스 URL 형식 검증
        self._validate_database_url()
        
        # 선택적 환경변수들 (기본값 포함)
        self.monitoring_interval_minutes = int(
            os.getenv('MONITORING_INTERVAL_MINUTES', '5')
        )
        
        # 기타 설정들
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # dotenv 로드 확인용 플래그
        self._dotenv_loaded = True
    
    def _get_required_env(self, key: str) -> str:
        """필수 환경변수를 가져오거나 에러를 발생시킵니다."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"필수 환경변수 '{key}'가 설정되지 않았습니다.")
        return value
    
    def _validate_database_url(self):
        """데이터베이스 URL 형식을 검증합니다."""
        if not self.database_url.startswith(('postgresql://', 'postgres://')):
            raise ValueError("DATABASE_URL은 PostgreSQL URL 형식이어야 합니다.")
    
    @property
    def is_development(self) -> bool:
        """개발 환경인지 확인합니다."""
        return os.getenv('ENVIRONMENT', 'development').lower() == 'development'
    
    @property
    def is_production(self) -> bool:
        """프로덕션 환경인지 확인합니다."""
        return os.getenv('ENVIRONMENT', 'development').lower() == 'production' 