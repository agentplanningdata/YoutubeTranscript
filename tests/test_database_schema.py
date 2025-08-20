"""
SQLite 데이터베이스 스키마 테스트

LangChain SQLDatabase를 활용한 SQLite 기반 데이터베이스 스키마 생성 및 관리를 테스트합니다.
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime

# LangChain SQL 관련 임포트
from langchain_community.utilities import SQLDatabase

# 프로젝트 모듈 임포트
from src.services.models import ChannelInfo, VideoInfo
from src.utils.exceptions import DatabaseError


@pytest.fixture
def temp_db_path():
    """임시 SQLite 데이터베이스 파일 경로 생성"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_path = f.name
    
    # 파일 삭제 (빈 파일이 있으면 SQLite가 문제 생길 수 있음)
    os.unlink(temp_path)
    
    yield temp_path
    
    # 테스트 후 파일 정리
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_channel_data():
    """테스트용 채널 데이터"""
    return {
        'channel_id': 'UC1234567890',
        'title': '테스트 채널',
        'description': '이것은 테스트용 채널입니다',
        'thumbnail_url': 'https://example.com/thumbnail.jpg',
        'published_at': '2023-01-01T00:00:00Z',
        'subscriber_count': 10000,
        'video_count': 100,
        'custom_url': 'testchannel'
    }


@pytest.fixture  
def sample_video_data():
    """테스트용 영상 데이터"""
    return {
        'video_id': 'dQw4w9WgXcQ',
        'title': '테스트 영상',
        'description': '이것은 테스트용 영상입니다',
        'channel_id': 'UC1234567890',
        'channel_title': '테스트 채널',
        'published_at': '2023-01-15T12:00:00Z',
        'thumbnail_url': 'https://example.com/video_thumb.jpg',
        'duration': 'PT4M33S',
        'view_count': 50000,
        'like_count': 1000
    }


class TestChannelTableCreation:
    """채널 테이블 생성 테스트"""
    
    def test_channel_table_creation(self, temp_db_path):
        """채널 테이블 생성 테스트"""
        # 아직 DatabaseSchema 클래스가 없으므로 실패해야 함
        from src.database.schema import DatabaseSchema
        
        # SQLite 데이터베이스 연결
        db_url = f"sqlite:///{temp_db_path}"
        
        # 스키마 매니저 초기화
        schema = DatabaseSchema(db_url)
        
        # 채널 테이블 생성
        schema.create_channel_table()
        
        # 테이블 생성 확인
        db = SQLDatabase.from_uri(db_url)
        tables = db.get_usable_table_names()
        
        assert 'channels' in tables, "채널 테이블이 생성되지 않았습니다"
        
        # 테이블 스키마 확인
        table_info = db.get_table_info(['channels'])
        
        # 필수 컬럼 확인
        required_columns = [
            'channel_id', 'title', 'description', 'thumbnail_url', 
            'published_at', 'subscriber_count', 'video_count',
            'custom_url', 'country', 'default_language',
            'added_to_watchlist_at', 'is_active', 'created_at', 'updated_at'
        ]
        
        for column in required_columns:
            assert column in table_info, f"필수 컬럼 '{column}'이 없습니다"


class TestVideoTableCreation:
    """영상 테이블 생성 테스트"""
    
    def test_video_table_creation(self, temp_db_path):
        """영상 테이블 생성 테스트"""
        from src.database.schema import DatabaseSchema
        
        db_url = f"sqlite:///{temp_db_path}"
        schema = DatabaseSchema(db_url)
        
        # 외래키 참조 때문에 채널 테이블 먼저 생성
        schema.create_channel_table()
        
        # 영상 테이블 생성
        schema.create_video_table()
        
        # 테이블 생성 확인
        db = SQLDatabase.from_uri(db_url)
        tables = db.get_usable_table_names()
        
        assert 'videos' in tables, "영상 테이블이 생성되지 않았습니다"
        
        # 테이블 스키마 확인
        table_info = db.get_table_info(['videos'])
        
        # 필수 컬럼 확인
        required_columns = [
            'video_id', 'title', 'description', 'channel_id',
            'published_at', 'thumbnail_url', 'duration', 
            'view_count', 'like_count', 'comment_count',
            'category_id', 'tags', 'is_processed', 
            'processing_status', 'created_at', 'updated_at'
        ]
        
        for column in required_columns:
            assert column in table_info, f"필수 컬럼 '{column}'이 없습니다"


class TestTranscriptTableCreation:
    """자막 테이블 생성 테스트"""
    
    def test_transcript_table_creation(self, temp_db_path):
        """자막 테이블 생성 테스트"""
        from src.database.schema import DatabaseSchema
        
        db_url = f"sqlite:///{temp_db_path}"
        schema = DatabaseSchema(db_url)
        
        # 외래키 참조 때문에 부모 테이블들 먼저 생성
        schema.create_channel_table()
        schema.create_video_table()
        
        # 자막 테이블 생성
        schema.create_transcript_table()
        
        # 테이블 생성 확인
        db = SQLDatabase.from_uri(db_url)
        tables = db.get_usable_table_names()
        
        assert 'transcripts' in tables, "자막 테이블이 생성되지 않았습니다"
        
        # 테이블 스키마 확인
        table_info = db.get_table_info(['transcripts'])
        
        # 필수 컬럼 확인
        required_columns = [
            'transcript_id', 'video_id', 'chunk_index', 
            'text_content', 'start_time', 'end_time',
            'language', 'chunk_hash', 'token_count',
            'character_count', 'created_at'
        ]
        
        for column in required_columns:
            assert column in table_info, f"필수 컬럼 '{column}'이 없습니다"


class TestForeignKeyConstraints:
    """외래키 제약조건 테스트"""
    
    def test_foreign_key_constraints(self, temp_db_path):
        """외래키 제약조건 확인 테스트"""
        from src.database.schema import DatabaseSchema
        
        db_url = f"sqlite:///{temp_db_path}"
        schema = DatabaseSchema(db_url)
        
        # 모든 테이블 생성
        schema.create_all_tables()
        
        # 외래키 활성화 확인
        db = SQLDatabase.from_uri(db_url)
        
        # 새로운 연결에서도 외래키 활성화
        db.run("PRAGMA foreign_keys = ON;")
        
        # SQLite 외래키 설정 확인
        result = db.run("PRAGMA foreign_keys;")
        assert "1" in str(result), "외래키가 활성화되지 않았습니다"
        
        # videos 테이블의 외래키 제약조건 확인
        foreign_keys = db.run("PRAGMA foreign_key_list(videos);")
        
        # channel_id가 channels 테이블을 참조하는지 확인
        assert "channels" in str(foreign_keys), "videos.channel_id가 channels 테이블을 참조하지 않습니다"
        
        # transcripts 테이블의 외래키 제약조건 확인  
        transcript_fks = db.run("PRAGMA foreign_key_list(transcripts);")
        
        # video_id가 videos 테이블을 참조하는지 확인
        assert "videos" in str(transcript_fks), "transcripts.video_id가 videos 테이블을 참조하지 않습니다"


class TestDatabaseSchemaIntegration:
    """데이터베이스 스키마 통합 테스트"""
    
    def test_create_all_tables(self, temp_db_path):
        """모든 테이블 한번에 생성 테스트"""
        from src.database.schema import DatabaseSchema
        
        db_url = f"sqlite:///{temp_db_path}"
        schema = DatabaseSchema(db_url)
        
        # 모든 테이블 생성
        schema.create_all_tables()
        
        # 모든 테이블 존재 확인
        db = SQLDatabase.from_uri(db_url)
        tables = db.get_usable_table_names()
        
        expected_tables = ['channels', 'videos', 'transcripts']
        for table in expected_tables:
            assert table in tables, f"테이블 '{table}'이 생성되지 않았습니다"
    
    def test_drop_all_tables(self, temp_db_path):
        """모든 테이블 삭제 테스트"""
        from src.database.schema import DatabaseSchema
        
        db_url = f"sqlite:///{temp_db_path}"
        schema = DatabaseSchema(db_url)
        
        # 테이블 생성 후 삭제
        schema.create_all_tables()
        schema.drop_all_tables()
        
        # 테이블 삭제 확인
        db = SQLDatabase.from_uri(db_url)
        tables = db.get_usable_table_names()
        
        expected_tables = ['channels', 'videos', 'transcripts']
        for table in expected_tables:
            assert table not in tables, f"테이블 '{table}'이 삭제되지 않았습니다"
    
    def test_table_recreation_safety(self, temp_db_path):
        """테이블 재생성 안전성 테스트"""
        from src.database.schema import DatabaseSchema
        
        db_url = f"sqlite:///{temp_db_path}"
        schema = DatabaseSchema(db_url)
        
        # 첫 번째 생성
        schema.create_all_tables()
        
        # 재생성 (에러 없이 처리되어야 함)
        schema.create_all_tables()
        
        # 여전히 테이블이 존재하는지 확인
        db = SQLDatabase.from_uri(db_url)
        tables = db.get_usable_table_names()
        
        expected_tables = ['channels', 'videos', 'transcripts']
        for table in expected_tables:
            assert table in tables, f"재생성 후 테이블 '{table}'이 없습니다"


class TestDatabaseSchemaErrors:
    """데이터베이스 스키마 오류 처리 테스트"""
    
    def test_invalid_database_url(self):
        """잘못된 데이터베이스 URL 처리 테스트"""
        from src.database.schema import DatabaseSchema
        
        with pytest.raises(DatabaseError):
            # 잘못된 URL로 초기화
            DatabaseSchema("invalid://database/url")
    
    def test_permission_denied_database(self):
        """권한 없는 데이터베이스 경로 처리 테스트"""
        from src.database.schema import DatabaseSchema
        
        # 읽기 전용 경로 (예: /root/readonly.db)
        readonly_path = "/dev/null/readonly.db"
        
        with pytest.raises(DatabaseError):
            schema = DatabaseSchema(f"sqlite:///{readonly_path}")
            schema.create_all_tables() 