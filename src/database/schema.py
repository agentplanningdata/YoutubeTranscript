"""
SQLite 데이터베이스 스키마 관리

LangChain SQLDatabase를 활용한 SQLite 기반 데이터베이스 스키마 생성 및 관리를 담당합니다.
YouTube Transcript 프로젝트의 채널, 영상, 자막 데이터를 위한 테이블을 관리합니다.
"""

import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

# LangChain SQL 관련 임포트
from langchain_community.utilities import SQLDatabase

# 프로젝트 모듈 임포트
from src.utils.exceptions import DatabaseError
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)


class DatabaseSchema:
    """
    SQLite 데이터베이스 스키마 관리 클래스
    
    LangChain SQLDatabase를 활용하여 YouTube Transcript 프로젝트의
    데이터베이스 스키마를 생성하고 관리합니다.
    """
    
    def __init__(self, db_url: str):
        """
        데이터베이스 스키마 관리자를 초기화합니다.
        
        Args:
            db_url: SQLite 데이터베이스 URL (예: sqlite:///youtube.db)
        """
        try:
            self.db_url = db_url
            self.db = SQLDatabase.from_uri(db_url)
            
            # SQLite 외래키 활성화
            self._enable_foreign_keys()
            
            logger.info(f"DatabaseSchema initialized with: {db_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseSchema: {e}")
            raise DatabaseError(f"Failed to connect to database: {e}", operation="initialization")
    
    def _enable_foreign_keys(self):
        """SQLite 외래키 제약조건을 활성화합니다."""
        try:
            self.db.run("PRAGMA foreign_keys = ON;")
            logger.debug("Foreign keys enabled")
        except Exception as e:
            logger.warning(f"Failed to enable foreign keys: {e}")
    
    def create_channel_table(self):
        """
        채널 테이블을 생성합니다.
        
        채널의 메타데이터와 와치리스트 관리 정보를 저장합니다.
        """
        try:
            create_sql = """
            CREATE TABLE IF NOT EXISTS channels (
                channel_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                thumbnail_url TEXT NOT NULL,
                published_at TEXT NOT NULL,
                subscriber_count INTEGER,
                video_count INTEGER,
                custom_url TEXT,
                country TEXT,
                default_language TEXT,
                added_to_watchlist_at TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            self.db.run(create_sql)
            
            # 인덱스 생성
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_channels_title ON channels(title)",
                "CREATE INDEX IF NOT EXISTS idx_channels_is_active ON channels(is_active)",
                "CREATE INDEX IF NOT EXISTS idx_channels_added_to_watchlist ON channels(added_to_watchlist_at)"
            ]
            
            for index_sql in indexes:
                self.db.run(index_sql)
            
            logger.info("Channel table created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create channel table: {e}")
            raise DatabaseError(f"Failed to create channel table: {e}", operation="create_table")
    
    def create_video_table(self):
        """
        영상 테이블을 생성합니다.
        
        YouTube 영상의 메타데이터와 처리 상태 정보를 저장합니다.
        """
        try:
            create_sql = """
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                channel_id TEXT NOT NULL,
                published_at TEXT NOT NULL,
                thumbnail_url TEXT NOT NULL,
                duration TEXT,
                view_count INTEGER,
                like_count INTEGER,
                comment_count INTEGER,
                category_id TEXT,
                tags TEXT DEFAULT '',
                is_processed BOOLEAN DEFAULT FALSE,
                processing_status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id) ON DELETE CASCADE
            )
            """
            
            self.db.run(create_sql)
            
            # 인덱스 생성
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id)",
                "CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at)",
                "CREATE INDEX IF NOT EXISTS idx_videos_is_processed ON videos(is_processed)",
                "CREATE INDEX IF NOT EXISTS idx_videos_processing_status ON videos(processing_status)"
            ]
            
            for index_sql in indexes:
                self.db.run(index_sql)
            
            logger.info("Video table created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create video table: {e}")
            raise DatabaseError(f"Failed to create video table: {e}", operation="create_table")
    
    def create_transcript_table(self):
        """
        자막 테이블을 생성합니다.
        
        자막 청크와 메타데이터를 저장합니다.
        """
        try:
            create_sql = """
            CREATE TABLE IF NOT EXISTS transcripts (
                transcript_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text_content TEXT NOT NULL,
                start_time REAL,
                end_time REAL,
                language TEXT DEFAULT 'ko',
                chunk_hash TEXT NOT NULL,
                token_count INTEGER,
                character_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            )
            """
            
            self.db.run(create_sql)
            
            # 인덱스 생성
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_transcripts_video_id ON transcripts(video_id)",
                "CREATE INDEX IF NOT EXISTS idx_transcripts_chunk_hash ON transcripts(chunk_hash)",
                "CREATE INDEX IF NOT EXISTS idx_transcripts_time_range ON transcripts(start_time, end_time)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_unique_chunk ON transcripts(video_id, chunk_index)"
            ]
            
            for index_sql in indexes:
                self.db.run(index_sql)
            
            logger.info("Transcript table created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create transcript table: {e}")
            raise DatabaseError(f"Failed to create transcript table: {e}", operation="create_table")
    
    def create_all_tables(self):
        """
        모든 테이블을 순서대로 생성합니다.
        
        외래키 제약조건을 고려하여 올바른 순서로 테이블을 생성합니다.
        """
        try:
            logger.info("Creating all database tables...")
            
            # 순서 중요: 부모 테이블부터 생성
            self.create_channel_table()
            self.create_video_table()
            self.create_transcript_table()
            
            logger.info("All database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create all tables: {e}")
            raise DatabaseError(f"Failed to create all tables: {e}", operation="create_all_tables")
    
    def drop_all_tables(self):
        """
        모든 테이블을 삭제합니다.
        
        외래키 제약조건을 고려하여 올바른 순서로 테이블을 삭제합니다.
        """
        try:
            logger.info("Dropping all database tables...")
            
            # 순서 중요: 자식 테이블부터 삭제
            tables = ['transcripts', 'videos', 'channels']
            
            for table in tables:
                self.db.run(f"DROP TABLE IF EXISTS {table}")
            
            logger.info("All database tables dropped successfully")
            
        except Exception as e:
            logger.error(f"Failed to drop all tables: {e}")
            raise DatabaseError(f"Failed to drop all tables: {e}", operation="drop_all_tables")
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        특정 테이블의 정보를 조회합니다.
        
        Args:
            table_name: 조회할 테이블명
            
        Returns:
            테이블 정보 딕셔너리
        """
        try:
            table_info = self.db.get_table_info([table_name])
            return {"table_name": table_name, "info": table_info}
            
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            raise DatabaseError(f"Failed to get table info: {e}", operation="get_table_info")
    
    def get_all_tables(self) -> List[str]:
        """
        데이터베이스의 모든 테이블 목록을 조회합니다.
        
        Returns:
            테이블명 리스트
        """
        try:
            return self.db.get_usable_table_names()
            
        except Exception as e:
            logger.error(f"Failed to get table list: {e}")
            raise DatabaseError(f"Failed to get table list: {e}", operation="get_all_tables")
    
    def execute_sql(self, sql: str) -> Any:
        """
        SQL 쿼리를 직접 실행합니다.
        
        Args:
            sql: 실행할 SQL 문
            
        Returns:
            쿼리 실행 결과
        """
        try:
            logger.debug(f"Executing SQL: {sql[:100]}...")
            return self.db.run(sql)
            
        except Exception as e:
            logger.error(f"Failed to execute SQL: {e}")
            raise DatabaseError(f"Failed to execute SQL: {e}", operation="execute_sql")
    
    def check_foreign_keys(self) -> bool:
        """
        외래키 제약조건이 활성화되어 있는지 확인합니다.
        
        Returns:
            외래키 활성화 여부
        """
        try:
            result = self.db.run("PRAGMA foreign_keys;")
            return "1" in str(result)
            
        except Exception as e:
            logger.error(f"Failed to check foreign keys: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        데이터베이스 전체 정보를 조회합니다.
        
        Returns:
            데이터베이스 정보 딕셔너리
        """
        try:
            info = {
                "db_url": self.db_url,
                "tables": self.get_all_tables(),
                "foreign_keys_enabled": self.check_foreign_keys(),
                "created_at": datetime.now().isoformat()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            raise DatabaseError(f"Failed to get database info: {e}", operation="get_database_info")


# 팩토리 함수
def create_database_schema(db_url: str) -> DatabaseSchema:
    """
    DatabaseSchema 인스턴스를 생성합니다.
    
    Args:
        db_url: SQLite 데이터베이스 URL
        
    Returns:
        DatabaseSchema 인스턴스
    """
    return DatabaseSchema(db_url) 