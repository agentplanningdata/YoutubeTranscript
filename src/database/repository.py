"""
데이터베이스 저장소 클래스들

SQLite + LangChain 기반 데이터베이스 저장소를 제공합니다.
채널, 영상, 자막 데이터의 CRUD 연산을 담당합니다.
"""

import sqlite3
from typing import List, Optional, Dict, Any
from datetime import datetime

# LangChain SQL 관련 임포트
from langchain_community.utilities import SQLDatabase

# 프로젝트 모듈 임포트
from src.database.schema import DatabaseSchema
from src.services.models import ChannelInfo, VideoInfo
from src.utils.exceptions import DatabaseError
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)


class ChannelRepository:
    """
    채널 관리 저장소 클래스
    
    SQLite + LangChain을 활용하여 YouTube 채널의 CRUD 연산을 제공합니다.
    와치리스트 관리와 채널 메타데이터 관리를 담당합니다.
    """
    
    def __init__(self, db_url: str):
        """
        채널 저장소를 초기화합니다.
        
        Args:
            db_url: SQLite 데이터베이스 URL (예: sqlite:///youtube.db)
        """
        try:
            self.db_url = db_url
            self.db = SQLDatabase.from_uri(db_url)
            self.schema = DatabaseSchema(db_url)
            
            # SQLite 외래키 활성화
            self._enable_foreign_keys()
            
            logger.info(f"ChannelRepository initialized with: {db_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChannelRepository: {e}")
            raise DatabaseError(f"Failed to connect to database: {e}", operation="initialization")
    
    def _enable_foreign_keys(self):
        """SQLite 외래키 제약조건을 활성화합니다."""
        try:
            self.db.run("PRAGMA foreign_keys = ON;")
            logger.debug("Foreign keys enabled")
        except Exception as e:
            logger.warning(f"Failed to enable foreign keys: {e}")
    
    def initialize_tables(self):
        """
        필요한 테이블들을 초기화합니다.
        
        데이터베이스 스키마가 없는 경우 생성합니다.
        """
        try:
            self.schema.create_all_tables()
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tables: {e}")
            raise DatabaseError(f"Failed to initialize tables: {e}", operation="initialize_tables")
    
    def add_channel_to_watchlist(self, channel_info: ChannelInfo) -> bool:
        """
        와치리스트에 채널을 추가합니다.
        
        Args:
            channel_info: 추가할 채널 정보
            
        Returns:
            추가 성공 여부
        """
        try:
            # 필수 필드 검증
            if not channel_info.title.strip():
                raise DatabaseError("Channel title is required", operation="add_channel")
            
            # 중복 확인
            existing = self._get_channel_by_id(channel_info.channel_id)
            if existing:
                logger.warning(f"Channel {channel_info.channel_id} already exists in watchlist")
                return False
            
            # 현재 시간으로 추가 시간 설정
            current_time = datetime.now().isoformat()
            
            # INSERT 쿼리 실행
            insert_sql = """
            INSERT INTO channels (
                channel_id, title, description, thumbnail_url, published_at,
                subscriber_count, video_count, custom_url, country, default_language,
                added_to_watchlist_at, is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                channel_info.channel_id,
                channel_info.title,
                channel_info.description or '',
                channel_info.thumbnail_url,
                channel_info.published_at,
                channel_info.subscriber_count,
                channel_info.video_count,
                channel_info.custom_url,
                channel_info.country,
                channel_info.default_language,
                current_time,
                True,
                current_time,
                current_time
            )
            
            # SQLite에서는 파라미터 바인딩을 위해 직접 실행
            self._execute_with_params(insert_sql, values)
            
            logger.info(f"Channel {channel_info.channel_id} added to watchlist successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add channel to watchlist: {e}")
            raise DatabaseError(f"Failed to add channel: {e}", operation="add_channel")
    
    def get_watchlist_channels(self, is_active: Optional[bool] = True, channel_ids: Optional[List[str]] = None) -> List[ChannelInfo]:
        """
        와치리스트 채널들을 조회합니다.
        
        Args:
            is_active: 활성 채널만 조회할지 여부 (None이면 전체 조회)
            channel_ids: 특정 채널 ID들만 조회할 경우
            
        Returns:
            ChannelInfo 객체 리스트
        """
        try:
            # 기본 쿼리
            query = "SELECT * FROM channels WHERE 1=1"
            params = []
            
            # 활성 상태 필터
            if is_active is not None:
                query += " AND is_active = ?"
                params.append(is_active)
            
            # 특정 채널 ID 필터
            if channel_ids:
                placeholders = ",".join(["?" for _ in channel_ids])
                query += f" AND channel_id IN ({placeholders})"
                params.extend(channel_ids)
            
            # 정렬
            query += " ORDER BY added_to_watchlist_at DESC"
            
            # 쿼리 실행
            results = self._execute_with_params(query, params)
            
            # ChannelInfo 객체로 변환
            channels = []
            for row in results:
                channel_data = {
                    'channel_id': row[0],
                    'title': row[1],
                    'description': row[2],
                    'thumbnail_url': row[3],
                    'published_at': row[4],
                    'subscriber_count': row[5],
                    'video_count': row[6],
                    'custom_url': row[7],
                    'country': row[8],
                    'default_language': row[9]
                }
                channels.append(ChannelInfo.from_dict(channel_data))
            
            logger.debug(f"Retrieved {len(channels)} channels from watchlist")
            return channels
            
        except Exception as e:
            logger.error(f"Failed to get watchlist channels: {e}")
            raise DatabaseError(f"Failed to get channels: {e}", operation="get_channels")
    
    def remove_channel_from_watchlist(self, channel_id: str) -> bool:
        """
        와치리스트에서 채널을 제거합니다.
        
        Args:
            channel_id: 제거할 채널 ID
            
        Returns:
            제거 성공 여부
        """
        try:
            # 존재 확인
            existing = self._get_channel_by_id(channel_id)
            if not existing:
                logger.warning(f"Channel {channel_id} not found in watchlist")
                return False
            
            # DELETE 쿼리 실행
            delete_sql = "DELETE FROM channels WHERE channel_id = ?"
            result = self._execute_with_params(delete_sql, (channel_id,))
            
            logger.info(f"Channel {channel_id} removed from watchlist successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove channel from watchlist: {e}")
            raise DatabaseError(f"Failed to remove channel: {e}", operation="remove_channel")
    
    def update_channel_metadata(self, channel_info: ChannelInfo) -> bool:
        """
        채널 메타데이터를 업데이트합니다.
        
        Args:
            channel_info: 업데이트할 채널 정보
            
        Returns:
            업데이트 성공 여부
        """
        try:
            # 존재 확인
            existing = self._get_channel_by_id(channel_info.channel_id)
            if not existing:
                logger.warning(f"Channel {channel_info.channel_id} not found for update")
                return False
            
            # 현재 시간
            current_time = datetime.now().isoformat()
            
            # UPDATE 쿼리 실행
            update_sql = """
            UPDATE channels 
            SET title = ?, description = ?, thumbnail_url = ?, 
                subscriber_count = ?, video_count = ?, custom_url = ?,
                country = ?, default_language = ?, updated_at = ?
            WHERE channel_id = ?
            """
            
            values = (
                channel_info.title,
                channel_info.description or '',
                channel_info.thumbnail_url,
                channel_info.subscriber_count,
                channel_info.video_count,
                channel_info.custom_url,
                channel_info.country,
                channel_info.default_language,
                current_time,
                channel_info.channel_id
            )
            
            self._execute_with_params(update_sql, values)
            
            logger.info(f"Channel {channel_info.channel_id} metadata updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update channel metadata: {e}")
            raise DatabaseError(f"Failed to update channel: {e}", operation="update_channel")
    
    def deactivate_channel(self, channel_id: str) -> bool:
        """
        채널을 비활성화합니다 (소프트 삭제).
        
        Args:
            channel_id: 비활성화할 채널 ID
            
        Returns:
            비활성화 성공 여부
        """
        try:
            # 존재 확인
            existing = self._get_channel_by_id(channel_id)
            if not existing:
                logger.warning(f"Channel {channel_id} not found for deactivation")
                return False
            
            # UPDATE 쿼리 실행
            update_sql = """
            UPDATE channels 
            SET is_active = ?, updated_at = ?
            WHERE channel_id = ?
            """
            
            current_time = datetime.now().isoformat()
            values = (False, current_time, channel_id)
            
            self._execute_with_params(update_sql, values)
            
            logger.info(f"Channel {channel_id} deactivated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate channel: {e}")
            raise DatabaseError(f"Failed to deactivate channel: {e}", operation="deactivate_channel")
    
    def update_channel_field(self, channel_id: str, field: str, value: Any) -> bool:
        """
        채널의 특정 필드만 업데이트합니다.
        
        Args:
            channel_id: 업데이트할 채널 ID
            field: 업데이트할 필드명
            value: 새로운 값
            
        Returns:
            업데이트 성공 여부
        """
        try:
            # 존재 확인
            existing = self._get_channel_by_id(channel_id)
            if not existing:
                logger.warning(f"Channel {channel_id} not found for field update")
                return False
            
            # 허용되는 필드 확인
            allowed_fields = ['title', 'description', 'subscriber_count', 'video_count', 'custom_url']
            if field not in allowed_fields:
                raise DatabaseError(f"Field {field} is not allowed for update", operation="update_field")
            
            # UPDATE 쿼리 실행
            current_time = datetime.now().isoformat()
            update_sql = f"UPDATE channels SET {field} = ?, updated_at = ? WHERE channel_id = ?"
            
            self._execute_with_params(update_sql, (value, current_time, channel_id))
            
            logger.info(f"Channel {channel_id} field {field} updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update channel field: {e}")
            raise DatabaseError(f"Failed to update field: {e}", operation="update_field")
    
    def _get_channel_by_id(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        채널 ID로 채널 정보를 조회합니다.
        
        Args:
            channel_id: 조회할 채널 ID
            
        Returns:
            채널 데이터 딕셔너리 또는 None
        """
        try:
            query = "SELECT * FROM channels WHERE channel_id = ?"
            results = self._execute_with_params(query, (channel_id,))
            
            if results:
                return dict(results[0]) if hasattr(results[0], 'keys') else results[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get channel by ID: {e}")
            return None
    
    def _execute_with_params(self, query: str, params: tuple = ()) -> List[Any]:
        """
        파라미터와 함께 SQL 쿼리를 실행합니다.
        
        Args:
            query: SQL 쿼리
            params: 쿼리 파라미터
            
        Returns:
            쿼리 결과 리스트
        """
        try:
            # SQLite 연결을 직접 사용 (LangChain이 파라미터 바인딩을 완전히 지원하지 않음)
            import sqlite3
            
            # DB URL에서 파일 경로 추출
            db_path = self.db_url.replace('sqlite:///', '')
            
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")  # 외래키 활성화
                cursor = conn.cursor()
                
                if query.strip().upper().startswith('SELECT'):
                    cursor.execute(query, params)
                    return cursor.fetchall()
                else:
                    cursor.execute(query, params)
                    conn.commit()
                    return []
        
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise DatabaseError(f"Query execution failed: {e}", operation="execute_query")


class VideoRepository:
    """
    영상 관리 저장소 클래스
    
    SQLite + LangChain을 활용하여 YouTube 영상의 CRUD 연산을 제공합니다.
    영상 정보 저장, 처리 상태 관리, 채널별 조회 기능을 담당합니다.
    """
    
    def __init__(self, db_url: str):
        """
        영상 저장소를 초기화합니다.
        
        Args:
            db_url: SQLite 데이터베이스 URL (예: sqlite:///youtube.db)
        """
        try:
            self.db_url = db_url
            self.db = SQLDatabase.from_uri(db_url)
            self.schema = DatabaseSchema(db_url)
            
            # SQLite 외래키 활성화
            self._enable_foreign_keys()
            
            logger.info(f"VideoRepository initialized with: {db_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize VideoRepository: {e}")
            raise DatabaseError(f"Failed to connect to database: {e}", operation="initialization")
    
    def _enable_foreign_keys(self):
        """SQLite 외래키 제약조건을 활성화합니다."""
        try:
            self.db.run("PRAGMA foreign_keys = ON;")
            logger.debug("Foreign keys enabled")
        except Exception as e:
            logger.warning(f"Failed to enable foreign keys: {e}")
    
    def initialize_tables(self):
        """
        필요한 테이블들을 초기화합니다.
        
        데이터베이스 스키마가 없는 경우 생성합니다.
        """
        try:
            self.schema.create_all_tables()
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tables: {e}")
            raise DatabaseError(f"Failed to initialize tables: {e}", operation="initialize_tables")
    
    def add_channel_for_testing(self, channel_info: ChannelInfo):
        """
        테스트용으로 채널을 추가합니다.
        
        Args:
            channel_info: 추가할 채널 정보
        """
        try:
            current_time = datetime.now().isoformat()
            
            insert_sql = """
            INSERT OR IGNORE INTO channels (
                channel_id, title, description, thumbnail_url, published_at,
                subscriber_count, video_count, custom_url, country, default_language,
                added_to_watchlist_at, is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                channel_info.channel_id,
                channel_info.title,
                channel_info.description or '',
                channel_info.thumbnail_url,
                channel_info.published_at,
                channel_info.subscriber_count,
                channel_info.video_count,
                channel_info.custom_url,
                channel_info.country,
                channel_info.default_language,
                current_time,
                True,
                current_time,
                current_time
            )
            
            self._execute_with_params(insert_sql, values)
            logger.debug(f"Test channel {channel_info.channel_id} added")
            
        except Exception as e:
            logger.error(f"Failed to add test channel: {e}")
            # 테스트 중이므로 예외를 다시 발생시키지 않음
    
    def save_video_info(self, video_info: VideoInfo) -> bool:
        """
        영상 정보를 저장합니다.
        
        Args:
            video_info: 저장할 영상 정보
            
        Returns:
            저장 성공 여부
        """
        try:
            # 중복 확인
            existing = self._get_video_by_id_internal(video_info.video_id)
            if existing:
                logger.warning(f"Video {video_info.video_id} already exists")
                return False
            
            # 현재 시간
            current_time = datetime.now().isoformat()
            
            # tags를 문자열로 변환
            tags_str = ','.join(video_info.tags) if video_info.tags else ''
            
            # INSERT 쿼리 실행
            insert_sql = """
            INSERT INTO videos (
                video_id, title, description, channel_id, published_at, thumbnail_url,
                duration, view_count, like_count, comment_count, category_id, tags,
                is_processed, processing_status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                video_info.video_id,
                video_info.title,
                video_info.description or '',
                video_info.channel_id,
                video_info.published_at,
                video_info.thumbnail_url,
                video_info.duration,
                video_info.view_count,
                video_info.like_count,
                video_info.comment_count,
                video_info.category_id,
                tags_str,
                False,  # is_processed
                'pending',  # processing_status
                current_time,
                current_time
            )
            
            self._execute_with_params(insert_sql, values)
            
            logger.info(f"Video {video_info.video_id} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save video info: {e}")
            raise DatabaseError(f"Failed to save video: {e}", operation="save_video")
    
    def get_video_by_id(self, video_id: str) -> Optional[VideoInfo]:
        """
        영상 ID로 영상 정보를 조회합니다.
        
        Args:
            video_id: 조회할 영상 ID
            
        Returns:
            VideoInfo 객체 또는 None
        """
        try:
            query = "SELECT * FROM videos WHERE video_id = ?"
            results = self._execute_with_params(query, (video_id,))
            
            if results:
                return self._row_to_video_info(results[0])
            return None
            
        except Exception as e:
            logger.error(f"Failed to get video by ID: {e}")
            raise DatabaseError(f"Failed to get video: {e}", operation="get_video")
    
    def get_videos_by_channel(self, channel_id: str) -> List[VideoInfo]:
        """
        채널 ID로 영상 목록을 조회합니다.
        
        Args:
            channel_id: 조회할 채널 ID
            
        Returns:
            VideoInfo 객체 리스트
        """
        try:
            query = "SELECT * FROM videos WHERE channel_id = ? ORDER BY published_at DESC"
            results = self._execute_with_params(query, (channel_id,))
            
            return [self._row_to_video_info(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get videos by channel: {e}")
            raise DatabaseError(f"Failed to get videos: {e}", operation="get_videos")
    
    def get_latest_videos_by_channel(self, channel_id: str, limit: int = 10) -> List[VideoInfo]:
        """
        채널별 최신 영상을 조회합니다.
        
        Args:
            channel_id: 조회할 채널 ID
            limit: 조회할 영상 수 제한
            
        Returns:
            VideoInfo 객체 리스트
        """
        try:
            query = """
            SELECT * FROM videos 
            WHERE channel_id = ? 
            ORDER BY published_at DESC 
            LIMIT ?
            """
            results = self._execute_with_params(query, (channel_id, limit))
            
            return [self._row_to_video_info(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get latest videos: {e}")
            raise DatabaseError(f"Failed to get latest videos: {e}", operation="get_latest_videos")
    
    def mark_video_as_processed(self, video_id: str) -> bool:
        """
        영상을 처리 완료로 마킹합니다.
        
        Args:
            video_id: 처리 완료할 영상 ID
            
        Returns:
            처리 성공 여부
        """
        try:
            # 존재 확인
            existing = self._get_video_by_id_internal(video_id)
            if not existing:
                logger.warning(f"Video {video_id} not found for processing")
                return False
            
            # UPDATE 쿼리 실행
            current_time = datetime.now().isoformat()
            update_sql = """
            UPDATE videos 
            SET is_processed = ?, processing_status = ?, updated_at = ?
            WHERE video_id = ?
            """
            
            values = (True, 'completed', current_time, video_id)
            self._execute_with_params(update_sql, values)
            
            logger.info(f"Video {video_id} marked as processed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark video as processed: {e}")
            raise DatabaseError(f"Failed to mark video: {e}", operation="mark_processed")
    
    def get_unprocessed_videos(self, limit: Optional[int] = None, order_by: str = 'oldest') -> List[VideoInfo]:
        """
        미처리 영상 목록을 조회합니다.
        
        Args:
            limit: 조회할 영상 수 제한
            order_by: 정렬 방식 ('oldest', 'latest')
            
        Returns:
            VideoInfo 객체 리스트
        """
        try:
            order_clause = "published_at ASC" if order_by == 'oldest' else "published_at DESC"
            
            query = f"""
            SELECT * FROM videos 
            WHERE is_processed = 0
            ORDER BY {order_clause}
            """
            
            if limit:
                query += " LIMIT ?"
                results = self._execute_with_params(query, (limit,))
            else:
                results = self._execute_with_params(query, ())
            
            return [self._row_to_video_info(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get unprocessed videos: {e}")
            raise DatabaseError(f"Failed to get unprocessed videos: {e}", operation="get_unprocessed")
    
    def update_processing_status(self, video_id: str, status: str) -> bool:
        """
        영상의 처리 상태를 업데이트합니다.
        
        Args:
            video_id: 업데이트할 영상 ID
            status: 새로운 처리 상태
            
        Returns:
            업데이트 성공 여부
        """
        try:
            # 존재 확인
            existing = self._get_video_by_id_internal(video_id)
            if not existing:
                logger.warning(f"Video {video_id} not found for status update")
                return False
            
            # UPDATE 쿼리 실행
            current_time = datetime.now().isoformat()
            update_sql = """
            UPDATE videos 
            SET processing_status = ?, updated_at = ?
            WHERE video_id = ?
            """
            
            values = (status, current_time, video_id)
            self._execute_with_params(update_sql, values)
            
            logger.info(f"Video {video_id} processing status updated to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update processing status: {e}")
            raise DatabaseError(f"Failed to update status: {e}", operation="update_status")
    
    def save_videos_batch(self, videos: List[VideoInfo]) -> List[bool]:
        """
        여러 영상을 일괄 저장합니다.
        
        Args:
            videos: 저장할 영상 정보 리스트
            
        Returns:
            각 영상별 저장 성공 여부 리스트
        """
        results = []
        for video in videos:
            try:
                result = self.save_video_info(video)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to save video {video.video_id} in batch: {e}")
                results.append(False)
        
        return results
    
    def mark_videos_batch_as_processed(self, video_ids: List[str]) -> List[bool]:
        """
        여러 영상을 일괄로 처리 완료 마킹합니다.
        
        Args:
            video_ids: 처리 완료할 영상 ID 리스트
            
        Returns:
            각 영상별 처리 성공 여부 리스트
        """
        results = []
        for video_id in video_ids:
            try:
                result = self.mark_video_as_processed(video_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to mark video {video_id} in batch: {e}")
                results.append(False)
        
        return results
    
    def get_unprocessed_videos_by_channel(self, channel_id: str) -> List[VideoInfo]:
        """
        특정 채널의 미처리 영상 목록을 조회합니다.
        
        Args:
            channel_id: 조회할 채널 ID
            
        Returns:
            VideoInfo 객체 리스트
        """
        try:
            query = """
            SELECT * FROM videos 
            WHERE channel_id = ? AND is_processed = 0
            ORDER BY published_at ASC
            """
            results = self._execute_with_params(query, (channel_id,))
            
            return [self._row_to_video_info(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get unprocessed videos by channel: {e}")
            raise DatabaseError(f"Failed to get unprocessed videos: {e}", operation="get_unprocessed_by_channel")
    
    def get_videos_by_channel_after_date(self, channel_id: str, after_date: str) -> List[VideoInfo]:
        """
        특정 날짜 이후의 채널 영상을 조회합니다.
        
        Args:
            channel_id: 조회할 채널 ID
            after_date: 기준 날짜 (ISO 8601 형식)
            
        Returns:
            VideoInfo 객체 리스트
        """
        try:
            query = """
            SELECT * FROM videos 
            WHERE channel_id = ? AND published_at > ?
            ORDER BY published_at DESC
            """
            results = self._execute_with_params(query, (channel_id, after_date))
            
            return [self._row_to_video_info(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get videos after date: {e}")
            raise DatabaseError(f"Failed to get videos after date: {e}", operation="get_videos_after_date")
    
    def _get_video_by_id_internal(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        내부용 영상 ID 조회 메서드
        
        Args:
            video_id: 조회할 영상 ID
            
        Returns:
            영상 데이터 딕셔너리 또는 None
        """
        try:
            query = "SELECT * FROM videos WHERE video_id = ?"
            results = self._execute_with_params(query, (video_id,))
            
            if results:
                return dict(results[0]) if hasattr(results[0], 'keys') else results[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get video by ID (internal): {e}")
            return None
    
    def _row_to_video_info(self, row) -> VideoInfo:
        """
        데이터베이스 행을 VideoInfo 객체로 변환합니다.
        
        Args:
            row: 데이터베이스 행 데이터
            
        Returns:
            VideoInfo 객체
        """
        try:
            # tags 문자열을 리스트로 변환
            tags_str = row[11] if row[11] else ''
            tags = tags_str.split(',') if tags_str else []
            
            # VideoInfo 객체 생성을 위한 데이터 준비
            video_data = {
                'video_id': row[0],
                'title': row[1], 
                'description': row[2],
                'channel_id': row[3],
                'published_at': row[4],
                'thumbnail_url': row[5],
                'duration': row[6],
                'view_count': row[7],
                'like_count': row[8],
                'comment_count': row[9],
                'category_id': row[10],
                'tags': tags
            }
            
            # VideoInfo 객체 생성
            video_info = VideoInfo.from_dict(video_data)
            
            # 처리 상태 정보 추가 (VideoInfo 모델에 없는 필드들은 메타데이터로)
            video_info.add_metadata('is_processed', bool(row[12]))
            video_info.add_metadata('processing_status', row[13])
            video_info.add_metadata('created_at', row[14])
            video_info.add_metadata('updated_at', row[15])
            
            return video_info
            
        except Exception as e:
            logger.error(f"Failed to convert row to VideoInfo: {e}")
            raise DatabaseError(f"Failed to convert data: {e}", operation="row_conversion")
    
    def _execute_with_params(self, query: str, params: tuple = ()) -> List[Any]:
        """
        파라미터와 함께 SQL 쿼리를 실행합니다.
        
        Args:
            query: SQL 쿼리
            params: 쿼리 파라미터
            
        Returns:
            쿼리 결과 리스트
        """
        try:
            # SQLite 연결을 직접 사용
            import sqlite3
            
            # DB URL에서 파일 경로 추출
            db_path = self.db_url.replace('sqlite:///', '')
            
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")  # 외래키 활성화
                cursor = conn.cursor()
                
                if query.strip().upper().startswith('SELECT'):
                    cursor.execute(query, params)
                    return cursor.fetchall()
                else:
                    cursor.execute(query, params)
                    conn.commit()
                    return []
        
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise DatabaseError(f"Query execution failed: {e}", operation="execute_query")


# 팩토리 함수
def create_channel_repository(db_url: str) -> ChannelRepository:
    """
    ChannelRepository 인스턴스를 생성합니다.
    
    Args:
        db_url: SQLite 데이터베이스 URL
        
    Returns:
        ChannelRepository 인스턴스
    """
    return ChannelRepository(db_url)


def create_video_repository(db_url: str) -> VideoRepository:
    """
    VideoRepository 인스턴스를 생성합니다.
    
    Args:
        db_url: SQLite 데이터베이스 URL
        
    Returns:
        VideoRepository 인스턴스
    """
    return VideoRepository(db_url) 