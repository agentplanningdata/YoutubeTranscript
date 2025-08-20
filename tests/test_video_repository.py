"""
영상 관리 저장소 테스트

SQLite + LangChain 기반 영상 CRUD 연산을 테스트합니다.
영상 정보 저장, 처리 상태 관리, 채널별 조회 기능을 포함합니다.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from typing import List, Optional

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
def sample_channel_info():
    """테스트용 채널 정보"""
    return ChannelInfo(
        channel_id='UC1234567890',
        title='테스트 채널',
        description='이것은 테스트용 채널입니다',
        thumbnail_url='https://example.com/thumbnail.jpg',
        published_at='2023-01-01T00:00:00Z',
        subscriber_count=10000,
        video_count=100
    )


@pytest.fixture
def sample_video_info():
    """테스트용 영상 정보"""
    return VideoInfo(
        video_id='dQw4w9WgXcQ',
        title='테스트 영상',
        description='이것은 테스트용 영상입니다',
        channel_id='UC1234567890',
        published_at='2023-01-15T12:00:00Z',
        thumbnail_url='https://example.com/video_thumb.jpg',
        duration='PT4M33S',
        view_count=50000,
        like_count=1000,
        tags=['테스트', '영상', 'YouTube']
    )


@pytest.fixture
def another_video_info():
    """추가 테스트용 영상 정보"""
    return VideoInfo(
        video_id='dQw4w9WgXcR',
        title='두 번째 테스트 영상',
        description='또 다른 테스트 영상',
        channel_id='UC1234567890',
        published_at='2023-01-20T15:30:00Z',
        thumbnail_url='https://example.com/video_thumb2.jpg',
        duration='PT7M15S',
        view_count=75000,
        like_count=1500
    )


@pytest.fixture
def initialized_repository(temp_db_path, sample_channel_info):
    """초기화된 영상 저장소"""
    from src.database.repository import VideoRepository
    
    db_url = f"sqlite:///{temp_db_path}"
    repository = VideoRepository(db_url)
    
    # 테이블 초기화
    repository.initialize_tables()
    
    # 외래키 참조를 위해 채널 먼저 추가
    repository.add_channel_for_testing(sample_channel_info)
    
    return repository


class TestSaveVideoInfo:
    """영상 정보 저장 테스트"""
    
    def test_save_video_info(self, initialized_repository, sample_video_info):
        """영상 정보 저장 테스트"""
        repository = initialized_repository
        
        # 영상 정보 저장
        result = repository.save_video_info(sample_video_info)
        
        assert result is True, "영상 정보 저장이 실패했습니다"
        
        # 저장된 영상 확인
        saved_video = repository.get_video_by_id(sample_video_info.video_id)
        
        assert saved_video is not None, "저장된 영상을 찾을 수 없습니다"
        assert saved_video.video_id == sample_video_info.video_id
        assert saved_video.title == sample_video_info.title
        assert saved_video.channel_id == sample_video_info.channel_id
    
    def test_save_duplicate_video_info(self, initialized_repository, sample_video_info):
        """중복 영상 정보 저장 방지 테스트"""
        repository = initialized_repository
        
        # 첫 번째 저장
        result1 = repository.save_video_info(sample_video_info)
        assert result1 is True
        
        # 중복 저장 시도
        result2 = repository.save_video_info(sample_video_info)
        assert result2 is False, "중복 영상이 저장되었습니다"
        
        # 영상 수가 1개인지 확인
        videos = repository.get_videos_by_channel(sample_video_info.channel_id)
        assert len(videos) == 1
    
    def test_save_video_with_invalid_channel(self, initialized_repository):
        """존재하지 않는 채널의 영상 저장 테스트"""
        repository = initialized_repository
        
        # 존재하지 않는 채널 ID
        invalid_video = VideoInfo(
            video_id='invalid123',
            title='잘못된 영상',
            description='존재하지 않는 채널의 영상',
            channel_id='UC_NONEXISTENT',
            published_at='2023-01-01T00:00:00Z',
            thumbnail_url='https://example.com/invalid.jpg'
        )
        
        with pytest.raises(DatabaseError):
            repository.save_video_info(invalid_video)
    
    def test_save_video_batch(self, initialized_repository, sample_video_info, another_video_info):
        """여러 영상 일괄 저장 테스트"""
        repository = initialized_repository
        
        videos = [sample_video_info, another_video_info]
        
        # 일괄 저장
        results = repository.save_videos_batch(videos)
        
        assert len(results) == 2, "일괄 저장 결과 수가 올바르지 않습니다"
        assert all(results), "일괄 저장 중 실패가 있습니다"
        
        # 저장 확인
        saved_videos = repository.get_videos_by_channel(sample_video_info.channel_id)
        assert len(saved_videos) == 2


class TestGetLatestVideosByChannel:
    """채널별 최신 영상 조회 테스트"""
    
    def test_get_latest_videos_by_channel(self, initialized_repository, sample_video_info, another_video_info):
        """채널별 최신 영상 조회 테스트"""
        repository = initialized_repository
        
        # 두 영상 저장
        repository.save_video_info(sample_video_info)
        repository.save_video_info(another_video_info)
        
        # 최신 영상 조회
        latest_videos = repository.get_latest_videos_by_channel(
            sample_video_info.channel_id, limit=10
        )
        
        assert len(latest_videos) == 2, "조회된 영상 수가 올바르지 않습니다"
        
        # 최신 순으로 정렬되었는지 확인 (another_video_info가 더 최신)
        assert latest_videos[0].video_id == another_video_info.video_id
        assert latest_videos[1].video_id == sample_video_info.video_id
    
    def test_get_latest_videos_with_limit(self, initialized_repository, sample_video_info, another_video_info):
        """영상 조회 개수 제한 테스트"""
        repository = initialized_repository
        
        # 두 영상 저장
        repository.save_video_info(sample_video_info)
        repository.save_video_info(another_video_info)
        
        # 1개만 조회
        latest_videos = repository.get_latest_videos_by_channel(
            sample_video_info.channel_id, limit=1
        )
        
        assert len(latest_videos) == 1, "제한된 개수가 조회되지 않았습니다"
        assert latest_videos[0].video_id == another_video_info.video_id  # 가장 최신
    
    def test_get_videos_from_nonexistent_channel(self, initialized_repository):
        """존재하지 않는 채널의 영상 조회 테스트"""
        repository = initialized_repository
        
        videos = repository.get_latest_videos_by_channel('UC_NONEXISTENT')
        
        assert videos == [], "존재하지 않는 채널에서 빈 리스트가 반환되지 않았습니다"
    
    def test_get_videos_with_date_filter(self, initialized_repository, sample_video_info):
        """날짜 필터링 영상 조회 테스트"""
        repository = initialized_repository
        
        # 영상 저장
        repository.save_video_info(sample_video_info)
        
        # 특정 날짜 이후 영상 조회
        after_date = '2023-01-10T00:00:00Z'
        recent_videos = repository.get_videos_by_channel_after_date(
            sample_video_info.channel_id, after_date
        )
        
        assert len(recent_videos) == 1, "날짜 필터링이 올바르지 않습니다"
        assert recent_videos[0].video_id == sample_video_info.video_id


class TestMarkVideoAsProcessed:
    """영상 처리 완료 마킹 테스트"""
    
    def test_mark_video_as_processed(self, initialized_repository, sample_video_info):
        """영상 처리 완료 마킹 테스트"""
        repository = initialized_repository
        
        # 영상 저장
        repository.save_video_info(sample_video_info)
        
        # 처리 완료 마킹
        result = repository.mark_video_as_processed(sample_video_info.video_id)
        
        assert result is True, "영상 처리 완료 마킹이 실패했습니다"
        
        # 처리 상태 확인
        video = repository.get_video_by_id(sample_video_info.video_id)
        assert video.get_metadata('is_processed') is True, "영상 처리 상태가 업데이트되지 않았습니다"
        assert video.get_metadata('processing_status') == 'completed', "처리 상태가 올바르지 않습니다"
    
    def test_mark_nonexistent_video_as_processed(self, initialized_repository):
        """존재하지 않는 영상 처리 완료 마킹 테스트"""
        repository = initialized_repository
        
        result = repository.mark_video_as_processed('NONEXISTENT_VIDEO')
        
        assert result is False, "존재하지 않는 영상 처리 마킹이 성공했습니다"
    
    def test_update_processing_status(self, initialized_repository, sample_video_info):
        """처리 상태 업데이트 테스트"""
        repository = initialized_repository
        
        # 영상 저장
        repository.save_video_info(sample_video_info)
        
        # 처리 중으로 상태 변경
        result = repository.update_processing_status(
            sample_video_info.video_id, 'processing'
        )
        
        assert result is True, "처리 상태 업데이트가 실패했습니다"
        
        # 상태 확인
        video = repository.get_video_by_id(sample_video_info.video_id)
        assert video.get_metadata('processing_status') == 'processing'
    
    def test_mark_videos_batch_as_processed(self, initialized_repository, sample_video_info, another_video_info):
        """여러 영상 일괄 처리 완료 마킹 테스트"""
        repository = initialized_repository
        
        # 영상들 저장
        repository.save_video_info(sample_video_info)
        repository.save_video_info(another_video_info)
        
        video_ids = [sample_video_info.video_id, another_video_info.video_id]
        
        # 일괄 처리 완료 마킹
        results = repository.mark_videos_batch_as_processed(video_ids)
        
        assert len(results) == 2, "일괄 처리 결과 수가 올바르지 않습니다"
        assert all(results), "일괄 처리 중 실패가 있습니다"


class TestGetUnprocessedVideos:
    """미처리 영상 목록 조회 테스트"""
    
    def test_get_unprocessed_videos(self, initialized_repository, sample_video_info, another_video_info):
        """미처리 영상 목록 조회 테스트"""
        repository = initialized_repository
        
        # 두 영상 저장
        repository.save_video_info(sample_video_info)
        repository.save_video_info(another_video_info)
        
        # 한 영상만 처리 완료 마킹
        repository.mark_video_as_processed(sample_video_info.video_id)
        
        # 미처리 영상 조회
        unprocessed = repository.get_unprocessed_videos()
        
        assert len(unprocessed) == 1, "미처리 영상 수가 올바르지 않습니다"
        assert unprocessed[0].video_id == another_video_info.video_id
    
    def test_get_unprocessed_videos_by_channel(self, initialized_repository, sample_video_info, another_video_info):
        """채널별 미처리 영상 조회 테스트"""
        repository = initialized_repository
        
        # 두 영상 저장
        repository.save_video_info(sample_video_info)
        repository.save_video_info(another_video_info)
        
        # 채널별 미처리 영상 조회
        unprocessed = repository.get_unprocessed_videos_by_channel(
            sample_video_info.channel_id
        )
        
        assert len(unprocessed) == 2, "채널별 미처리 영상 수가 올바르지 않습니다"
    
    def test_get_unprocessed_videos_with_limit(self, initialized_repository, sample_video_info, another_video_info):
        """미처리 영상 개수 제한 조회 테스트"""
        repository = initialized_repository
        
        # 두 영상 저장
        repository.save_video_info(sample_video_info)
        repository.save_video_info(another_video_info)
        
        # 1개만 조회
        unprocessed = repository.get_unprocessed_videos(limit=1)
        
        assert len(unprocessed) == 1, "제한된 개수의 미처리 영상이 조회되지 않았습니다"
    
    def test_get_unprocessed_videos_priority_order(self, initialized_repository, sample_video_info, another_video_info):
        """미처리 영상 우선순위 조회 테스트"""
        repository = initialized_repository
        
        # 두 영상 저장 (another_video_info가 더 최신)
        repository.save_video_info(sample_video_info)
        repository.save_video_info(another_video_info)
        
        # 최신순으로 미처리 영상 조회
        unprocessed = repository.get_unprocessed_videos(order_by='latest')
        
        assert len(unprocessed) == 2
        assert unprocessed[0].video_id == another_video_info.video_id  # 더 최신이 먼저


class TestVideoRepositoryIntegration:
    """영상 저장소 통합 테스트"""
    
    def test_complete_video_processing_workflow(self, initialized_repository, sample_video_info):
        """완전한 영상 처리 워크플로우 테스트"""
        repository = initialized_repository
        
        # 1. 영상 정보 저장
        save_result = repository.save_video_info(sample_video_info)
        assert save_result is True
        
        # 2. 미처리 영상으로 확인
        unprocessed = repository.get_unprocessed_videos()
        assert len(unprocessed) == 1
        assert unprocessed[0].video_id == sample_video_info.video_id
        
        # 3. 처리 중으로 상태 변경
        update_result = repository.update_processing_status(
            sample_video_info.video_id, 'processing'
        )
        assert update_result is True
        
        # 4. 처리 완료 마킹
        complete_result = repository.mark_video_as_processed(sample_video_info.video_id)
        assert complete_result is True
        
        # 5. 최종 상태 확인
        final_video = repository.get_video_by_id(sample_video_info.video_id)
        assert final_video.get_metadata('is_processed') is True
        assert final_video.get_metadata('processing_status') == 'completed'
        
        # 6. 미처리 목록에서 제외 확인
        final_unprocessed = repository.get_unprocessed_videos()
        assert len(final_unprocessed) == 0


class TestVideoRepositoryErrors:
    """영상 저장소 오류 처리 테스트"""
    
    def test_repository_initialization_with_invalid_db(self):
        """잘못된 데이터베이스로 저장소 초기화 테스트"""
        from src.database.repository import VideoRepository
        
        with pytest.raises(DatabaseError):
            VideoRepository("invalid://database/url")
    
    def test_save_video_with_invalid_data(self, initialized_repository):
        """잘못된 데이터로 영상 저장 테스트"""
        repository = initialized_repository
        
        # VideoInfo 모델 검증을 통과하지 못하는 데이터는 이미 생성 시 실패
        # 대신 필수 참조 무결성 테스트 (이미 test_save_video_with_invalid_channel에서 수행)
        pass
    
    def test_database_connection_failure_handling(self, temp_db_path):
        """데이터베이스 연결 실패 처리 테스트"""
        from src.database.repository import VideoRepository
        
        # 권한 없는 경로
        invalid_path = "/dev/null/readonly.db"
        
        with pytest.raises(DatabaseError):
            repository = VideoRepository(f"sqlite:///{invalid_path}")
            repository.initialize_tables() 