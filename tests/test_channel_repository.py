"""
채널 관리 저장소 테스트

SQLite + LangChain 기반 채널 CRUD 연산을 테스트합니다.
와치리스트 관리와 채널 메타데이터 관리 기능을 포함합니다.
"""

import pytest
import tempfile
import os
from datetime import datetime
from typing import List, Optional

# LangChain SQL 관련 임포트
from langchain_community.utilities import SQLDatabase

# 프로젝트 모듈 임포트
from src.services.models import ChannelInfo
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
        video_count=100,
        custom_url='testchannel',
        country='KR',
        default_language='ko'
    )


@pytest.fixture
def another_channel_info():
    """추가 테스트용 채널 정보"""
    return ChannelInfo(
        channel_id='UC0987654321',
        title='또 다른 채널',
        description='두 번째 테스트 채널',
        thumbnail_url='https://example.com/thumbnail2.jpg',
        published_at='2023-02-01T00:00:00Z',
        subscriber_count=5000,
        video_count=50
    )


@pytest.fixture
def initialized_repository(temp_db_path):
    """초기화된 채널 저장소"""
    from src.database.repository import ChannelRepository
    
    db_url = f"sqlite:///{temp_db_path}"
    repository = ChannelRepository(db_url)
    
    # 테이블 초기화
    repository.initialize_tables()
    
    return repository


class TestAddChannelToWatchlist:
    """와치리스트에 채널 추가 테스트"""
    
    def test_add_channel_to_watchlist(self, initialized_repository, sample_channel_info):
        """와치리스트에 채널 추가 테스트"""
        repository = initialized_repository
        
        # 채널을 와치리스트에 추가
        result = repository.add_channel_to_watchlist(sample_channel_info)
        
        assert result is True, "채널 추가가 실패했습니다"
        
        # 추가된 채널 확인
        channels = repository.get_watchlist_channels()
        
        assert len(channels) == 1, "추가된 채널 수가 올바르지 않습니다"
        assert channels[0].channel_id == sample_channel_info.channel_id
        assert channels[0].title == sample_channel_info.title
    
    def test_add_duplicate_channel_to_watchlist(self, initialized_repository, sample_channel_info):
        """중복 채널 추가 방지 테스트"""
        repository = initialized_repository
        
        # 첫 번째 추가
        result1 = repository.add_channel_to_watchlist(sample_channel_info)
        assert result1 is True
        
        # 중복 추가 시도
        result2 = repository.add_channel_to_watchlist(sample_channel_info)
        assert result2 is False, "중복 채널이 추가되었습니다"
        
        # 채널 수가 1개인지 확인
        channels = repository.get_watchlist_channels()
        assert len(channels) == 1
    
    def test_add_channel_with_missing_required_fields(self, initialized_repository):
        """필수 필드가 없는 채널 추가 테스트"""
        repository = initialized_repository
        
        # ChannelInfo 모델 자체에서 이미 검증하므로, ValueError를 예상
        with pytest.raises(ValueError):
            invalid_channel = ChannelInfo(
                channel_id='UC1234567890',
                title='',  # 빈 제목
                description='설명',
                thumbnail_url='https://example.com/thumb.jpg',
                published_at='2023-01-01T00:00:00Z'
            )


class TestGetWatchlistChannels:
    """와치리스트 채널 조회 테스트"""
    
    def test_get_watchlist_channels(self, initialized_repository, sample_channel_info, another_channel_info):
        """와치리스트 채널 조회 테스트"""
        repository = initialized_repository
        
        # 두 개의 채널 추가
        repository.add_channel_to_watchlist(sample_channel_info)
        repository.add_channel_to_watchlist(another_channel_info)
        
        # 채널 조회
        channels = repository.get_watchlist_channels()
        
        assert len(channels) == 2, "조회된 채널 수가 올바르지 않습니다"
        
        # 채널 ID 확인
        channel_ids = [ch.channel_id for ch in channels]
        assert sample_channel_info.channel_id in channel_ids
        assert another_channel_info.channel_id in channel_ids
    
    def test_get_watchlist_channels_empty(self, initialized_repository):
        """빈 와치리스트 조회 테스트"""
        repository = initialized_repository
        
        channels = repository.get_watchlist_channels()
        
        assert channels == [], "빈 와치리스트가 올바르게 반환되지 않았습니다"
    
    def test_get_watchlist_channels_with_filter(self, initialized_repository, sample_channel_info, another_channel_info):
        """필터링된 와치리스트 조회 테스트"""
        repository = initialized_repository
        
        # 두 채널 추가
        repository.add_channel_to_watchlist(sample_channel_info)
        repository.add_channel_to_watchlist(another_channel_info)
        
        # 활성 채널만 조회
        active_channels = repository.get_watchlist_channels(is_active=True)
        assert len(active_channels) == 2
        
        # 특정 채널 ID로 조회
        specific_channels = repository.get_watchlist_channels(
            channel_ids=[sample_channel_info.channel_id]
        )
        assert len(specific_channels) == 1
        assert specific_channels[0].channel_id == sample_channel_info.channel_id


class TestRemoveChannelFromWatchlist:
    """와치리스트에서 채널 제거 테스트"""
    
    def test_remove_channel_from_watchlist(self, initialized_repository, sample_channel_info):
        """와치리스트에서 채널 제거 테스트"""
        repository = initialized_repository
        
        # 채널 추가
        repository.add_channel_to_watchlist(sample_channel_info)
        
        # 채널 제거
        result = repository.remove_channel_from_watchlist(sample_channel_info.channel_id)
        
        assert result is True, "채널 제거가 실패했습니다"
        
        # 제거 확인
        channels = repository.get_watchlist_channels()
        assert len(channels) == 0, "채널이 제거되지 않았습니다"
    
    def test_remove_nonexistent_channel(self, initialized_repository):
        """존재하지 않는 채널 제거 테스트"""
        repository = initialized_repository
        
        # 존재하지 않는 채널 제거 시도
        result = repository.remove_channel_from_watchlist('UC_NONEXISTENT')
        
        assert result is False, "존재하지 않는 채널 제거가 성공했습니다"
    
    def test_soft_delete_channel(self, initialized_repository, sample_channel_info):
        """채널 소프트 삭제 테스트"""
        repository = initialized_repository
        
        # 채널 추가
        repository.add_channel_to_watchlist(sample_channel_info)
        
        # 소프트 삭제 (is_active = False로 설정)
        result = repository.deactivate_channel(sample_channel_info.channel_id)
        
        assert result is True, "채널 비활성화가 실패했습니다"
        
        # 활성 채널만 조회 시 나타나지 않아야 함
        active_channels = repository.get_watchlist_channels(is_active=True)
        assert len(active_channels) == 0
        
        # 모든 채널 조회 시는 나타나야 함
        all_channels = repository.get_watchlist_channels(is_active=None)
        assert len(all_channels) == 1


class TestUpdateChannelMetadata:
    """채널 메타데이터 업데이트 테스트"""
    
    def test_update_channel_metadata(self, initialized_repository, sample_channel_info):
        """채널 메타데이터 업데이트 테스트"""
        repository = initialized_repository
        
        # 채널 추가
        repository.add_channel_to_watchlist(sample_channel_info)
        
        # 메타데이터 업데이트
        updated_info = ChannelInfo(
            channel_id=sample_channel_info.channel_id,
            title='업데이트된 제목',
            description='업데이트된 설명',
            thumbnail_url=sample_channel_info.thumbnail_url,
            published_at=sample_channel_info.published_at,
            subscriber_count=15000,  # 증가
            video_count=120  # 증가
        )
        
        result = repository.update_channel_metadata(updated_info)
        
        assert result is True, "채널 메타데이터 업데이트가 실패했습니다"
        
        # 업데이트 확인
        channels = repository.get_watchlist_channels()
        updated_channel = channels[0]
        
        assert updated_channel.title == '업데이트된 제목'
        assert updated_channel.subscriber_count == 15000
        assert updated_channel.video_count == 120
    
    def test_update_nonexistent_channel_metadata(self, initialized_repository, sample_channel_info):
        """존재하지 않는 채널 메타데이터 업데이트 테스트"""
        repository = initialized_repository
        
        # 존재하지 않는 채널 업데이트 시도
        result = repository.update_channel_metadata(sample_channel_info)
        
        assert result is False, "존재하지 않는 채널 업데이트가 성공했습니다"
    
    def test_partial_metadata_update(self, initialized_repository, sample_channel_info):
        """부분 메타데이터 업데이트 테스트"""
        repository = initialized_repository
        
        # 채널 추가
        repository.add_channel_to_watchlist(sample_channel_info)
        
        # 구독자 수만 업데이트
        result = repository.update_channel_field(
            sample_channel_info.channel_id,
            field='subscriber_count',
            value=20000
        )
        
        assert result is True, "부분 업데이트가 실패했습니다"
        
        # 확인
        channels = repository.get_watchlist_channels()
        updated_channel = channels[0]
        
        assert updated_channel.subscriber_count == 20000
        assert updated_channel.title == sample_channel_info.title  # 기존 값 유지


class TestChannelRepositoryIntegration:
    """채널 저장소 통합 테스트"""
    
    def test_complete_channel_lifecycle(self, initialized_repository, sample_channel_info):
        """완전한 채널 생명주기 테스트"""
        repository = initialized_repository
        
        # 1. 채널 추가
        add_result = repository.add_channel_to_watchlist(sample_channel_info)
        assert add_result is True
        
        # 2. 채널 조회
        channels = repository.get_watchlist_channels()
        assert len(channels) == 1
        
        # 3. 메타데이터 업데이트
        sample_channel_info.subscriber_count = 25000
        update_result = repository.update_channel_metadata(sample_channel_info)
        assert update_result is True
        
        # 4. 업데이트 확인
        updated_channels = repository.get_watchlist_channels()
        assert updated_channels[0].subscriber_count == 25000
        
        # 5. 채널 비활성화
        deactivate_result = repository.deactivate_channel(sample_channel_info.channel_id)
        assert deactivate_result is True
        
        # 6. 활성 채널 조회 (비어있어야 함)
        active_channels = repository.get_watchlist_channels(is_active=True)
        assert len(active_channels) == 0
        
        # 7. 채널 제거
        remove_result = repository.remove_channel_from_watchlist(sample_channel_info.channel_id)
        assert remove_result is True
        
        # 8. 최종 확인
        final_channels = repository.get_watchlist_channels(is_active=None)
        assert len(final_channels) == 0


class TestChannelRepositoryErrors:
    """채널 저장소 오류 처리 테스트"""
    
    def test_repository_initialization_with_invalid_db(self):
        """잘못된 데이터베이스로 저장소 초기화 테스트"""
        from src.database.repository import ChannelRepository
        
        with pytest.raises(DatabaseError):
            ChannelRepository("invalid://database/url")
    
    def test_database_connection_failure_handling(self, temp_db_path):
        """데이터베이스 연결 실패 처리 테스트"""
        from src.database.repository import ChannelRepository
        
        # 권한 없는 경로
        invalid_path = "/dev/null/readonly.db"
        
        with pytest.raises(DatabaseError):
            repository = ChannelRepository(f"sqlite:///{invalid_path}")
            repository.initialize_tables() 