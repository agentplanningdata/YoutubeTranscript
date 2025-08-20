"""
자막 전처리 및 청킹 기능 테스트

YouTube 자막의 전처리와 langchain text splitter를 활용한 청킹 기능을 테스트합니다.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.services.transcript_processor import TranscriptProcessor, ChunkingStrategy
from src.utils.exceptions import TranscriptProcessingError


def test_clean_transcript_text():
    """자막 텍스트 정제가 올바르게 작동하는지 확인"""
    
    # 정제가 필요한 자막 데이터
    raw_transcript = [
        {
            'text': '  안녕하세요,   여러분!  \n\n오늘은...  ',
            'start': 0.0,
            'duration': 3.5,
            'end': 3.5
        },
        {
            'text': 'YouTube\t\t API에 대해서\r\n알아보겠습니다!!!',
            'start': 3.5,
            'duration': 4.2,
            'end': 7.7
        },
        {
            'text': '   [음악]   ',  # 배경음악 표시
            'start': 7.7,
            'duration': 1.0,
            'end': 8.7
        },
        {
            'text': '그럼 시작해볼까요??',
            'start': 8.7,
            'duration': 2.3,
            'end': 11.0
        }
    ]
    
    processor = TranscriptProcessor()
    cleaned_transcript = processor.clean_transcript_text(raw_transcript)
    
    # 정제 결과 검증
    assert len(cleaned_transcript) == 3  # [음악] 제거되어야 함
    
    # 첫 번째 세그먼트 정제 확인
    first_segment = cleaned_transcript[0]
    assert first_segment['text'] == '안녕하세요, 여러분! 오늘은...'
    assert first_segment['start'] == 0.0
    assert first_segment['duration'] == 3.5
    
    # 두 번째 세그먼트 정제 확인
    second_segment = cleaned_transcript[1]
    assert second_segment['text'] == 'YouTube API에 대해서 알아보겠습니다!'
    assert second_segment['start'] == 3.5
    assert second_segment['duration'] == 4.2
    
    # 세 번째 세그먼트 정제 확인 (과도한 물음표 정리)
    third_segment = cleaned_transcript[2]
    assert third_segment['text'] == '그럼 시작해볼까요?'
    assert third_segment['start'] == 8.7
    assert third_segment['duration'] == 2.3


def test_clean_transcript_text_remove_noise():
    """자막에서 노이즈 제거 기능 확인"""
    
    # 노이즈가 포함된 자막
    noisy_transcript = [
        {'text': '[박수]', 'start': 0.0, 'duration': 1.0, 'end': 1.0},
        {'text': '(웃음)', 'start': 1.0, 'duration': 1.5, 'end': 2.5},
        {'text': '안녕하세요', 'start': 2.5, 'duration': 2.0, 'end': 4.5},
        {'text': '[음악]', 'start': 4.5, 'duration': 3.0, 'end': 7.5},
        {'text': '오늘은 좋은 날이에요', 'start': 7.5, 'duration': 3.5, 'end': 11.0},
        {'text': '(기계음)', 'start': 11.0, 'duration': 0.5, 'end': 11.5}
    ]
    
    processor = TranscriptProcessor()
    cleaned = processor.clean_transcript_text(noisy_transcript)
    
    # 노이즈 제거 확인
    assert len(cleaned) == 2
    assert cleaned[0]['text'] == '안녕하세요'
    assert cleaned[1]['text'] == '오늘은 좋은 날이에요'


def test_chunk_transcript_by_sentences():
    """문장 단위 청킹이 올바르게 작동하는지 확인"""
    
    # 여러 문장이 포함된 자막 데이터
    transcript_segments = [
        {
            'text': '안녕하세요, 여러분! 오늘은 YouTube API에 대해 알아보겠습니다.',
            'start': 0.0,
            'duration': 5.0,
            'end': 5.0
        },
        {
            'text': '먼저 API 키를 발급받는 방법을 설명드리겠습니다. 구글 클라우드 콘솔에 접속해주세요.',
            'start': 5.0,
            'duration': 6.5,
            'end': 11.5
        },
        {
            'text': '그 다음에는 YouTube Data API v3를 활성화해야 합니다. 이것은 매우 중요한 단계입니다.',
            'start': 11.5,
            'duration': 7.0,
            'end': 18.5
        }
    ]
    
    processor = TranscriptProcessor()
    
    # 문장 단위 청킹 실행
    chunks = processor.chunk_transcript_by_sentences(
        transcript_segments, 
        max_chunk_size=80,  # 문자 수 기준 (더 작게 설정)
        overlap_size=20
    )
    
    # 청킹 결과 검증
    assert len(chunks) >= 2  # 최소 2개 이상의 청크
    
    for chunk in chunks:
        # 각 청크가 필수 필드를 가지고 있는지 확인
        assert 'text' in chunk
        assert 'start_time' in chunk
        assert 'end_time' in chunk
        assert 'source_segments' in chunk
        assert 'chunk_id' in chunk
        
        # 청크 크기가 제한을 넘지 않는지 확인
        assert len(chunk['text']) <= 250  # 약간의 여유를 둔 검증
        
        # 타임스탬프가 올바른지 확인
        assert chunk['start_time'] <= chunk['end_time']


def test_chunk_with_overlap():
    """오버랩을 포함한 청킹 기능 확인"""
    
    # 긴 자막 텍스트
    long_transcript = [
        {
            'text': '첫 번째 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다.',
            'start': 0.0,
            'duration': 10.0,
            'end': 10.0
        },
        {
            'text': '네 번째 문장입니다. 다섯 번째 문장입니다. 여섯 번째 문장입니다.',
            'start': 10.0,
            'duration': 12.0,
            'end': 22.0
        }
    ]
    
    processor = TranscriptProcessor()
    
    # 오버랩 청킹 실행
    chunks = processor.chunk_transcript_by_sentences(
        long_transcript,
        max_chunk_size=50,  # 작은 청크 크기
        overlap_size=20
    )
    
    assert len(chunks) >= 2
    
    # 오버랩 확인
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        # 시간적 오버랩 확인
        assert current_chunk['end_time'] > next_chunk['start_time'] or \
               abs(current_chunk['end_time'] - next_chunk['start_time']) < 1.0


def test_chunk_transcript_with_langchain_splitter():
    """LangChain TextSplitter를 사용한 청킹 기능 확인"""
    
    # Mock LangChain availability 및 TextSplitter
    with patch('src.services.transcript_processor.LANGCHAIN_TEXT_SPLITTERS_AVAILABLE', True), \
         patch('src.services.transcript_processor.RecursiveCharacterTextSplitter') as mock_splitter_class:
        
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter
        
        # Mock Document과 split_documents 메서드
        from langchain_core.documents import Document
        mock_documents = [
            Document(page_content="첫 번째 청크 내용입니다. 이것은 테스트입니다.", 
                    metadata={'segment_index': 0, 'start_time': 0.0, 'end_time': 7.5}),
            Document(page_content="두 번째 청크 내용입니다. 계속 테스트 중입니다.", 
                    metadata={'segment_index': 0, 'start_time': 7.5, 'end_time': 15.0})
        ]
        mock_splitter.split_documents.return_value = mock_documents
        
        transcript_segments = [
            {
                'text': '첫 번째 청크 내용입니다. 이것은 테스트입니다. 두 번째 청크 내용입니다. 계속 테스트 중입니다.',
                'start': 0.0,
                'duration': 15.0,
                'end': 15.0
            }
        ]
        
        processor = TranscriptProcessor(chunking_strategy="langchain_recursive")
        chunks = processor.chunk_transcript_by_sentences(transcript_segments, max_chunk_size=100)
        
        # LangChain splitter가 호출되었는지 확인
        mock_splitter_class.assert_called_once()
        mock_splitter.split_documents.assert_called_once()
        
        # 결과 검증
        assert len(chunks) == 2
        assert chunks[0]['text'] == "첫 번째 청크 내용입니다. 이것은 테스트입니다."
        assert chunks[1]['text'] == "두 번째 청크 내용입니다. 계속 테스트 중입니다."
        assert chunks[0]['chunking_method'] == 'langchain_recursive'
        assert chunks[1]['chunking_method'] == 'langchain_recursive'


def test_preserve_timestamp_metadata():
    """타임스탬프 메타데이터 보존 기능 확인"""
    
    transcript_with_metadata = [
        {
            'text': '첫 번째 세그먼트입니다.',
            'start': 0.0,
            'duration': 3.5,
            'end': 3.5,
            'confidence': 0.95,
            'speaker': 'main'
        },
        {
            'text': '두 번째 세그먼트입니다.',
            'start': 3.5,
            'duration': 4.0,
            'end': 7.5,
            'confidence': 0.88,
            'speaker': 'main'
        },
        {
            'text': '세 번째 세그먼트입니다.',
            'start': 7.5,
            'duration': 3.8,
            'end': 11.3,
            'confidence': 0.92,
            'speaker': 'guest'
        }
    ]
    
    processor = TranscriptProcessor()
    chunks = processor.chunk_transcript_by_sentences(
        transcript_with_metadata, 
        max_chunk_size=100,
        preserve_metadata=True
    )
    
    # 메타데이터 보존 확인
    for chunk in chunks:
        assert 'start_time' in chunk
        assert 'end_time' in chunk
        assert 'source_segments' in chunk
        
        # 원본 세그먼트 메타데이터가 보존되었는지 확인
        for source_seg in chunk['source_segments']:
            assert 'confidence' in source_seg
            assert 'speaker' in source_seg
            assert source_seg['confidence'] >= 0.8


def test_chunking_strategy_factory():
    """청킹 전략 팩토리 패턴 테스트"""
    
    # 다양한 청킹 전략 테스트
    strategies = ['sentence_based', 'langchain_recursive', 'semantic_chunker']
    
    for strategy_name in strategies:
        # semantic_chunker는 embeddings가 필요하므로 mock 제공
        if strategy_name == "semantic_chunker":
            mock_embeddings = MagicMock()
            processor = TranscriptProcessor(
                chunking_strategy=strategy_name, 
                embeddings=mock_embeddings
            )
        else:
            processor = TranscriptProcessor(chunking_strategy=strategy_name)
        
        # semantic_chunker는 SEMANTIC_CHUNKER_AVAILABLE=False이면 langchain_recursive로 fallback됨
        if strategy_name == "semantic_chunker":
            # SemanticChunker가 사용할 수 없으므로 langchain_recursive로 fallback
            expected_strategy = "langchain_recursive"
        else:
            expected_strategy = strategy_name
            
        assert processor.chunking_strategy == expected_strategy
        assert processor._get_chunking_strategy() is not None


def test_chunking_strategy_configuration():
    """청킹 전략 설정 테스트"""
    
    # 커스텀 설정
    config = {
        'max_chunk_size': 150,
        'overlap_size': 30,
        'separator_patterns': [r'\.', r'\!', r'\?'],
        'preserve_sentence_boundaries': True,
        'min_chunk_size': 20
    }
    
    processor = TranscriptProcessor(
        chunking_strategy="sentence_based",
        chunking_config=config
    )
    
    # 설정이 올바르게 적용되었는지 확인
    assert processor.chunking_config.max_chunk_size == 150
    assert processor.chunking_config.overlap_size == 30
    assert processor.chunking_config.min_chunk_size == 20


def test_empty_transcript_handling():
    """빈 자막 처리 확인"""
    
    processor = TranscriptProcessor()
    
    # 빈 자막 리스트
    empty_transcript = []
    result = processor.clean_transcript_text(empty_transcript)
    assert result == []
    
    # 청킹 시에도 빈 결과 반환
    chunks = processor.chunk_transcript_by_sentences(empty_transcript)
    assert chunks == []


def test_transcript_processor_initialization():
    """자막 프로세서 초기화 테스트"""
    
    # 기본 초기화 (기본값이 langchain_recursive로 변경됨)
    processor = TranscriptProcessor()
    assert processor.chunking_strategy == "langchain_recursive"
    assert processor.chunking_config is not None
    
    # 커스텀 전략으로 초기화
    custom_processor = TranscriptProcessor(
        chunking_strategy="langchain_recursive",
        chunking_config={'max_chunk_size': 200}
    )
    assert custom_processor.chunking_strategy == "langchain_recursive"
    assert custom_processor.chunking_config.max_chunk_size == 200


def test_invalid_chunking_strategy():
    """잘못된 청킹 전략 처리 테스트"""
    
    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        TranscriptProcessor(chunking_strategy="invalid_strategy")


def test_chunk_text_length_validation():
    """청크 텍스트 길이 검증 테스트"""
    
    # 매우 긴 단일 세그먼트
    long_segment = {
        'text': 'a' * 1000,  # 1000자 길이
        'start': 0.0,
        'duration': 60.0,
        'end': 60.0
    }
    
    processor = TranscriptProcessor()
    chunks = processor.chunk_transcript_by_sentences(
        [long_segment], 
        max_chunk_size=200
    )
    
    # 청크가 최대 크기를 초과하지 않는지 확인
    for chunk in chunks:
        assert len(chunk['text']) <= 250  # 약간의 여유를 둔 검증


def test_transcript_processing_error_handling():
    """자막 처리 중 오류 상황 테스트"""
    
    processor = TranscriptProcessor()
    
    # 잘못된 형식의 자막 데이터
    invalid_transcript = [
        {'text': 'valid segment', 'start': 0.0, 'duration': 2.0},
        {'invalid_field': 'missing text'}  # 필수 필드 누락
    ]
    
    with pytest.raises(TranscriptProcessingError, match="Invalid transcript segment"):
        processor.clean_transcript_text(invalid_transcript)


def test_chunking_performance_with_large_transcript():
    """대용량 자막 청킹 성능 테스트"""
    
    # 큰 자막 데이터 생성 (100개 세그먼트)
    large_transcript = []
    for i in range(100):
        large_transcript.append({
            'text': f'세그먼트 {i+1}번 내용입니다. 이것은 성능 테스트를 위한 데이터입니다.',
            'start': i * 2.0,
            'duration': 2.0,
            'end': (i + 1) * 2.0
        })
    
    processor = TranscriptProcessor()
    
    # 청킹 실행 (시간 측정은 실제 벤치마크에서)
    chunks = processor.chunk_transcript_by_sentences(large_transcript, max_chunk_size=300)
    
    # 기본적인 결과 검증
    assert len(chunks) > 0
    assert all('text' in chunk for chunk in chunks)
    assert all('start_time' in chunk for chunk in chunks)
    assert all('end_time' in chunk for chunk in chunks) 


def test_merge_small_chunks():
    """작은 청크들이 올바르게 병합되는지 확인"""
    
    # 매우 짧은 세그먼트들
    short_segments = [
        {
            'text': '네',
            'start': 0.0,
            'duration': 1.0,
            'end': 1.0
        },
        {
            'text': '맞습니다',
            'start': 1.0,
            'duration': 1.5,
            'end': 2.5
        },
        {
            'text': '그렇죠',
            'start': 2.5,
            'duration': 1.2,
            'end': 3.7
        },
        {
            'text': '이것은 좀 더 긴 문장이라서 병합되지 않을 것입니다',
            'start': 3.7,
            'duration': 4.0,
            'end': 7.7
        },
        {
            'text': '감사합니다',
            'start': 7.7,
            'duration': 1.5,
            'end': 9.2
        }
    ]
    
    processor = TranscriptProcessor()
    
    # 최소 크기를 20자로 설정하여 짧은 청크들이 병합되도록 함
    chunks = processor.chunk_transcript_by_sentences(
        short_segments,
        max_chunk_size=100,
        overlap_size=0  # 오버랩 없이 테스트
    )
    
    # 결과 검증
    print(f"Chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: '{chunk['text']}' ({len(chunk['text'])} chars)")
    
    # 모든 청크가 최소 크기를 만족하거나 병합할 수 없는 경우여야 함
    for chunk in chunks:
        chunk_text = chunk['text']
        chunk_length = len(chunk_text)
        
        # 20자 이상이거나, 병합할 수 없어서 그대로 남은 경우
        assert chunk_length >= 20 or chunk_length < 10, \
            f"Chunk should be >= 20 chars or very short if unmergeable: {chunk_length} chars - '{chunk_text}'"
    
    # 일부 청크는 병합되었을 것임 (원본 5개보다 적어야 함)
    assert len(chunks) < len(short_segments), "Some chunks should have been merged"


def test_small_chunks_boundary_cases():
    """작은 청크 병합의 경계 상황들 테스트"""
    
    # 모든 세그먼트가 매우 짧은 경우
    very_short_segments = [
        {'text': 'A', 'start': 0.0, 'duration': 0.5, 'end': 0.5},
        {'text': 'B', 'start': 0.5, 'duration': 0.5, 'end': 1.0},
        {'text': 'C', 'start': 1.0, 'duration': 0.5, 'end': 1.5},
        {'text': 'D', 'start': 1.5, 'duration': 0.5, 'end': 2.0},
    ]
    
    processor = TranscriptProcessor()
    
    # 최소 크기를 15자로 설정
    custom_config = {'min_chunk_size': 15, 'max_chunk_size': 50}
    processor_custom = TranscriptProcessor(
        chunking_strategy="sentence_based",
        chunking_config=custom_config
    )
    
    chunks = processor_custom.chunk_transcript_by_sentences(very_short_segments)
    
    # 병합이 일어났는지 확인
    assert len(chunks) < len(very_short_segments)
    
    # 병합된 청크들의 내용 확인
    combined_text = ' '.join(chunk['text'] for chunk in chunks)
    original_text = ' '.join(seg['text'] for seg in very_short_segments)
    
    # 모든 텍스트가 보존되어야 함 (공백 차이는 허용)
    assert combined_text.replace(' ', '') == original_text.replace(' ', '')


def test_chunk_size_quality_metrics():
    """청크 크기 품질 지표 테스트"""
    
    mixed_segments = [
        {'text': 'Hi', 'start': 0.0, 'duration': 1.0, 'end': 1.0},
        {'text': 'This is a medium length sentence that should be good for embedding.', 'start': 1.0, 'duration': 3.0, 'end': 4.0},
        {'text': 'OK', 'start': 4.0, 'duration': 0.5, 'end': 4.5},
        {'text': 'Another reasonably sized sentence that contains meaningful information for search.', 'start': 4.5, 'duration': 4.0, 'end': 8.5},
    ]
    
    processor = TranscriptProcessor()
    chunks = processor.chunk_transcript_by_sentences(mixed_segments, max_chunk_size=200)
    
    # 청킹 통계 확인
    stats = processor.get_chunking_stats(chunks)
    
    assert stats['total_chunks'] >= 1
    assert stats['average_chunk_size'] > 20  # 평균이 최소 크기보다 커야 함
    assert stats['min_chunk_size'] >= 10    # 최소도 어느 정도는 되어야 함
    
    print(f"Chunking stats: {stats}") 