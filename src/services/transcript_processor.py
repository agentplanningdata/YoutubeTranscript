"""
자막 전처리 및 청킹 서비스 - LangChain 기반

YouTube 자막의 전처리와 최신 LangChain text splitter를 활용한 청킹 기능을 제공합니다.
"""

from typing import Optional, List, Dict, Any, Protocol
import re
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field

# LangChain Core Imports
from langchain_core.documents import Document

# LangChain Text Splitters (최신 패키지)
try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
        TokenTextSplitter,
        SpacyTextSplitter,
        SentenceTransformersTextSplitter,
        HTMLHeaderTextSplitter,
        MarkdownHeaderTextSplitter
    )
    
    # Semantic Chunker 시도 (실험적 기능)
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        SEMANTIC_CHUNKER_AVAILABLE = True
    except ImportError:
        SEMANTIC_CHUNKER_AVAILABLE = False
    
    LANGCHAIN_TEXT_SPLITTERS_AVAILABLE = True
except ImportError:
    # 테스트 환경에서는 Mock으로 대체
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        def split_text(self, text: str) -> List[str]:
            raise ImportError("langchain-text-splitters not available")
        
        def create_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[Document]:
            raise ImportError("langchain-text-splitters not available")
    
    class CharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        def split_text(self, text: str) -> List[str]:
            raise ImportError("langchain-text-splitters not available")
    
    class TokenTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        def split_text(self, text: str) -> List[str]:
            raise ImportError("langchain-text-splitters not available")
    
    class SpacyTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
    
    class SentenceTransformersTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
    
    class SemanticChunker:
        def __init__(self, embeddings=None, **kwargs):
            self.embeddings = embeddings
    
    LANGCHAIN_TEXT_SPLITTERS_AVAILABLE = False
    SEMANTIC_CHUNKER_AVAILABLE = False

from src.utils.exceptions import TranscriptProcessingError
from src.utils.logger import get_project_logger

logger = get_project_logger(__name__)


class ChunkingStrategy(Enum):
    """청킹 전략 열거형 - LangChain 기반"""
    SENTENCE_BASED = "sentence_based"
    LANGCHAIN_RECURSIVE = "langchain_recursive"
    LANGCHAIN_CHARACTER = "langchain_character" 
    LANGCHAIN_TOKEN = "langchain_token"
    LANGCHAIN_SPACY = "langchain_spacy"
    LANGCHAIN_SENTENCE_TRANSFORMER = "langchain_sentence_transformer"
    SEMANTIC_CHUNKER = "semantic_chunker"


@dataclass
class ChunkingConfig:
    """LangChain 기반 청킹 설정"""
    # 기본 청킹 설정
    max_chunk_size: int = 400
    overlap_size: int = 50
    min_chunk_size: int = 200
    separator_patterns: List[str] = field(default_factory=lambda: [r'\.', r'\!', r'\?'])
    preserve_sentence_boundaries: bool = True
    preserve_metadata: bool = True
    
    # LangChain TextSplitter 설정
    langchain_chunk_size: int = 400  # 사용자가 설정한 크기 반영
    langchain_chunk_overlap: int = 50  # 사용자가 설정한 크기 반영
    langchain_separators: List[str] = field(default_factory=lambda: [
        "\n\n", "\n", "。", ".", "!", "?", " ", ""  # 한국어 지원 강화
    ])
    
    # 토큰 기반 설정
    encoding_name: str = "cl100k_base"  # OpenAI GPT 모델용
    
    # SpaCy 설정
    spacy_pipeline: str = "ko_core_news_sm"  # 한국어 모델
    
    # Semantic Chunker 설정
    semantic_breakpoint_threshold_type: str = "percentile"
    semantic_breakpoint_threshold_amount: float = 95.0


class ChunkingStrategyProtocol(Protocol):
    """청킹 전략 프로토콜"""
    
    @abstractmethod
    def chunk_segments(
        self, 
        segments: List[Dict[str, Any]], 
        config: ChunkingConfig
    ) -> List[Dict[str, Any]]:
        """세그먼트를 청킹합니다."""
        pass


class LangChainDocumentChunker:
    """최신 LangChain 기반 청킹 전략 (Document 활용)"""
    
    def __init__(self, splitter_type: str = "recursive"):
        self.splitter_type = splitter_type
        logger.info(f"LangChain document chunker initialized with type: {splitter_type}")
    
    def chunk_segments(
        self, 
        segments: List[Dict[str, Any]], 
        config: ChunkingConfig
    ) -> List[Dict[str, Any]]:
        """LangChain Document를 활용하여 청킹합니다."""
        if not LANGCHAIN_TEXT_SPLITTERS_AVAILABLE:
            logger.warning("LangChain text splitters not available, falling back to sentence-based chunking")
            return SentenceBasedChunker().chunk_segments(segments, config)
        
        try:
            # LangChain Document 생성
            documents = self._create_documents_from_segments(segments)
            
            # LangChain splitter 생성
            splitter = self._create_splitter(config)
            
            # Document 분할
            split_documents = splitter.split_documents(documents)
            
            # LangChain Document를 우리 청크 형식으로 변환
            chunks = self._convert_documents_to_chunks(split_documents, segments)
            
            logger.info(f"LangChain chunking: {len(segments)} segments -> {len(split_documents)} documents -> {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in LangChain chunking: {e}")
            logger.warning("Falling back to sentence-based chunking")
            return SentenceBasedChunker().chunk_segments(segments, config)
    
    def _create_documents_from_segments(self, segments: List[Dict[str, Any]]) -> List[Document]:
        """세그먼트들로부터 LangChain Document를 생성합니다."""
        documents = []
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            if not text:
                continue
            
            # 메타데이터 생성
            metadata = {
                'segment_index': i,
                'start_time': segment.get('start', 0.0),
                'end_time': segment.get('end', segment.get('start', 0.0) + segment.get('duration', 0.0)),
                'duration': segment.get('duration', 0.0),
                'original_segment': segment
            }
            
            document = Document(page_content=text, metadata=metadata)
            documents.append(document)
        
        return documents
    
    def _create_splitter(self, config: ChunkingConfig):
        """설정에 따라 LangChain splitter를 생성합니다."""
        base_kwargs = {
            'chunk_size': config.langchain_chunk_size,
            'chunk_overlap': config.langchain_chunk_overlap,
            'length_function': len,
            'is_separator_regex': False
        }
        
        if self.splitter_type == "recursive":
            base_kwargs['separators'] = config.langchain_separators
            return RecursiveCharacterTextSplitter(**base_kwargs)
            
        elif self.splitter_type == "character":
            base_kwargs['separator'] = "\n\n"
            return CharacterTextSplitter(**base_kwargs)
            
        elif self.splitter_type == "token":
            token_kwargs = {
                'chunk_size': config.langchain_chunk_size,
                'chunk_overlap': config.langchain_chunk_overlap,
                'encoding_name': config.encoding_name
            }
            return TokenTextSplitter(**token_kwargs)
            
        elif self.splitter_type == "spacy":
            spacy_kwargs = {
                'chunk_size': config.langchain_chunk_size,
                'chunk_overlap': config.langchain_chunk_overlap,
                'pipeline': config.spacy_pipeline
            }
            return SpacyTextSplitter(**spacy_kwargs)
            
        elif self.splitter_type == "sentence_transformer":
            st_kwargs = {
                'chunk_size': config.langchain_chunk_size,
                'chunk_overlap': config.langchain_chunk_overlap
            }
            return SentenceTransformersTextSplitter(**st_kwargs)
            
        else:
            raise ValueError(f"Unknown splitter type: {self.splitter_type}")
    
    def _convert_documents_to_chunks(
        self, 
        documents: List[Document], 
        original_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """LangChain Document들을 우리 청크 형식으로 변환합니다."""
        chunks = []
        
        for doc in documents:
            text = doc.page_content.strip()
            if not text:
                continue
            
            metadata = doc.metadata
            
            # 시간 정보 추출 또는 추정
            start_time = metadata.get('start_time', 0.0)
            end_time = metadata.get('end_time', start_time)
            
            # 원본 세그먼트 정보가 있으면 활용
            original_segment = metadata.get('original_segment')
            if original_segment:
                source_segments = [original_segment]
            else:
                # 텍스트 매칭을 통해 원본 세그먼트 찾기
                source_segments = self._find_matching_segments(text, original_segments)
            
            # 청크 생성
            chunk = {
                'text': text,
                'start_time': start_time,
                'end_time': end_time,
                'source_segments': source_segments,
                'chunk_id': str(uuid.uuid4()),
                'word_count': len(text.split()),
                'char_count': len(text),
                'chunking_method': f'langchain_{self.splitter_type}',
                'langchain_metadata': metadata
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def _find_matching_segments(
        self, 
        chunk_text: str, 
        original_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """청크 텍스트와 일치하는 원본 세그먼트들을 찾습니다."""
        matching_segments = []
        
        for segment in original_segments:
            segment_text = segment.get('text', '').strip()
            if segment_text and segment_text in chunk_text:
                matching_segments.append(segment)
        
        return matching_segments


class SemanticChunkerWrapper:
    """LangChain SemanticChunker 래퍼"""
    
    def __init__(self, embeddings=None):
        self.embeddings = embeddings
        logger.info("Semantic chunker wrapper initialized")
    
    def chunk_segments(
        self, 
        segments: List[Dict[str, Any]], 
        config: ChunkingConfig
    ) -> List[Dict[str, Any]]:
        """의미적 유사성 기반으로 청킹합니다."""
        if not SEMANTIC_CHUNKER_AVAILABLE:
            logger.warning("SemanticChunker not available, falling back to recursive chunking")
            return LangChainDocumentChunker("recursive").chunk_segments(segments, config)
        
        if not self.embeddings:
            logger.warning("No embeddings provided for semantic chunking, falling back to recursive chunking")
            return LangChainDocumentChunker("recursive").chunk_segments(segments, config)
        
        try:
            # 전체 텍스트 결합
            full_text = " ".join(segment.get('text', '') for segment in segments)
            
            # SemanticChunker 생성
            text_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type=config.semantic_breakpoint_threshold_type,
                breakpoint_threshold_amount=config.semantic_breakpoint_threshold_amount
            )
            
            # 텍스트 분할
            chunks_text = text_splitter.split_text(full_text)
            
            # 시간 정보 매핑하여 청크 생성
            chunks = self._map_chunks_to_segments(chunks_text, segments)
            
            logger.info(f"Semantic chunking: {len(segments)} segments -> {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            logger.warning("Falling back to recursive chunking")
            return LangChainDocumentChunker("recursive").chunk_segments(segments, config)
    
    def _map_chunks_to_segments(
        self, 
        chunks_text: List[str], 
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """의미적 청크에 시간 정보를 매핑합니다."""
        chunks = []
        full_text = " ".join(segment.get('text', '') for segment in segments)
        
        # 문자 위치와 시간 매핑 생성
        char_to_time_map = self._create_char_time_mapping(segments)
        
        current_pos = 0
        for chunk_text in chunks_text:
            if not chunk_text.strip():
                continue
            
            # 청크 텍스트의 위치 찾기
            chunk_start_pos = full_text.find(chunk_text.strip(), current_pos)
            if chunk_start_pos == -1:
                chunk_start_pos = current_pos
            
            chunk_end_pos = chunk_start_pos + len(chunk_text.strip())
            
            # 시간 정보 매핑
            start_time = self._get_time_from_position(chunk_start_pos, char_to_time_map)
            end_time = self._get_time_from_position(chunk_end_pos, char_to_time_map)
            
            # 해당 세그먼트들 찾기
            source_segments = self._find_segments_in_time_range(start_time, end_time, segments)
            
            chunk = {
                'text': chunk_text.strip(),
                'start_time': start_time,
                'end_time': end_time,
                'source_segments': source_segments,
                'chunk_id': str(uuid.uuid4()),
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text),
                'chunking_method': 'semantic_chunker'
            }
            
            chunks.append(chunk)
            current_pos = chunk_end_pos
        
        return chunks
    
    def _create_char_time_mapping(self, segments: List[Dict[str, Any]]) -> Dict[int, float]:
        """문자 위치와 시간의 매핑을 생성합니다."""
        char_to_time = {}
        current_pos = 0
        
        for segment in segments:
            text = segment.get('text', '')
            start_time = segment.get('start', 0.0)
            end_time = segment.get('end', start_time + segment.get('duration', 0.0))
            
            # 각 문자에 시간 정보 매핑
            for i, char in enumerate(text):
                if char.strip():
                    progress = i / len(text) if len(text) > 0 else 0
                    time_at_char = start_time + (end_time - start_time) * progress
                    char_to_time[current_pos + i] = time_at_char
            
            current_pos += len(text) + 1  # 공백 포함
        
        return char_to_time
    
    def _get_time_from_position(self, pos: int, char_to_time_map: Dict[int, float]) -> float:
        """문자 위치에서 시간을 추정합니다."""
        if pos in char_to_time_map:
            return char_to_time_map[pos]
        
        # 가장 가까운 위치의 시간 찾기
        if not char_to_time_map:
            return 0.0
        
        closest_pos = min(char_to_time_map.keys(), key=lambda x: abs(x - pos))
        return char_to_time_map[closest_pos]
    
    def _find_segments_in_time_range(
        self, 
        start_time: float, 
        end_time: float, 
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """시간 범위에 해당하는 세그먼트들을 찾습니다."""
        matching_segments = []
        
        for segment in segments:
            seg_start = segment.get('start', 0.0)
            seg_end = segment.get('end', seg_start + segment.get('duration', 0.0))
            
            # 시간 범위 겹침 확인
            if seg_end >= start_time and seg_start <= end_time:
                matching_segments.append(segment)
        
        return matching_segments


class SentenceBasedChunker:
    """문장 기반 청킹 전략"""
    
    def chunk_segments(
        self, 
        segments: List[Dict[str, Any]], 
        config: ChunkingConfig
    ) -> List[Dict[str, Any]]:
        """문장 기반으로 세그먼트를 청킹합니다."""
        chunks = []
        current_chunk_text = ""
        current_start_time = None
        current_end_time = None
        current_source_segments = []
        
        for segment in segments:
            text = segment.get('text', '')
            start_time = segment.get('start', 0.0)
            end_time = segment.get('end', start_time + segment.get('duration', 0.0))
            
            # 첫 번째 세그먼트인 경우
            if current_start_time is None:
                current_start_time = start_time
            
            # 현재 세그먼트의 텍스트가 너무 긴 경우 강제로 분할
            if len(text) > config.max_chunk_size:
                # 긴 텍스트를 여러 청크로 분할
                text_chunks = self._force_split_long_text(text, config.max_chunk_size)
                
                for i, text_chunk in enumerate(text_chunks):
                    if current_chunk_text:
                        # 현재 청크 완성
                        chunks.append(self._create_chunk(
                            current_chunk_text,
                            current_start_time,
                            current_end_time,
                            current_source_segments
                        ))
                        current_chunk_text = ""
                        current_source_segments = []
                    
                    # 새로운 청크 생성
                    chunk_start = start_time + (end_time - start_time) * i / len(text_chunks)
                    chunk_end = start_time + (end_time - start_time) * (i + 1) / len(text_chunks)
                    
                    chunk_segment = segment.copy()
                    chunk_segment['text'] = text_chunk
                    chunk_segment['start'] = chunk_start
                    chunk_segment['end'] = chunk_end
                    
                    chunks.append(self._create_chunk(
                        text_chunk,
                        chunk_start,
                        chunk_end,
                        [chunk_segment]
                    ))
                
                continue
            
            # 청크에 추가할지 확인
            potential_text = current_chunk_text + (" " if current_chunk_text else "") + text
            
            if len(potential_text) <= config.max_chunk_size or not current_chunk_text:
                # 현재 청크에 추가
                current_chunk_text = potential_text
                current_end_time = end_time
                current_source_segments.append(segment)
            else:
                # 현재 청크 완성 후 새 청크 시작
                if current_chunk_text:
                    chunks.append(self._create_chunk(
                        current_chunk_text,
                        current_start_time,
                        current_end_time,
                        current_source_segments
                    ))
                
                # 오버랩 처리
                overlap_text = self._get_overlap_text(current_chunk_text, config.overlap_size)
                current_chunk_text = overlap_text + (" " if overlap_text else "") + text
                current_start_time = start_time
                current_end_time = end_time
                current_source_segments = [segment]
        
        # 마지막 청크 추가
        if current_chunk_text:
            chunks.append(self._create_chunk(
                current_chunk_text,
                current_start_time,
                current_end_time,
                current_source_segments
            ))
        
        # 최소 크기 미달 청크들 후처리
        chunks = self._merge_small_chunks(chunks, config)
        
        return chunks
    
    def _create_chunk(
        self, 
        text: str, 
        start_time: float, 
        end_time: float, 
        source_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """청크 객체를 생성합니다."""
        return {
            'text': text.strip(),
            'start_time': start_time,
            'end_time': end_time,
            'source_segments': source_segments.copy(),
            'chunk_id': str(uuid.uuid4()),
            'word_count': len(text.split()),
            'char_count': len(text),
            'chunking_method': 'sentence_based'
        }
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """오버랩 텍스트를 추출합니다."""
        if not text or overlap_size <= 0:
            return ""
        
        words = text.split()
        if len(words) <= overlap_size:
            return text
        
        return " ".join(words[-overlap_size:])
    
    def _force_split_long_text(self, text: str, max_size: int) -> List[str]:
        """긴 텍스트를 강제로 분할합니다."""
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = ""
        
        for word in words:
            # 단어 하나가 max_size를 초과하는 경우
            if len(word) > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 단어를 문자 단위로 분할
                for i in range(0, len(word), max_size):
                    chunks.append(word[i:i + max_size])
                continue
            
            # 현재 청크에 단어를 추가할 수 있는지 확인
            potential_chunk = current_chunk + (" " if current_chunk else "") + word
            
            if len(potential_chunk) <= max_size:
                current_chunk = potential_chunk
            else:
                # 현재 청크를 완료하고 새 청크 시작
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _merge_small_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        config: ChunkingConfig
    ) -> List[Dict[str, Any]]:
        """최소 크기에 미달하는 청크들을 인접 청크와 병합합니다."""
        if not chunks or config.min_chunk_size <= 0:
            return chunks
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            current_text = current_chunk.get('text', '')
            
            # 현재 청크가 최소 크기를 만족하는 경우
            if len(current_text) >= config.min_chunk_size:
                merged_chunks.append(current_chunk)
                i += 1
                continue
            
            # 현재 청크가 너무 작은 경우 - 다음 청크와 병합 시도
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                next_text = next_chunk.get('text', '')
                
                # 병합 후 크기 확인
                combined_text = current_text + ' ' + next_text
                if len(combined_text) <= config.max_chunk_size:
                    # 병합 청크 생성
                    merged_chunk = self._create_merged_chunk(current_chunk, next_chunk, combined_text)
                    merged_chunks.append(merged_chunk)
                    i += 2  # 두 청크를 모두 건너뛰기
                    continue
            
            # 이전 청크와 병합 시도
            if merged_chunks and len(merged_chunks) > 0:
                prev_chunk = merged_chunks[-1]
                prev_text = prev_chunk.get('text', '')
                
                combined_text = prev_text + ' ' + current_text
                if len(combined_text) <= config.max_chunk_size:
                    # 이전 청크와 병합
                    merged_chunks[-1] = self._create_merged_chunk(prev_chunk, current_chunk, combined_text)
                    i += 1
                    continue
            
            # 병합할 수 없는 경우 그대로 추가 (경고 로그)
            logger.warning(f"Small chunk detected but cannot merge: {len(current_text)} chars")
            merged_chunks.append(current_chunk)
            i += 1
        
        return merged_chunks
    
    def _create_merged_chunk(
        self, 
        chunk1: Dict[str, Any], 
        chunk2: Dict[str, Any], 
        combined_text: str
    ) -> Dict[str, Any]:
        """두 청크를 병합한 새로운 청크를 생성합니다."""
        return {
            'text': combined_text.strip(),
            'start_time': min(chunk1.get('start_time', 0), chunk2.get('start_time', 0)),
            'end_time': max(chunk1.get('end_time', 0), chunk2.get('end_time', 0)),
            'source_segments': chunk1.get('source_segments', []) + chunk2.get('source_segments', []),
            'chunk_id': str(uuid.uuid4()),
            'word_count': len(combined_text.split()),
            'char_count': len(combined_text),
            'chunking_method': 'sentence_based_merged',
            'merged_from': [chunk1.get('chunk_id'), chunk2.get('chunk_id')]
        }


# TranscriptProcessor - 메인 클래스 업데이트
class TranscriptProcessor:
    """LangChain 기반 자막 전처리 및 청킹 서비스"""
    
    def __init__(
        self, 
        chunking_strategy: str = "langchain_recursive",  # 기본값을 langchain으로 변경
        chunking_config: Optional[Dict[str, Any]] = None,
        embeddings = None  # SemanticChunker용
    ):
        """
        자막 프로세서를 초기화합니다.
        
        Args:
            chunking_strategy: 청킹 전략 이름
            chunking_config: 청킹 설정 딕셔너리
            embeddings: SemanticChunker용 임베딩 모델
        """
        self.chunking_strategy = chunking_strategy
        self.chunking_config = ChunkingConfig(
            **(chunking_config or {})
        )
        self.embeddings = embeddings
        
        # 청킹 전략 유효성 검사
        if not self._is_valid_strategy(chunking_strategy):
            raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")
        
        # LangChain 패키지 상태 확인
        self._check_langchain_availability()
        
        logger.info(f"LangChain-based transcript processor initialized with strategy: {chunking_strategy}")
        logger.info(f"LangChain text splitters available: {LANGCHAIN_TEXT_SPLITTERS_AVAILABLE}")
        logger.info(f"Semantic chunker available: {SEMANTIC_CHUNKER_AVAILABLE}")
    
    def _check_langchain_availability(self):
        """LangChain 패키지 사용 가능성 확인"""
        if not LANGCHAIN_TEXT_SPLITTERS_AVAILABLE:
            logger.warning("langchain-text-splitters not available, some features may be limited")
        
        if self.chunking_strategy == "semantic_chunker" and not SEMANTIC_CHUNKER_AVAILABLE:
            logger.warning("SemanticChunker not available, falling back to recursive chunker")
            self.chunking_strategy = "langchain_recursive"
        
        if self.chunking_strategy == "semantic_chunker" and not self.embeddings:
            logger.warning("No embeddings provided for semantic chunking, falling back to recursive chunker")
            self.chunking_strategy = "langchain_recursive"
    
    def clean_transcript_text(self, transcript_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        자막 텍스트를 정제합니다.
        
        Args:
            transcript_segments: 원본 자막 세그먼트들
            
        Returns:
            정제된 자막 세그먼트들
            
        Raises:
            TranscriptProcessingError: 자막 처리 중 오류 발생 시
        """
        if not transcript_segments:
            return []
        
        try:
            cleaned_segments = []
            
            for segment in transcript_segments:
                # 필수 필드 검증
                if not isinstance(segment, dict) or 'text' not in segment:
                    raise TranscriptProcessingError("Invalid transcript segment: missing 'text' field")
                
                text = segment.get('text', '')
                
                # 텍스트 정제
                cleaned_text = self._clean_text(text)
                
                # 노이즈 제거 (음악, 효과음, 박수 등)
                if self._is_noise(cleaned_text):
                    continue
                
                # 빈 텍스트 스킵
                if not cleaned_text.strip():
                    continue
                
                # 정제된 세그먼트 생성
                cleaned_segment = segment.copy()
                cleaned_segment['text'] = cleaned_text
                
                # end 시간이 없다면 계산
                if 'end' not in cleaned_segment:
                    start = cleaned_segment.get('start', 0.0)
                    duration = cleaned_segment.get('duration', 0.0)
                    cleaned_segment['end'] = start + duration
                
                cleaned_segments.append(cleaned_segment)
            
            logger.info(f"Cleaned {len(transcript_segments)} segments to {len(cleaned_segments)}")
            return cleaned_segments
            
        except Exception as e:
            logger.error(f"Error cleaning transcript text: {e}")
            raise TranscriptProcessingError(f"Failed to clean transcript text: {e}")
    
    def chunk_transcript_by_sentences(
        self,
        transcript_segments: List[Dict[str, Any]],
        max_chunk_size: Optional[int] = None,
        overlap_size: Optional[int] = None,
        preserve_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        선택된 전략으로 자막을 청킹합니다. (LangChain 기반)
        
        Args:
            transcript_segments: 자막 세그먼트들
            max_chunk_size: 최대 청크 크기 (문자 수)
            overlap_size: 오버랩 크기
            preserve_metadata: 메타데이터 보존 여부
            
        Returns:
            청킹된 결과 리스트
        """
        if not transcript_segments:
            return []
        
        # 설정 업데이트
        config = self.chunking_config
        if max_chunk_size is not None:
            config.max_chunk_size = max_chunk_size
            config.langchain_chunk_size = max_chunk_size
        if overlap_size is not None:
            config.overlap_size = overlap_size
            config.langchain_chunk_overlap = overlap_size
        config.preserve_metadata = preserve_metadata
        
        try:
            # 청킹 전략 실행
            chunker = self._get_chunking_strategy()
            chunks = chunker.chunk_segments(transcript_segments, config)
            
            logger.info(f"Chunked {len(transcript_segments)} segments into {len(chunks)} chunks using {self.chunking_strategy}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking transcript: {e}")
            raise TranscriptProcessingError(f"Failed to chunk transcript: {e}")
    
    def _get_chunking_strategy(self) -> ChunkingStrategyProtocol:
        """청킹 전략 인스턴스를 반환합니다."""
        strategy = self.chunking_strategy
        
        if strategy == "sentence_based":
            return SentenceBasedChunker()
        elif strategy == "langchain_recursive":
            return LangChainDocumentChunker("recursive")
        elif strategy == "langchain_character":
            return LangChainDocumentChunker("character")
        elif strategy == "langchain_token":
            return LangChainDocumentChunker("token")
        elif strategy == "langchain_spacy":
            return LangChainDocumentChunker("spacy")
        elif strategy == "langchain_sentence_transformer":
            return LangChainDocumentChunker("sentence_transformer")
        elif strategy == "semantic_chunker":
            return SemanticChunkerWrapper(self.embeddings)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _is_valid_strategy(self, strategy: str) -> bool:
        """청킹 전략이 유효한지 확인합니다."""
        valid_strategies = {
            "sentence_based",
            "langchain_recursive", 
            "langchain_character",
            "langchain_token",
            "langchain_spacy",
            "langchain_sentence_transformer",
            "semantic_chunker"
        }
        return strategy in valid_strategies
    
    def _clean_text(self, text: str) -> str:
        """텍스트를 정제합니다."""
        if not text:
            return ""
        
        # 앞뒤 공백 제거
        cleaned = text.strip()
        
        # 연속된 공백을 단일 공백으로 변경
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 개행 문자, 탭 문자를 공백으로 변경
        cleaned = re.sub(r'[\n\r\t]', ' ', cleaned)
        
        # 다시 연속 공백 제거
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 과도한 문장부호 정리 (예: !!! -> !, ??? -> ?, 단, ...는 보존)
        cleaned = re.sub(r'([!?])\1+', r'\1', cleaned)  # !!, ??? 처리 (2개 이상 -> 1개)
        cleaned = re.sub(r'\.{4,}', '...', cleaned)  # 4개 이상의 점은 ... 로
        
        # 최종 공백 제거
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _is_noise(self, text: str) -> bool:
        """텍스트가 노이즈인지 확인합니다."""
        if not text.strip():
            return True
        
        # 노이즈 패턴 정의
        noise_patterns = [
            r'^\[.*\]$',  # [음악], [박수] 등
            r'^\(.*\)$',  # (웃음), (기계음) 등
            r'^[^가-힣a-zA-Z0-9\s]*$',  # 특수문자만 있는 경우
        ]
        
        text_lower = text.lower().strip()
        
        for pattern in noise_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # 길이가 너무 짧은 의미없는 텍스트
        if len(text.strip()) <= 2:
            return True
        
        return False
    
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """청킹 결과 통계를 반환합니다."""
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'total_words': 0,
                'average_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'chunking_methods': []
            }
        
        total_chars = sum(chunk.get('char_count', len(chunk.get('text', ''))) for chunk in chunks)
        total_words = sum(chunk.get('word_count', len(chunk.get('text', '').split())) for chunk in chunks)
        chunk_sizes = [chunk.get('char_count', len(chunk.get('text', ''))) for chunk in chunks]
        
        # 사용된 청킹 방법 통계
        chunking_methods = {}
        for chunk in chunks:
            method = chunk.get('chunking_method', 'unknown')
            chunking_methods[method] = chunking_methods.get(method, 0) + 1
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'total_words': total_words,
            'average_chunk_size': total_chars / len(chunks),
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'total_duration': self._calculate_total_duration(chunks),
            'chunking_methods': chunking_methods,
            'langchain_features_used': LANGCHAIN_TEXT_SPLITTERS_AVAILABLE
        }
    
    def _calculate_total_duration(self, chunks: List[Dict[str, Any]]) -> float:
        """전체 청크의 총 길이를 계산합니다."""
        if not chunks:
            return 0.0
        
        try:
            min_start = min(chunk.get('start_time', 0) for chunk in chunks)
            max_end = max(chunk.get('end_time', 0) for chunk in chunks)
            return max_end - min_start
        except (ValueError, TypeError):
            return 0.0
    
    def create_langchain_documents(self, chunks: List[Dict[str, Any]]) -> List[Document]:
        """청크들을 LangChain Document 형식으로 변환합니다."""
        documents = []
        
        for chunk in chunks:
            text = chunk.get('text', '').strip()
            if not text:
                continue
            
            metadata = {
                'chunk_id': chunk.get('chunk_id'),
                'start_time': chunk.get('start_time', 0.0),
                'end_time': chunk.get('end_time', 0.0),
                'word_count': chunk.get('word_count', 0),
                'char_count': chunk.get('char_count', 0),
                'chunking_method': chunk.get('chunking_method', 'unknown'),
                'source_segments_count': len(chunk.get('source_segments', []))
            }
            
            document = Document(page_content=text, metadata=metadata)
            documents.append(document)
        
        return documents 