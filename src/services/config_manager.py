"""
설정 관리자 - UI 연동을 위한 통합 설정 시스템

사용자가 UI에서 쉽게 설정할 수 있도록 하는 통합 설정 관리 시스템입니다.
설정 저장/로드, 프리셋, 검증, 메타데이터 등을 제공합니다.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from pathlib import Path

from src.utils.logger import get_project_logger
from src.utils.exceptions import ConfigurationError

logger = get_project_logger(__name__)


class UISettingType(Enum):
    """UI 설정 타입"""
    SELECT = "select"           # 드롭다운 선택
    SLIDER = "slider"          # 슬라이더 (숫자)
    INPUT = "input"            # 텍스트 입력
    CHECKBOX = "checkbox"      # 체크박스
    MULTISELECT = "multiselect"  # 다중 선택


@dataclass
class UISettingOption:
    """UI 설정 옵션 메타데이터"""
    key: str                    # 설정 키
    name: str                   # 사용자 친화적 이름
    description: str            # 설명
    setting_type: UISettingType # UI 타입
    default_value: Any          # 기본값
    options: List[Any] = field(default_factory=list)  # 선택지 (SELECT, MULTISELECT용)
    min_value: Optional[float] = None    # 최소값 (SLIDER용)
    max_value: Optional[float] = None    # 최대값 (SLIDER용)
    step: Optional[float] = None         # 스텝 (SLIDER용)
    validator: Optional[str] = None      # 검증 함수명
    category: str = "기본"              # 카테고리
    advanced: bool = False              # 고급 옵션 여부


@dataclass
class ServicePreset:
    """서비스 프리셋"""
    name: str                   # 프리셋 이름
    description: str            # 프리셋 설명
    chunking_config: Dict[str, Any] = field(default_factory=dict)
    embedding_config: Dict[str, Any] = field(default_factory=dict)
    search_config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)  # 태그 (예: "초보자", "고성능", "한국어")


# ============================================================================
# 청킹 설정 옵션들
# ============================================================================

CHUNKING_UI_OPTIONS = [
    UISettingOption(
        key="chunking_strategy",
        name="청킹 방법",
        description="텍스트를 나누는 방법을 선택합니다",
        setting_type=UISettingType.SELECT,
        default_value="langchain_recursive",
        options=[
            {"value": "langchain_recursive", "label": "재귀적 분할 (권장)", "description": "문서 구조를 고려한 지능적 분할"},
            {"value": "langchain_token", "label": "토큰 기반 분할", "description": "정확한 토큰 수 기반 분할"},
            {"value": "langchain_spacy", "label": "SpaCy 분할", "description": "자연어 처리 기반 분할"},
            {"value": "semantic_chunker", "label": "의미적 분할", "description": "의미 유사성 기반 분할 (고급)"}
        ],
        category="청킹",
        advanced=False
    ),
    UISettingOption(
        key="max_chunk_size",
        name="청크 최대 크기",
        description="각 청크의 최대 문자 수를 설정합니다",
        setting_type=UISettingType.SLIDER,
        default_value=400,
        min_value=100,
        max_value=2000,
        step=50,
        category="청킹",
        advanced=False
    ),
    UISettingOption(
        key="overlap_size", 
        name="청크 겹침 크기",
        description="인접한 청크 간의 겹치는 부분 크기입니다",
        setting_type=UISettingType.SLIDER,
        default_value=50,
        min_value=0,
        max_value=200,
        step=10,
        category="청킹",
        advanced=False
    ),
    UISettingOption(
        key="preserve_sentence_boundaries",
        name="문장 경계 보존",
        description="문장을 중간에 자르지 않고 완전한 문장 단위로 유지합니다",
        setting_type=UISettingType.CHECKBOX,
        default_value=True,
        category="청킹",
        advanced=False
    ),
    UISettingOption(
        key="langchain_separators",
        name="구분자 패턴",
        description="텍스트를 나누는 기준이 되는 구분자들입니다",
        setting_type=UISettingType.MULTISELECT,
        default_value=["\\n\\n", "\\n", "。", ".", "!", "?", " ", ""],
        options=[
            {"value": "\\n\\n", "label": "단락 구분 (\\n\\n)"},
            {"value": "\\n", "label": "줄바꿈 (\\n)"},
            {"value": "。", "label": "한국어 마침표 (。)"},
            {"value": ".", "label": "영어 마침표 (.)"},
            {"value": "!", "label": "느낌표 (!)"},
            {"value": "?", "label": "물음표 (?)"},
            {"value": " ", "label": "공백"},
            {"value": "", "label": "문자 단위"}
        ],
        category="청킹",
        advanced=True
    )
]


# ============================================================================
# 임베딩 설정 옵션들
# ============================================================================

EMBEDDING_UI_OPTIONS = [
    UISettingOption(
        key="provider",
        name="임베딩 제공자",
        description="임베딩을 생성할 AI 서비스를 선택합니다",
        setting_type=UISettingType.SELECT,
        default_value="huggingface",
        options=[
            {"value": "openai", "label": "OpenAI (고품질, 유료)", "description": "GPT 기반 고성능 임베딩"},
            {"value": "huggingface", "label": "HuggingFace (무료)", "description": "오픈소스 임베딩 모델"}
        ],
        category="임베딩",
        advanced=False
    ),
    UISettingOption(
        key="model_name",
        name="모델 선택",
        description="사용할 임베딩 모델을 선택합니다",
        setting_type=UISettingType.SELECT,
        default_value="sentence-transformers/all-MiniLM-L6-v2",
        options=[
            # OpenAI 모델들
            {"value": "text-embedding-ada-002", "label": "OpenAI Ada-002 (권장)", "provider": "openai"},
            {"value": "text-embedding-3-small", "label": "OpenAI Embedding-3-Small", "provider": "openai"},
            {"value": "text-embedding-3-large", "label": "OpenAI Embedding-3-Large", "provider": "openai"},
            
            # HuggingFace 모델들
            {"value": "sentence-transformers/all-MiniLM-L6-v2", "label": "All-MiniLM-L6-v2 (경량, 빠름)", "provider": "huggingface"},
            {"value": "sentence-transformers/all-mpnet-base-v2", "label": "All-MPNet-Base-v2 (고성능)", "provider": "huggingface"},
            {"value": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "label": "다국어 MiniLM (한국어 지원)", "provider": "huggingface"},
            {"value": "jhgan/ko-sroberta-multitask", "label": "한국어 특화 RoBERTa", "provider": "huggingface"}
        ],
        category="임베딩",
        advanced=False
    ),
    UISettingOption(
        key="batch_size",
        name="배치 크기",
        description="한 번에 처리할 텍스트 수입니다 (메모리 사용량에 영향)",
        setting_type=UISettingType.SLIDER,
        default_value=100,
        min_value=10,
        max_value=500,
        step=10,
        category="임베딩",
        advanced=True
    ),
    UISettingOption(
        key="normalize_embeddings",
        name="임베딩 정규화",
        description="벡터를 정규화하여 코사인 유사도 계산을 최적화합니다",
        setting_type=UISettingType.CHECKBOX,
        default_value=True,
        category="임베딩",
        advanced=True
    ),
    UISettingOption(
        key="device",
        name="처리 장치",
        description="임베딩 계산에 사용할 장치입니다",
        setting_type=UISettingType.SELECT,
        default_value="cpu",
        options=[
            {"value": "cpu", "label": "CPU (안정적)", "description": "모든 환경에서 작동"},
            {"value": "cuda", "label": "GPU (고속)", "description": "NVIDIA GPU 필요"}
        ],
        category="임베딩",
        advanced=True
    )
]


# ============================================================================
# 검색 설정 옵션들
# ============================================================================

SEARCH_UI_OPTIONS = [
    UISettingOption(
        key="search_type",
        name="검색 방법",
        description="텍스트를 찾는 방법을 선택합니다",
        setting_type=UISettingType.SELECT,
        default_value="similarity",
        options=[
            {"value": "similarity", "label": "유사도 검색 (권장)", "description": "의미적 유사성 기반 검색"},
            {"value": "mmr", "label": "MMR 검색", "description": "다양성을 고려한 검색"},
            {"value": "bm25", "label": "BM25 검색", "description": "키워드 정확도 우선 검색"},
            {"value": "hybrid", "label": "하이브리드 검색", "description": "의미적 + 키워드 검색 결합"}
        ],
        category="검색",
        advanced=False
    ),
    UISettingOption(
        key="k",
        name="검색 결과 수",
        description="한 번에 가져올 검색 결과의 개수입니다",
        setting_type=UISettingType.SLIDER,
        default_value=5,
        min_value=1,
        max_value=20,
        step=1,
        category="검색",
        advanced=False
    ),
    UISettingOption(
        key="score_threshold",
        name="유사도 임계값",
        description="이 값보다 낮은 유사도의 결과는 제외합니다 (0~1)",
        setting_type=UISettingType.SLIDER,
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        category="검색",
        advanced=True
    ),
    UISettingOption(
        key="lambda_mult",
        name="MMR 다양성",
        description="MMR 검색에서 다양성과 관련성의 균형을 조절합니다",
        setting_type=UISettingType.SLIDER,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        category="검색",
        advanced=True
    ),
    UISettingOption(
        key="hybrid_weights",
        name="하이브리드 가중치",
        description="하이브리드 검색에서 의미적/키워드 검색 비율 [의미적, 키워드]",
        setting_type=UISettingType.INPUT,
        default_value="[0.5, 0.5]",
        category="검색",
        advanced=True,
        validator="validate_hybrid_weights"
    )
]


# ============================================================================
# 프리셋 정의
# ============================================================================

PRESET_CONFIGURATIONS = [
    ServicePreset(
        name="초보자용 - 기본 설정",
        description="처음 사용하는 사용자를 위한 안정적이고 빠른 설정",
        chunking_config={
            "chunking_strategy": "langchain_recursive",
            "max_chunk_size": 300,
            "overlap_size": 30,
            "preserve_sentence_boundaries": True
        },
        embedding_config={
            "provider": "huggingface",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 50,
            "device": "cpu"
        },
        search_config={
            "search_type": "similarity",
            "k": 5,
            "score_threshold": 0.0
        },
        tags=["초보자", "빠름", "안정적"]
    ),
    ServicePreset(
        name="한국어 최적화",
        description="한국어 콘텐츠에 최적화된 설정",
        chunking_config={
            "chunking_strategy": "langchain_spacy",
            "max_chunk_size": 400,
            "overlap_size": 50,
            "langchain_separators": ["\\n\\n", "\\n", "。", ".", "!", "?", " ", ""]
        },
        embedding_config={
            "provider": "huggingface",
            "model_name": "jhgan/ko-sroberta-multitask",
            "batch_size": 100,
            "device": "cpu"
        },
        search_config={
            "search_type": "hybrid",
            "k": 7,
            "hybrid_weights": [0.7, 0.3]
        },
        tags=["한국어", "고품질", "하이브리드"]
    ),
    ServicePreset(
        name="고성능 - OpenAI",
        description="OpenAI API를 사용한 최고 품질 설정 (유료)",
        chunking_config={
            "chunking_strategy": "semantic_chunker",
            "max_chunk_size": 500,
            "overlap_size": 75
        },
        embedding_config={
            "provider": "openai", 
            "model_name": "text-embedding-ada-002",
            "batch_size": 200,
            "normalize_embeddings": True
        },
        search_config={
            "search_type": "mmr",
            "k": 10,
            "lambda_mult": 0.7,
            "score_threshold": 0.1
        },
        tags=["고성능", "유료", "의미적분할"]
    ),
    ServicePreset(
        name="빠른 처리 - 경량",
        description="속도를 중시하는 가벼운 설정",
        chunking_config={
            "chunking_strategy": "langchain_recursive",
            "max_chunk_size": 200,
            "overlap_size": 20
        },
        embedding_config={
            "provider": "huggingface",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 200,
            "device": "cpu"
        },
        search_config={
            "search_type": "similarity",
            "k": 3,
            "score_threshold": 0.0
        },
        tags=["빠름", "경량", "무료"]
    )
]


# ============================================================================
# 설정 관리자 메인 클래스
# ============================================================================

class ConfigurationManager:
    """UI 연동을 위한 통합 설정 관리자"""
    
    def __init__(self, config_dir: str = "./configs"):
        """
        설정 관리자를 초기화합니다.
        
        Args:
            config_dir: 설정 파일을 저장할 디렉토리
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # 현재 설정
        self.current_config = {
            "chunking": {},
            "embedding": {},
            "search": {}
        }
        
        # 기본 설정 로드
        self.load_default_config()
        
        logger.info(f"Configuration manager initialized. Config dir: {config_dir}")
    
    def get_ui_options(self, category: str = None) -> List[UISettingOption]:
        """
        UI 옵션들을 가져옵니다.
        
        Args:
            category: 특정 카테고리만 필터링 ("chunking", "embedding", "search")
            
        Returns:
            UI 설정 옵션 리스트
        """
        all_options = CHUNKING_UI_OPTIONS + EMBEDDING_UI_OPTIONS + SEARCH_UI_OPTIONS
        
        if category:
            category_map = {
                "chunking": "청킹",
                "embedding": "임베딩", 
                "search": "검색"
            }
            target_category = category_map.get(category, category)
            return [opt for opt in all_options if opt.category == target_category]
        
        return all_options
    
    def get_presets(self, tags: List[str] = None) -> List[ServicePreset]:
        """
        프리셋을 가져옵니다.
        
        Args:
            tags: 필터링할 태그들
            
        Returns:
            서비스 프리셋 리스트
        """
        if not tags:
            return PRESET_CONFIGURATIONS
        
        filtered_presets = []
        for preset in PRESET_CONFIGURATIONS:
            if any(tag in preset.tags for tag in tags):
                filtered_presets.append(preset)
        
        return filtered_presets
    
    def apply_preset(self, preset_name: str):
        """
        프리셋을 적용합니다.
        
        Args:
            preset_name: 적용할 프리셋 이름
        """
        preset = next((p for p in PRESET_CONFIGURATIONS if p.name == preset_name), None)
        if not preset:
            raise ConfigurationError(f"Preset not found: {preset_name}")
        
        self.current_config = {
            "chunking": preset.chunking_config.copy(),
            "embedding": preset.embedding_config.copy(),
            "search": preset.search_config.copy()
        }
        
        logger.info(f"Applied preset: {preset_name}")
    
    def update_setting(self, category: str, key: str, value: Any):
        """
        특정 설정을 업데이트합니다.
        
        Args:
            category: 설정 카테고리 ("chunking", "embedding", "search")
            key: 설정 키
            value: 새로운 값
        """
        if category not in self.current_config:
            raise ConfigurationError(f"Invalid category: {category}")
        
        # 값 검증
        self._validate_setting_value(category, key, value)
        
        self.current_config[category][key] = value
        logger.debug(f"Updated {category}.{key} = {value}")
    
    def get_current_config(self) -> Dict[str, Any]:
        """현재 설정을 가져옵니다."""
        return self.current_config.copy()
    
    def save_config(self, name: str):
        """
        현재 설정을 파일로 저장합니다.
        
        Args:
            name: 저장할 설정 이름
        """
        config_file = self.config_dir / f"{name}.json"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved: {config_file}")
    
    def load_config(self, name: str):
        """
        저장된 설정을 로드합니다.
        
        Args:
            name: 로드할 설정 이름
        """
        config_file = self.config_dir / f"{name}.json"
        
        if not config_file.exists():
            raise ConfigurationError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.current_config = json.load(f)
        
        logger.info(f"Configuration loaded: {config_file}")
    
    def get_saved_configs(self) -> List[str]:
        """저장된 설정 이름 목록을 가져옵니다."""
        config_files = self.config_dir.glob("*.json")
        return [f.stem for f in config_files]
    
    def load_default_config(self):
        """기본 설정을 로드합니다."""
        # 각 카테고리의 기본값 수집
        for option in CHUNKING_UI_OPTIONS:
            self.current_config["chunking"][option.key] = option.default_value
        
        for option in EMBEDDING_UI_OPTIONS:
            self.current_config["embedding"][option.key] = option.default_value
        
        for option in SEARCH_UI_OPTIONS:
            self.current_config["search"][option.key] = option.default_value
    
    def _validate_setting_value(self, category: str, key: str, value: Any):
        """설정 값을 검증합니다."""
        # 해당 설정의 옵션 찾기
        all_options = self.get_ui_options()
        option = next((opt for opt in all_options if opt.key == key), None)
        
        if not option:
            logger.warning(f"No validation rule for {category}.{key}")
            return
        
        # 타입별 검증
        if option.setting_type == UISettingType.SLIDER:
            if not isinstance(value, (int, float)):
                raise ConfigurationError(f"{key} must be a number")
            
            if option.min_value is not None and value < option.min_value:
                raise ConfigurationError(f"{key} must be >= {option.min_value}")
            
            if option.max_value is not None and value > option.max_value:
                raise ConfigurationError(f"{key} must be <= {option.max_value}")
        
        elif option.setting_type == UISettingType.SELECT:
            valid_values = [opt["value"] for opt in option.options]
            if value not in valid_values:
                raise ConfigurationError(f"{key} must be one of: {valid_values}")
        
        # 커스텀 검증 함수
        if option.validator:
            validator_func = getattr(self, option.validator, None)
            if validator_func and callable(validator_func):
                validator_func(value)
    
    def validate_hybrid_weights(self, value):
        """하이브리드 가중치 검증"""
        if isinstance(value, str):
            try:
                # 문자열을 리스트로 변환
                import ast
                weights = ast.literal_eval(value)
            except:
                raise ConfigurationError("하이브리드 가중치는 [0.5, 0.5] 형식으로 입력해주세요")
        else:
            weights = value
        
        if not isinstance(weights, list) or len(weights) != 2:
            raise ConfigurationError("하이브리드 가중치는 2개의 값을 가져야 합니다")
        
        if not all(isinstance(w, (int, float)) and 0 <= w <= 1 for w in weights):
            raise ConfigurationError("가중치는 0과 1 사이의 숫자여야 합니다")
    
    def create_service_configs(self):
        """
        현재 설정을 바탕으로 서비스 config 객체들을 생성합니다.
        
        Returns:
            (ChunkingConfig, EmbeddingConfig, SearchConfig) 튜플
        """
        # 동적 임포트 (순환 임포트 방지)
        from .transcript_processor import ChunkingConfig
        from .embedding_service import EmbeddingConfig
        from .similarity_search_service import SearchConfig, SearchType
        
        # ChunkingConfig 생성 (chunking_strategy는 제외)
        chunking_config_data = self.current_config["chunking"].copy()
        if "chunking_strategy" in chunking_config_data:
            chunking_config_data.pop("chunking_strategy")
        
        chunking_config = ChunkingConfig(**chunking_config_data)
        
        # EmbeddingConfig 생성
        embedding_config = EmbeddingConfig(**self.current_config["embedding"])
        
        # SearchConfig 생성 (SearchType enum 변환, hybrid_weights 처리)
        search_config_data = self.current_config["search"].copy()
        if "search_type" in search_config_data:
            search_config_data["search_type"] = SearchType(search_config_data["search_type"])
        
        # hybrid_weights 문자열을 리스트로 변환
        if "hybrid_weights" in search_config_data and isinstance(search_config_data["hybrid_weights"], str):
            try:
                import ast
                search_config_data["hybrid_weights"] = ast.literal_eval(search_config_data["hybrid_weights"])
            except:
                search_config_data["hybrid_weights"] = [0.5, 0.5]  # 기본값
        
        search_config = SearchConfig(**search_config_data)
        
        return chunking_config, embedding_config, search_config


# ============================================================================
# 팩토리 함수
# ============================================================================

def create_configuration_manager(config_dir: str = "./configs") -> ConfigurationManager:
    """
    설정 관리자를 생성합니다.
    
    Args:
        config_dir: 설정 파일 디렉토리
        
    Returns:
        ConfigurationManager 인스턴스
    """
    return ConfigurationManager(config_dir) 