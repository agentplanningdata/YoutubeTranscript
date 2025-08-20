"""
서비스 팩토리 - UI 연동을 위한 통합 서비스 생성

ConfigurationManager와 연동되어 사용자 설정에 따라
모든 서비스를 동적으로 생성하고 관리합니다.
"""

from typing import Optional, Tuple, Dict, Any, List
from dataclasses import asdict

from .config_manager import ConfigurationManager
from .transcript_processor import TranscriptProcessor, ChunkingConfig
from .embedding_service import EmbeddingService, EmbeddingConfig, create_embedding_service
from .similarity_search_service import SimilaritySearchService, SearchConfig, create_similarity_search_service

from src.utils.logger import get_project_logger
from src.utils.exceptions import ConfigurationError

logger = get_project_logger(__name__)


class ServiceFactory:
    """
    UI 설정 기반 서비스 팩토리
    
    ConfigurationManager의 설정을 바탕으로 모든 서비스를
    동적으로 생성하고 재구성할 수 있습니다.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        """
        서비스 팩토리를 초기화합니다.
        
        Args:
            config_manager: 설정 관리자
        """
        self.config_manager = config_manager
        
        # 현재 생성된 서비스들
        self._transcript_processor: Optional[TranscriptProcessor] = None
        self._embedding_service: Optional[EmbeddingService] = None
        self._similarity_search_service: Optional[SimilaritySearchService] = None
        
        logger.info("Service factory initialized")
    
    def get_transcript_processor(self, force_recreate: bool = False) -> TranscriptProcessor:
        """
        청킹 서비스를 가져오거나 생성합니다.
        
        Args:
            force_recreate: 강제로 재생성 여부
            
        Returns:
            TranscriptProcessor 인스턴스
        """
        if self._transcript_processor is None or force_recreate:
            chunking_config, _, _ = self.config_manager.create_service_configs()
            
            # 청킹 전략 추출
            chunking_strategy = self.config_manager.current_config["chunking"].get(
                "chunking_strategy", "langchain_recursive"
            )
            
            # SemanticChunker용 임베딩 (필요한 경우)
            embeddings = None
            if chunking_strategy == "semantic_chunker":
                try:
                    embeddings = self.get_embedding_service().embeddings
                except Exception as e:
                    logger.warning(f"Failed to get embeddings for semantic chunking: {e}")
                    logger.info("Falling back to recursive chunking")
                    chunking_strategy = "langchain_recursive"
            
            # 서비스 생성
            self._transcript_processor = TranscriptProcessor(
                chunking_strategy=chunking_strategy,
                chunking_config=asdict(chunking_config),
                embeddings=embeddings
            )
            
            logger.info(f"Transcript processor created with strategy: {chunking_strategy}")
        
        return self._transcript_processor
    
    def get_embedding_service(self, force_recreate: bool = False) -> EmbeddingService:
        """
        임베딩 서비스를 가져오거나 생성합니다.
        
        Args:
            force_recreate: 강제로 재생성 여부
            
        Returns:
            EmbeddingService 인스턴스
        """
        if self._embedding_service is None or force_recreate:
            _, embedding_config, _ = self.config_manager.create_service_configs()
            
            self._embedding_service = EmbeddingService(embedding_config)
            
            logger.info(f"Embedding service created with provider: {embedding_config.provider}")
        
        return self._embedding_service
    
    def get_similarity_search_service(self, force_recreate: bool = False) -> SimilaritySearchService:
        """
        검색 서비스를 가져오거나 생성합니다.
        
        Args:
            force_recreate: 강제로 재생성 여부
            
        Returns:
            SimilaritySearchService 인스턴스
        """
        if self._similarity_search_service is None or force_recreate:
            _, _, search_config = self.config_manager.create_service_configs()
            
            # 임베딩 서비스 먼저 생성
            embedding_service = self.get_embedding_service()
            
            self._similarity_search_service = SimilaritySearchService(
                embedding_service, search_config
            )
            
            logger.info(f"Similarity search service created with type: {search_config.search_type}")
        
        return self._similarity_search_service
    
    def recreate_all_services(self):
        """모든 서비스를 재생성합니다."""
        logger.info("Recreating all services with new configuration")
        
        self._transcript_processor = None
        self._embedding_service = None
        self._similarity_search_service = None
        
        # 순서대로 재생성 (의존성 고려)
        self.get_embedding_service(force_recreate=True)
        self.get_transcript_processor(force_recreate=True)
        self.get_similarity_search_service(force_recreate=True)
        
        logger.info("All services recreated successfully")
    
    def get_all_services(self) -> Tuple[TranscriptProcessor, EmbeddingService, SimilaritySearchService]:
        """
        모든 서비스를 한번에 가져옵니다.
        
        Returns:
            (TranscriptProcessor, EmbeddingService, SimilaritySearchService) 튜플
        """
        return (
            self.get_transcript_processor(),
            self.get_embedding_service(), 
            self.get_similarity_search_service()
        )
    
    def validate_current_config(self) -> Dict[str, Any]:
        """
        현재 설정의 유효성을 검사합니다.
        
        Returns:
            검증 결과 딕셔너리
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "config_summary": {}
        }
        
        try:
            # 설정으로 config 객체들 생성해보기
            chunking_config, embedding_config, search_config = self.config_manager.create_service_configs()
            
            # 설정 요약 생성
            validation_result["config_summary"] = {
                "chunking": {
                    "strategy": self.config_manager.current_config["chunking"].get("chunking_strategy"),
                    "chunk_size": chunking_config.max_chunk_size,
                    "overlap": chunking_config.overlap_size
                },
                "embedding": {
                    "provider": embedding_config.provider,
                    "model": embedding_config.model_name,
                    "device": embedding_config.device
                },
                "search": {
                    "type": search_config.search_type.value,
                    "k": search_config.k,
                    "threshold": search_config.score_threshold
                }
            }
            
            # 특별한 검증 로직들
            self._validate_model_compatibility(embedding_config, validation_result)
            self._validate_performance_settings(chunking_config, embedding_config, validation_result)
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Configuration error: {str(e)}")
        
        return validation_result
    
    def _validate_model_compatibility(self, embedding_config: EmbeddingConfig, result: Dict[str, Any]):
        """임베딩 모델 호환성 검증"""
        # OpenAI API 키 확인
        if embedding_config.provider == "openai" and not embedding_config.api_key:
            result["errors"].append("OpenAI provider requires API key")
        
        # CUDA 사용 가능성 확인 (실제로는 torch.cuda.is_available() 등으로)
        if embedding_config.device == "cuda":
            result["warnings"].append("GPU acceleration requires CUDA-compatible hardware")
    
    def _validate_performance_settings(
        self, 
        chunking_config: ChunkingConfig, 
        embedding_config: EmbeddingConfig, 
        result: Dict[str, Any]
    ):
        """성능 관련 설정 검증"""
        # 메모리 사용량 예측
        estimated_memory_mb = (
            chunking_config.max_chunk_size * embedding_config.batch_size * 0.001
        )
        
        if estimated_memory_mb > 1000:  # 1GB 초과
            result["warnings"].append(f"High memory usage expected (~{estimated_memory_mb:.0f}MB)")
        
        # 성능 권장사항
        if chunking_config.max_chunk_size > 1000:
            result["warnings"].append("Large chunk size may slow down processing")
        
        if embedding_config.batch_size > 200 and embedding_config.device == "cpu":
            result["warnings"].append("Large batch size with CPU may be slow")
    
    def get_performance_info(self) -> Dict[str, Any]:
        """현재 설정의 예상 성능 정보를 반환합니다."""
        config = self.config_manager.current_config
        
        # 간단한 성능 추정
        chunk_size = config["chunking"].get("max_chunk_size", 400)
        batch_size = config["embedding"].get("batch_size", 100)
        provider = config["embedding"].get("provider", "huggingface")
        device = config["embedding"].get("device", "cpu")
        
        # 처리 속도 추정 (chunks per second)
        if provider == "openai":
            base_speed = 50  # API 기반
        elif device == "cuda":
            base_speed = 200  # GPU 가속
        else:
            base_speed = 100  # CPU
        
        # 청크 크기와 배치 크기에 따른 조정
        speed_factor = min(batch_size / 100, 2.0) * (400 / max(chunk_size, 100))
        estimated_speed = int(base_speed * speed_factor)
        
        return {
            "estimated_chunks_per_second": estimated_speed,
            "estimated_memory_usage_mb": chunk_size * batch_size * 0.001,
            "gpu_acceleration": device == "cuda",
            "api_usage": provider == "openai",
            "performance_level": self._get_performance_level(estimated_speed),
            "recommendations": self._get_performance_recommendations(config)
        }
    
    def _get_performance_level(self, speed: int) -> str:
        """속도에 따른 성능 레벨 반환"""
        if speed >= 150:
            return "high"
        elif speed >= 100:
            return "medium"
        else:
            return "low"
    
    def _get_performance_recommendations(self, config: Dict[str, Any]) -> List[str]:
        """성능 개선 권장사항"""
        recommendations = []
        
        provider = config["embedding"].get("provider", "huggingface")
        device = config["embedding"].get("device", "cpu")
        chunk_size = config["chunking"].get("max_chunk_size", 400)
        batch_size = config["embedding"].get("batch_size", 100)
        
        if device == "cpu" and provider == "huggingface":
            recommendations.append("GPU가 있다면 device를 'cuda'로 설정하여 속도를 향상시킬 수 있습니다")
        
        if chunk_size > 800:
            recommendations.append("청크 크기를 줄이면 처리 속도가 빨라집니다")
        
        if batch_size < 50:
            recommendations.append("배치 크기를 늘리면 효율성이 향상될 수 있습니다")
        
        if provider == "huggingface" and "multilingual" not in config["embedding"].get("model_name", ""):
            recommendations.append("한국어 콘텐츠라면 다국어 모델이나 한국어 특화 모델을 고려해보세요")
        
        return recommendations


# ============================================================================
# 팩토리 함수
# ============================================================================

def create_service_factory(config_manager: Optional[ConfigurationManager] = None) -> ServiceFactory:
    """
    서비스 팩토리를 생성합니다.
    
    Args:
        config_manager: 설정 관리자 (없으면 기본 생성)
        
    Returns:
        ServiceFactory 인스턴스
    """
    if config_manager is None:
        from .config_manager import create_configuration_manager
        config_manager = create_configuration_manager()
    
    return ServiceFactory(config_manager) 