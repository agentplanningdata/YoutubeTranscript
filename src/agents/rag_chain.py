"""
LCEL 기반 RAG 체인

LangChain Expression Language를 사용하여 
검색-증강-생성(RAG) 파이프라인을 구현합니다.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

# LangChain Core
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

# 프로젝트 모듈
from src.services.similarity_search_service import SimilaritySearchService, SearchConfig, SearchType
from src.services.embedding_service import EmbeddingService
from src.services.service_factory import ServiceFactory, create_service_factory
from src.services.config_manager import ConfigurationManager
from src.utils.logger import get_project_logger
from src.utils.exceptions import ProcessingError
from src.utils.langfuse_utils import get_langfuse_callbacks

logger = get_project_logger(__name__)


@dataclass
class RAGConfig:
    """RAG 체인 설정"""
    
    # LLM 설정
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # 검색 설정
    search_type: str = "hybrid"  # similarity, hybrid, bm25, mmr
    max_docs: int = 5
    score_threshold: Optional[float] = None
    
    # 프롬프트 설정
    include_timestamps: bool = True
    include_sources: bool = True
    language: str = "korean"
    
    # 컨텍스트 포맷 설정
    max_context_length: int = 2000
    show_metadata: bool = True


class RAGChain:
    """LCEL 기반 RAG 체인"""
    
    def __init__(self, 
                 similarity_service: SimilaritySearchService,
                 rag_config: RAGConfig = None):
        """
        RAG 체인 초기화
        
        Args:
            similarity_service: 검색 서비스
            rag_config: RAG 설정
        """
        self.similarity_service = similarity_service
        self.config = rag_config or RAGConfig()
        
        # LLM 초기화
        import os
        google_api_key = os.getenv("GOOGLE_API_KEY")
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            google_api_key=google_api_key
        )
        
        # 프롬프트 템플릿 생성
        self.prompt_template = self._create_prompt_template()
        
        # RAG 체인 구성
        self.chain = self._build_chain()
        
        logger.info(f"RAG Chain initialized with {self.config.search_type} search")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """RAG용 프롬프트 템플릿 생성"""
        
        system_prompt = f"""
        당신은 YouTube 영상 분석 전문 AI 어시스턴트입니다.
        
        주요 역할:
        1. 제공된 영상 자막 컨텍스트를 바탕으로 정확한 답변을 제공합니다
        2. 타임스탬프와 출처를 포함하여 사용자가 해당 내용을 찾을 수 있도록 도웁니다
        3. 컨텍스트에 없는 정보는 추측하지 않고 명확히 표시합니다
        
        답변 가이드라인:
        - {self.config.language}로 답변하세요
        - 컨텍스트의 정보만 사용하여 답변하세요
        - 확실하지 않은 내용은 "컨텍스트에서 명확하지 않습니다"라고 명시하세요
        - 관련된 타임스탬프를 포함하여 답변하세요
        - 논리적이고 구조화된 답변을 제공하세요
        """
        
        if self.config.include_timestamps:
            user_template = """
            컨텍스트:
            {context}
            
            출처 정보:
            {sources}
            
            질문: {question}
            
            위 컨텍스트를 바탕으로 답변해주세요. 관련된 타임스탬프 [MM:SS]도 함께 포함해주세요.
            """
        else:
            user_template = """
            컨텍스트:
            {context}
            
            질문: {question}
            
            위 컨텍스트를 바탕으로 답변해주세요.
            """
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
    
    def _format_context(self, docs: List[Document]) -> str:
        """검색된 문서들을 컨텍스트로 포맷팅"""
        if not docs:
            return "관련 컨텍스트를 찾을 수 없습니다."
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(docs):
            # 타임스탬프 추출
            start_time = doc.metadata.get('start_time', 0)
            end_time = doc.metadata.get('end_time', start_time)
            
            # 시간 포맷팅
            start_min, start_sec = divmod(int(start_time), 60)
            timestamp = f"[{start_min:02d}:{start_sec:02d}]"
            
            # 컨텍스트 구성
            if self.config.include_timestamps:
                context_part = f"{timestamp} {doc.page_content}"
            else:
                context_part = doc.page_content
            
            # 길이 제한 확인
            if total_length + len(context_part) > self.config.max_context_length:
                context_parts.append("... (컨텍스트가 잘렸습니다)")
                break
            
            context_parts.append(context_part)
            total_length += len(context_part)
        
        return "\n\n".join(context_parts)
    
    def _format_sources(self, docs: List[Document]) -> str:
        """출처 정보 포맷팅"""
        if not docs or not self.config.include_sources:
            return ""
        
        sources = []
        for i, doc in enumerate(docs, 1):
            video_id = doc.metadata.get('video_id', 'Unknown')
            start_time = doc.metadata.get('start_time', 0)
            end_time = doc.metadata.get('end_time', start_time)
            
            start_min, start_sec = divmod(int(start_time), 60)
            timestamp = f"{start_min:02d}:{start_sec:02d}"
            
            source = f"출처 {i}: 영상 {video_id} ({timestamp})"
            sources.append(source)
        
        return "\n".join(sources)
    
    def _build_chain(self):
        """LCEL RAG 체인 구축"""
        
        # 1단계: 병렬로 문서 검색 및 질문 전달
        retrieval_chain = RunnableParallel({
            "docs": self.similarity_service.retriever,
            "question": RunnablePassthrough()
        })
        
        # 2단계: 컨텍스트 및 출처 포맷팅
        format_chain = RunnableParallel({
            "context": RunnableLambda(lambda x: self._format_context(x["docs"])),
            "sources": RunnableLambda(lambda x: self._format_sources(x["docs"])),
            "question": RunnableLambda(lambda x: x["question"]),
            "raw_docs": RunnableLambda(lambda x: x["docs"])  # 원본 문서도 보존
        })
        
        # 3단계: 전체 체인 구성
        full_chain = (
            retrieval_chain
            | format_chain 
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        return full_chain
    
    def invoke(self, question: str, **kwargs) -> Dict[str, Any]:
        """RAG 체인 실행"""
        start_time = time.time()
        
        try:
            # Langfuse 콜백 추가
            config = {"callbacks": get_langfuse_callbacks()}
            
            # 체인 실행
            answer = self.chain.invoke(question, config=config)
            
            # 검색 결과도 별도로 가져오기 (메타데이터용)
            docs = self.similarity_service.retriever.invoke(question)
            
            processing_time = time.time() - start_time
            
            result = {
                "answer": answer,
                "question": question,
                "sources": [doc.metadata for doc in docs],
                "context_docs": len(docs),
                "metadata": {
                    "model": self.config.model_name,
                    "search_type": self.config.search_type,
                    "processing_time": round(processing_time, 2),
                    "language": self.config.language,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            logger.info(f"RAG query completed: '{question}' -> {len(docs)} docs, {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"RAG chain failed: {e}")
            raise ProcessingError(f"RAG chain failed: {e}")
    
    async def ainvoke(self, question: str, **kwargs) -> Dict[str, Any]:
        """RAG 체인 비동기 실행"""
        start_time = time.time()
        
        try:
            config = {"callbacks": get_langfuse_callbacks()}
            
            # 비동기 체인 실행
            answer = await self.chain.ainvoke(question, config=config)
            
            # 검색 결과 가져오기
            docs = await self.similarity_service.retriever.ainvoke(question)
            
            processing_time = time.time() - start_time
            
            result = {
                "answer": answer,
                "question": question,
                "sources": [doc.metadata for doc in docs],
                "context_docs": len(docs),
                "metadata": {
                    "model": self.config.model_name,
                    "search_type": self.config.search_type,
                    "processing_time": round(processing_time, 2),
                    "language": self.config.language,
                    "execution_type": "async"
                }
            }
            
            logger.info(f"Async RAG query completed: '{question}'")
            return result
            
        except Exception as e:
            logger.error(f"Async RAG chain failed: {e}")
            raise ProcessingError(f"Async RAG chain failed: {e}")
    
    def stream(self, question: str, **kwargs):
        """RAG 체인 스트리밍 실행"""
        try:
            config = {"callbacks": get_langfuse_callbacks()}
            
            # 체인 스트리밍
            for chunk in self.chain.stream(question, config=config):
                yield chunk
                
        except Exception as e:
            logger.error(f"RAG streaming failed: {e}")
            raise ProcessingError(f"RAG streaming failed: {e}")
    
    def batch(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """RAG 체인 배치 실행"""
        try:
            config = {"callbacks": get_langfuse_callbacks()}
            
            # 배치 실행
            answers = self.chain.batch(questions, config=config)
            
            results = []
            for question, answer in zip(questions, answers):
                # 각 질문에 대한 검색 결과 가져오기
                docs = self.similarity_service.retriever.invoke(question)
                
                result = {
                    "answer": answer,
                    "question": question,
                    "sources": [doc.metadata for doc in docs],
                    "context_docs": len(docs),
                    "metadata": {
                        "model": self.config.model_name,
                        "search_type": self.config.search_type,
                        "language": self.config.language,
                        "execution_type": "batch"
                    }
                }
                results.append(result)
            
            logger.info(f"Batch RAG completed: {len(questions)} questions")
            return results
            
        except Exception as e:
            logger.error(f"Batch RAG failed: {e}")
            raise ProcessingError(f"Batch RAG failed: {e}")


def create_rag_chain(
    config_manager: ConfigurationManager = None,
    rag_config: RAGConfig = None
) -> RAGChain:
    """
    RAG 체인을 생성하는 팩토리 함수
    
    Args:
        config_manager: 설정 관리자 (없으면 기본 생성)
        rag_config: RAG 설정 (없으면 기본 생성)
        
    Returns:
        구성된 RAG 체인
    """
    try:
        # 설정 관리자가 없으면 기본 생성
        if config_manager is None:
            from src.services.config_manager import create_configuration_manager
            config_manager = create_configuration_manager()
        
        # 서비스 팩토리로 검색 서비스 생성
        service_factory = create_service_factory(config_manager)
        similarity_service = service_factory.get_similarity_search_service()
        
        # RAG 설정이 없으면 기본 생성
        if rag_config is None:
            rag_config = RAGConfig()
        
        # RAG 체인 생성
        rag_chain = RAGChain(similarity_service, rag_config)
        
        logger.info("RAG Chain created successfully")
        return rag_chain
        
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {e}")
        raise ProcessingError(f"Failed to create RAG chain: {e}") 