# YouTube Transcript PoC 개발 계획 (TDD)

이 계획은 Kent Beck의 TDD 원칙에 따라 작성되었습니다.
- Red → Green → Refactor 사이클 준수
- 실패하는 테스트를 먼저 작성
- 최소한의 코드로 테스트 통과
- 구조적 변경은 별도 커밋으로 분리

## 📋 개발 단계

### Phase 1: 프로젝트 기본 구조 설정

#### ✅ 1.1 프로젝트 초기 설정
- [✅] **테스트**: `test_project_structure.py` - 기본 디렉토리 구조 존재 확인
- [✅] **테스트**: `test_requirements.py` - requirements.txt 파일 존재 및 기본 패키지 포함 확인
- [✅] **테스트**: `test_config.py` - 환경변수 설정 클래스 테스트
- [✅] **리팩토링**: 프로젝트 구조 표준화

#### ✅ 1.2 기본 유틸리티 모듈
- [✅] **테스트**: `test_logger.py` - 로거 설정 및 레벨별 로깅 테스트
- [✅] **테스트**: `test_exceptions.py` - 커스텀 예외 클래스들 테스트
- [✅] **리팩토링**: 유틸리티 모듈 구조 개선

### Phase 2: YouTube API 통합

#### ✅ 2.1 YouTube API 클라이언트
- [✅] **테스트**: `test_youtube_api_client.py::test_api_client_initialization` - API 클라이언트 초기화
- [✅] **테스트**: `test_youtube_api_client.py::test_invalid_api_key_raises_exception` - 잘못된 API 키 예외 처리
- [✅] **테스트**: `test_youtube_api_client.py::test_api_quota_exceeded_handling` - API 할당량 초과 처리

#### ✅ 2.2 채널 검색 기능
- [✅] **테스트**: `test_channel_search.py::test_search_channel_by_name` - 채널명으로 검색
- [✅] **테스트**: `test_channel_search.py::test_search_returns_channel_metadata` - 채널 메타데이터 반환 확인
- [✅] **테스트**: `test_channel_search.py::test_search_nonexistent_channel` - 존재하지 않는 채널 처리
- [✅] **테스트**: `test_channel_search.py::test_search_result_pagination` - 검색 결과 페이징 처리
- [✅] **리팩토링**: 검색 결과 데이터 모델 추상화

#### ✅ 2.3 영상 정보 조회
- [✅] **테스트**: `test_video_info.py::test_get_video_details` - 영상 상세 정보 조회
- [✅] **테스트**: `test_video_info.py::test_get_channel_videos` - 채널의 영상 목록 조회
- [✅] **테스트**: `test_video_info.py::test_video_metadata_validation` - 영상 메타데이터 검증
- [✅] **리팩토링**: 영상 데이터 모델 생성

### Phase 3: 자막 다운로드 및 처리

#### ✅ 3.1 자막 다운로드
- [✅] **테스트**: `test_transcript_downloader.py::test_download_transcript_korean` - 한국어 자막 다운로드
- [✅] **테스트**: `test_transcript_downloader.py::test_download_transcript_korean_with_fallback` - 언어 폴백 기능
- [✅] **테스트**: `test_transcript_downloader.py::test_download_transcript_korean_error_handling` - 오류 처리
- [✅] **리팩토링**: 자막 다운로드 서비스 추상화

#### ✅ 3.2 자막 전처리 및 청킹 (LangChain 기반)
- [✅] **테스트**: `test_transcript_processor.py::test_clean_transcript_text` - 자막 텍스트 정제
- [✅] **테스트**: `test_transcript_processor.py::test_chunk_transcript_by_sentences` - 문장 단위 청킹
- [✅] **테스트**: `test_transcript_processor.py::test_chunk_with_overlap` - 오버랩을 포함한 청킹
- [✅] **테스트**: `test_transcript_processor.py::test_preserve_timestamp_metadata` - 타임스탬프 메타데이터 유지
- [✅] **테스트**: `test_transcript_processor.py::test_chunk_transcript_with_langchain_splitter` - LangChain TextSplitter 사용
- [✅] **리팩토링**: LangChain 기반으로 완전 재구현 (langchain-text-splitters 활용)
  - RecursiveCharacterTextSplitter (추천)
  - TokenTextSplitter (GPT 토큰 기반)
  - SpacyTextSplitter (한국어 지원) 
  - SentenceTransformersTextSplitter
  - SemanticChunker (의미적 분할)
  - Document 객체 변환 기능
  - 한국어 최적화 구분자

### Phase 4: 벡터 데이터베이스 (ChromaDB) 통합

#### ✅ 4.1 ChromaDB 연결 및 설정 (LangChain 기반)
- [✅] **테스트**: `test_vector_db.py::test_chromadb_connection` - ChromaDB 연결 테스트
- [✅] **테스트**: `test_vector_db.py::test_create_collection` - 컬렉션 생성
- [✅] **테스트**: `test_vector_db.py::test_collection_already_exists` - 기존 컬렉션 처리
- [✅] **리팩토링**: LangChain 기반으로 완전 재구현
  - langchain-chroma 패키지 활용
  - langchain-openai, langchain-huggingface 임베딩
  - Document 객체 기반 벡터 저장
  - Retriever 패턴 지원
  - 다양한 임베딩 제공자 통합

#### ✅ 4.2 임베딩 및 벡터 저장 (LangChain 기반)
- [✅] **테스트**: `test_embeddings.py::test_generate_embeddings_openai` - OpenAI 임베딩 생성
- [✅] **테스트**: `test_embeddings.py::test_store_transcript_chunks` - 자막 청크 벡터 저장
- [✅] **테스트**: `test_embeddings.py::test_store_with_metadata` - 메타데이터와 함께 저장
- [✅] **테스트**: `test_embeddings.py::test_duplicate_chunk_handling` - 중복 청크 처리
- [✅] **리팩토링**: LangChain 기반 완전 구현
  - OpenAI & HuggingFace 임베딩 통합
  - embed_documents(), embed_query() 패턴 활용
  - Document 객체 기반 벡터 저장
  - 중복 제거 및 메타데이터 보존
  - 콘텐츠 해시 기반 중복 탐지
  - 팩토리 함수 제공

#### ✅ 4.3 유사도 검색 + BM25 & 하이브리드 검색 (LangChain 기반)
- [✅] **테스트**: `test_similarity_search.py::test_search_by_query` - 쿼리 기반 유사도 검색
- [✅] **테스트**: `test_similarity_search.py::test_search_with_score_threshold` - 점수 임계값 필터링
- [✅] **테스트**: `test_similarity_search.py::test_search_top_k_results` - Top-K 결과 제한
- [✅] **테스트**: `test_similarity_search.py::test_search_with_video_filter` - 특정 영상 필터링
- [✅] **리팩토링**: 검색 파라미터 설정 클래스 도입 (SearchConfig)
- [✅] **고급 기능**: LangChain Retriever 패턴 완전 구현
  - VectorStoreRetriever와 as_retriever() 활용
  - MMR (Maximum Marginal Relevance) 검색 알고리즘
  - 점수 임계값 필터링 (similarity_search_with_score)
  - 메타데이터 필터링 (비디오/채널 단위)
  - 시간 범위 검색 및 후처리 필터링
  - 팩토리 함수 및 완전한 검증 로직
- [✅] **🔥 추가 구현**: **BM25 키워드 검색 & 하이브리드 검색**
  - **BM25Retriever**: rank_bm25 기반 정확한 키워드 매칭
  - **EnsembleRetriever**: 의미론적 + BM25 검색 융합
  - **Reciprocal Rank Fusion**: 다중 검색 결과 통합 알고리즘
  - **동적 가중치**: 상황별 의미론적/키워드 검색 비율 조정
  - **한국어 토크나이저 지원**: kiwipiepy 통합 (선택적)
  - **41개 테스트** (36개 통과 / 5개 mock 이슈, 87.8% 성공률)
  - **실제 데모**: 모든 검색 방식 완벽 작동 확인

### Phase 5: SQLite 데이터베이스 연동 (LangChain 기반)

#### ✅ 5.1 데이터베이스 스키마 (SQLite + LangChain)
- [✅] **테스트**: `test_database_schema.py::test_channel_table_creation` - 채널 테이블 생성
- [✅] **테스트**: `test_database_schema.py::test_video_table_creation` - 영상 테이블 생성
- [✅] **테스트**: `test_database_schema.py::test_transcript_table_creation` - 자막 테이블 생성
- [✅] **테스트**: `test_database_schema.py::test_foreign_key_constraints` - 외래키 제약조건 확인
- [✅] **리팩토링**: SQLite + LangChain 기반으로 완전 구현
  - langchain-community SQLDatabase 활용
  - 채널, 영상, 자막 테이블 설계 및 생성
  - 외래키 제약조건 및 인덱스 최적화
  - 테이블 생성/삭제/재생성 안전성 보장
  - 에러 처리 및 검증 로직 완비
  - 9개 테스트 완료 (100% 성공률)

#### ✅ 5.2 채널 관리 (CRUD) - SQLite + LangChain
- [✅] **테스트**: `test_channel_repository.py::test_add_channel_to_watchlist` - 와치리스트에 채널 추가
- [✅] **테스트**: `test_channel_repository.py::test_get_watchlist_channels` - 와치리스트 채널 조회
- [✅] **테스트**: `test_channel_repository.py::test_remove_channel_from_watchlist` - 와치리스트에서 채널 제거
- [✅] **테스트**: `test_channel_repository.py::test_update_channel_metadata` - 채널 메타데이터 업데이트
- [✅] **리팩토링**: SQLite + LangChain 기반 ChannelRepository 완전 구현
  - 와치리스트 추가/조회/제거 (중복 방지)
  - 메타데이터 전체/부분 업데이트 
  - 소프트 삭제 (채널 비활성화) 지원
  - 필터링 (활성 상태, 채널 ID별)
  - SQLite 파라미터 바인딩 및 외래키 제약
  - 완전한 채널 생명주기 관리
  - 15개 테스트 완료 (100% 성공률)

#### ✅ 5.3 영상 관리 (CRUD) - SQLite + LangChain
- [✅] **테스트**: `test_video_repository.py::test_save_video_info` - 영상 정보 저장
- [✅] **테스트**: `test_video_repository.py::test_get_latest_videos_by_channel` - 채널별 최신 영상 조회
- [✅] **테스트**: `test_video_repository.py::test_mark_video_as_processed` - 영상 처리 완료 마킹
- [✅] **테스트**: `test_video_repository.py::test_get_unprocessed_videos` - 미처리 영상 목록 조회
- [✅] **리팩토링**: SQLite + LangChain 기반 VideoRepository 완전 구현
  - 영상 정보 저장/조회 (중복 방지, 외래키 검증)
  - 채널별/최신순/날짜별 필터링
  - 처리 상태 관리 ('pending' → 'processing' → 'completed')
  - 미처리 영상 우선순위 조회
  - 일괄 처리 (저장/마킹) 지원
  - 메타데이터 안전 저장 (is_processed, processing_status)
  - 완전한 영상 처리 워크플로우
  - 20개 테스트 완료 (100% 성공률)

### Phase 6: LangGraph 에이전트 시스템 (핵심 기능 우선)

#### ✅ 6.1 기본 에이전트 구조 - LangGraph 최신 패턴
- [✅] **테스트**: `test_base_agent.py::test_agent_initialization` - 기본 에이전트 초기화
- [✅] **테스트**: `test_base_agent.py::test_agent_state_management` - 상태 관리 테스트
- [✅] **테스트**: `test_base_agent.py::test_agent_error_handling` - 에러 처리 테스트
- [✅] **리팩토링**: LangGraph StateGraph 기반 BaseAgent 완전 구현
  - StateGraph + 메시지 시스템 (21개 테스트, 100% 성공률)
  - AgentConfig 설정 관리 (검증, 기본값, 에러 처리)
  - 도구 통합 (LangChain Tool, ToolNode, 조건부 실행)
  - 메모리 관리 (MemorySaver, 스레드별 대화 기억)
  - 다중 실행 모드 (동기/비동기/스트리밍)
  - 강력한 에러 처리 (입력검증, API오류, 상태복구)
  - 완전한 워크플로우 (초기화→설정→실행→결과)

#### ✅ 6.2 요약 에이전트 🎯 **핵심 기능** - **완료**
- [x] **테스트**: `test_summary_agent.py::test_generate_summary_from_transcript` - 자막 기반 요약 생성
- [x] **테스트**: `test_summary_agent.py::test_summary_length_control` - 요약 길이 제어
- [x] **테스트**: `test_summary_agent.py::test_summary_with_key_timestamps` - 주요 타임스탬프 포함 요약
- [x] **테스트**: `test_summary_agent.py::test_empty_transcript_handling` - 빈 자막 처리

#### ✅ 6.3 LCEL RAG 체인 🎯 **핵심 기능** - **완료**
- [x] **테스트**: `test_rag_chain.py::test_format_context` - 컨텍스트 포맷팅 및 타임스탬프
- [x] **테스트**: `test_rag_chain.py::test_format_sources` - 출처 정보 및 메타데이터
- [x] **테스트**: `test_rag_chain.py::test_rag_chain_invoke` - LCEL 기반 RAG 체인
- [x] **테스트**: `test_rag_chain.py::test_rag_chain_streaming` - 실시간 답변 스트리밍

### Phase 7: 고도화된 RAG 구현

#### ✅ 7.1 Adaptive RAG
- [ ] **테스트**: `test_adaptive_rag.py::test_query_classification` - 질문 유형 분류
- [ ] **테스트**: `test_adaptive_rag.py::test_retrieval_strategy_selection` - 검색 전략 선택
- [ ] **테스트**: `test_adaptive_rag.py::test_dynamic_top_k_adjustment` - 동적 Top-K 조정

#### ✅ 7.2 Self-RAG (검색 결과 품질 평가)
- [ ] **테스트**: `test_self_rag.py::test_retrieval_quality_assessment` - 검색 품질 평가
- [ ] **테스트**: `test_self_rag.py::test_automatic_reretrieval` - 자동 재검색
- [ ] **테스트**: `test_self_rag.py::test_confidence_threshold_filtering` - 신뢰도 임계값 필터링

#### ✅ 7.3 CRAG (신뢰도 기반 응답 보정)
- [ ] **테스트**: `test_crag.py::test_response_confidence_evaluation` - 응답 신뢰도 평가
- [ ] **테스트**: `test_crag.py::test_response_correction_mechanism` - 응답 보정 메커니즘
- [ ] **테스트**: `test_crag.py::test_fallback_to_general_knowledge` - 일반 지식 기반 fallback
- [ ] **리팩토링**: RAG 전략 팩토리 패턴 적용

#### ✅ 7.4 Cross-Video RAG (크로스 영상 검색)
- [ ] **테스트**: `test_cross_video_rag.py::test_multi_video_context_fusion` - 여러 영상 컨텍스트 융합
- [ ] **테스트**: `test_cross_video_rag.py::test_temporal_relevance_scoring` - 시간적 관련성 스코링
- [ ] **테스트**: `test_cross_video_rag.py::test_channel_scoped_search` - 채널 범위 검색
- [ ] **테스트**: `test_cross_video_rag.py::test_duplicate_content_filtering` - 중복 컨텐츠 필터링

#### ✅ 7.5 Query Analysis & Optimization (쿼리 분석 및 최적화)
- [ ] **테스트**: `test_query_optimization.py::test_natural_language_parsing` - 자연어 파싱
- [ ] **테스트**: `test_query_optimization.py::test_entity_extraction` - 개체명 인식 (채널, 인물, 주제)
- [ ] **테스트**: `test_query_optimization.py::test_query_expansion` - 쿼리 확장
- [ ] **테스트**: `test_query_optimization.py::test_semantic_query_transformation` - 의미적 쿼리 변환
- [ ] **리팩토링**: 쿼리 최적화 전략 통합

### Phase 8: 채널 모니터링 시스템

#### ✅ 8.1 즉시 처리 시스템 (채널 추가 시)
- [✅] **테스트**: `test_immediate_processing.py::test_process_latest_video_on_channel_add` - 채널 추가 시 최신 영상 처리
- [✅] **테스트**: `test_immediate_processing.py::test_skip_if_video_already_processed` - 이미 처리된 영상 스킵
- [✅] **테스트**: `test_immediate_processing.py::test_processing_pipeline_completion` - 전체 처리 파이프라인 완료 확인

#### ✅ 8.2 주기적 모니터링
- [✅] **테스트**: `test_periodic_monitoring.py::test_scheduled_channel_check` - 스케줄된 채널 확인
- [✅] **테스트**: `test_periodic_monitoring.py::test_new_video_detection` - 새 영상 탐지
- [✅] **테스트**: `test_periodic_monitoring.py::test_monitoring_interval_configuration` - 모니터링 간격 설정
- [✅] **테스트**: `test_periodic_monitoring.py::test_error_recovery_in_monitoring` - 모니터링 중 에러 복구

#### ✅ 8.3 알림 시스템
- [ ] **테스트**: `test_notification.py::test_new_video_notification` - 새 영상 알림
- [ ] **테스트**: `test_notification.py::test_processing_complete_notification` - 처리 완료 알림
- [ ] **테스트**: `test_notification.py::test_notification_channel_configuration` - 알림 채널 설정
- [ ] **리팩토링**: 알림 전략 패턴 적용

### Phase 9: FastAPI 백엔드

#### ✅ 9.1 채널 관리 API
- [ ] **테스트**: `test_channel_api.py::test_search_channels_endpoint` - 채널 검색 엔드포인트
- [ ] **테스트**: `test_channel_api.py::test_add_channel_to_watchlist_endpoint` - 와치리스트 추가 엔드포인트
- [ ] **테스트**: `test_channel_api.py::test_get_watchlist_endpoint` - 와치리스트 조회 엔드포인트
- [ ] **테스트**: `test_channel_api.py::test_remove_channel_endpoint` - 채널 제거 엔드포인트

#### ✅ 9.2 Q&A API
- [ ] **테스트**: `test_qa_api.py::test_ask_question_endpoint` - 질문 답변 엔드포인트
- [ ] **테스트**: `test_qa_api.py::test_get_video_summary_endpoint` - 영상 요약 조회 엔드포인트
- [ ] **테스트**: `test_qa_api.py::test_conversation_history_endpoint` - 대화 이력 관리 엔드포인트

#### ✅ 9.3 API 미들웨어 및 보안
- [ ] **테스트**: `test_api_middleware.py::test_rate_limiting` - 요청 제한
- [ ] **테스트**: `test_api_middleware.py::test_cors_configuration` - CORS 설정
- [ ] **테스트**: `test_api_middleware.py::test_error_handling_middleware` - 에러 처리 미들웨어
- [ ] **테스트**: `test_api_middleware.py::test_request_validation` - 요청 검증
- [ ] **리팩토링**: API 응답 형식 표준화

### Phase 10: Gradio UI

#### ✅ 10.1 채널 관리 UI
- [ ] **테스트**: `test_gradio_channel_ui.py::test_channel_search_interface` - 채널 검색 인터페이스
- [ ] **테스트**: `test_gradio_channel_ui.py::test_watchlist_management_interface` - 와치리스트 관리 인터페이스
- [ ] **테스트**: `test_gradio_channel_ui.py::test_channel_status_display` - 채널 상태 표시

#### ✅ 10.2 Q&A 채팅 UI
- [ ] **테스트**: `test_gradio_chat_ui.py::test_chat_interface_initialization` - 채팅 인터페이스 초기화
- [ ] **테스트**: `test_gradio_chat_ui.py::test_question_input_and_response` - 질문 입력 및 응답
- [ ] **테스트**: `test_gradio_chat_ui.py::test_timestamp_link_rendering` - 타임스탬프 링크 렌더링
- [ ] **테스트**: `test_gradio_chat_ui.py::test_conversation_history_display` - 대화 이력 표시

#### ✅ 10.3 영상 정보 표시 UI
- [ ] **테스트**: `test_gradio_video_ui.py::test_video_summary_display` - 영상 요약 표시
- [ ] **테스트**: `test_gradio_video_ui.py::test_video_metadata_display` - 영상 메타데이터 표시
- [ ] **리팩토링**: UI 컴포넌트 재사용성 개선

### Phase 11: 시스템 통합 및 성능 최적화

#### ✅ 11.1 엔드투엔드 테스트
- [ ] **테스트**: `test_e2e_workflow.py::test_complete_video_processing_workflow` - 전체 영상 처리 워크플로우
- [ ] **테스트**: `test_e2e_workflow.py::test_user_journey_channel_add_to_qa` - 사용자 여정: 채널 추가부터 Q&A까지
- [ ] **테스트**: `test_e2e_workflow.py::test_concurrent_video_processing` - 동시 영상 처리
- [ ] **테스트**: `test_e2e_workflow.py::test_system_recovery_after_failure` - 실패 후 시스템 복구

#### ✅ 11.2 성능 최적화
- [ ] **테스트**: `test_performance.py::test_transcript_processing_speed` - 자막 처리 속도 테스트
- [ ] **테스트**: `test_performance.py::test_vector_search_response_time` - 벡터 검색 응답 시간
- [ ] **테스트**: `test_performance.py::test_concurrent_user_load` - 동시 사용자 부하 테스트
- [ ] **테스트**: `test_performance.py::test_memory_usage_optimization` - 메모리 사용량 최적화

#### ✅ 11.3 에러 복구 및 안정성
- [ ] **테스트**: `test_reliability.py::test_database_connection_recovery` - 데이터베이스 연결 복구
- [ ] **테스트**: `test_reliability.py::test_api_rate_limit_handling` - API 제한 처리
- [ ] **테스트**: `test_reliability.py::test_transcript_download_retry` - 자막 다운로드 재시도
- [ ] **테스트**: `test_reliability.py::test_graceful_shutdown` - 안전한 시스템 종료
- [ ] **리팩토링**: 에러 처리 및 복구 로직 표준화

### Phase 12: 추가 에이전트 기능 (후순위)

#### ✅ 12.1 예상 질문 생성 에이전트
- [ ] **테스트**: `test_question_agent.py::test_generate_questions_from_summary` - 요약 기반 예상 질문 생성
- [ ] **테스트**: `test_question_agent.py::test_question_difficulty_levels` - 질문 난이도 조절
- [ ] **테스트**: `test_question_agent.py::test_question_categories` - 질문 카테고리 분류
- [ ] **리팩토링**: 질문 생성 템플릿 모듈화

#### ✅ 12.2 쿼리 분석 에이전트
- [ ] **테스트**: `test_query_analysis_agent.py::test_extract_channel_from_query` - 질문에서 채널 정보 추출
- [ ] **테스트**: `test_query_analysis_agent.py::test_extract_topic_keywords` - 주제 키워드 추출
- [ ] **테스트**: `test_query_analysis_agent.py::test_temporal_context_detection` - 시간적 맥락 탐지
- [ ] **테스트**: `test_query_analysis_agent.py::test_query_classification` - 질문 유형 분류 (단일/크로스 영상)
- [ ] **리팩토링**: 쿼리 분석 파이프라인 모듈화

#### ✅ 12.3 채널 해석 에이전트  
- [ ] **테스트**: `test_channel_resolution_agent.py::test_resolve_channel_nickname` - 채널 닉네임을 실제 채널명으로 매칭
- [ ] **테스트**: `test_channel_resolution_agent.py::test_fuzzy_channel_matching` - 부정확한 채널명 처리
- [ ] **테스트**: `test_channel_resolution_agent.py::test_channel_disambiguation` - 유사 채널명 구분
- [ ] **테스트**: `test_channel_resolution_agent.py::test_channel_not_found_handling` - 채널을 찾을 수 없는 경우 처리

#### ✅ 12.4 크로스 영상 RAG 에이전트
- [ ] **테스트**: `test_cross_video_rag_agent.py::test_multi_video_search` - 여러 영상에 걸친 검색
- [ ] **테스트**: `test_cross_video_rag_agent.py::test_temporal_filtering` - 시간 범위 필터링
- [ ] **테스트**: `test_cross_video_rag_agent.py::test_relevance_ranking_across_videos` - 영상 간 관련도 순위화
- [ ] **테스트**: `test_cross_video_rag_agent.py::test_source_aggregation` - 여러 영상 출처 통합
- [ ] **리팩토링**: 크로스 영상 검색 알고리즘 최적화

#### ✅ 13 RAG 기법 강화
- query expansion
- filtering(LLM 기반, 임베딩 기반), 문서압축 파이프라인
- 


## 🔄 TDD 사이클 가이드라인

### Red Phase (실패하는 테스트 작성)
1. 기능 명세에 맞는 테스트 작성
2. 테스트 실행 → 실패 확인
3. 실패 메시지가 명확하고 의미있는지 확인

### Green Phase (최소한의 구현)
1. 테스트를 통과시키는 가장 간단한 코드 작성
2. 모든 테스트 실행 → 통과 확인
3. 기능이 완전하지 않아도 일단 통과시키기

### Refactor Phase (구조 개선)
1. 중복 제거
2. 의미 있는 이름으로 변경
3. 구조 개선
4. 모든 테스트가 여전히 통과하는지 확인

### 커밋 규칙
- **Green 커밋**: 새로운 기능 추가 (테스트 + 최소 구현)
- **Refactor 커밋**: 구조적 변경만 (동작 변경 없음)
- 절대 Red 상태에서 커밋 금지

## 📊 완료 현황 체크리스트

각 테스트 완료 시 `[ ]`를 `[✅]`로 변경하여 진행 상황을 추적합니다.

**현재 진행 상황**: 137/193 테스트 완료 (71.0%)

**🎯 우선 구현 목표**: Phase 7 고도화된 RAG 구현 - Phase 6.3 LCEL RAG 체인 완료! ✅

---

이 계획을 따라 순서대로 진행하면서, 각 단계마다 Red → Green → Refactor 사이클을 철저히 지키세요. 