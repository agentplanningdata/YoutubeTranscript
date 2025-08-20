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
- [ ] **테스트**: `test_youtube_api_client.py::test_api_client_initialization` - API 클라이언트 초기화
- [ ] **테스트**: `test_youtube_api_client.py::test_invalid_api_key_raises_exception` - 잘못된 API 키 예외 처리
- [ ] **테스트**: `test_youtube_api_client.py::test_api_quota_exceeded_handling` - API 할당량 초과 처리

#### ✅ 2.2 채널 검색 기능
- [ ] **테스트**: `test_channel_search.py::test_search_channel_by_name` - 채널명으로 검색
- [ ] **테스트**: `test_channel_search.py::test_search_returns_channel_metadata` - 채널 메타데이터 반환 확인
- [ ] **테스트**: `test_channel_search.py::test_search_nonexistent_channel` - 존재하지 않는 채널 처리
- [ ] **테스트**: `test_channel_search.py::test_search_result_pagination` - 검색 결과 페이징 처리
- [ ] **리팩토링**: 검색 결과 데이터 모델 추상화

#### ✅ 2.3 영상 정보 조회
- [ ] **테스트**: `test_video_info.py::test_get_latest_video_from_channel` - 채널의 최신 영상 조회
- [ ] **테스트**: `test_video_info.py::test_get_video_metadata` - 영상 메타데이터 추출
- [ ] **테스트**: `test_video_info.py::test_get_video_duration` - 영상 길이 정보 추출
- [ ] **테스트**: `test_video_info.py::test_channel_with_no_videos` - 영상이 없는 채널 처리

### Phase 3: 자막 다운로드 및 처리

#### ✅ 3.1 자막 다운로드
- [ ] **테스트**: `test_transcript_downloader.py::test_download_transcript_korean` - 한국어 자막 다운로드
- [ ] **테스트**: `test_transcript_downloader.py::test_download_transcript_auto_generated` - 자동 생성 자막 다운로드
- [ ] **테스트**: `test_transcript_downloader.py::test_transcript_not_available` - 자막 없는 영상 처리
- [ ] **테스트**: `test_transcript_downloader.py::test_transcript_with_timestamps` - 타임스탬프 메타데이터 보존

#### ✅ 3.2 자막 전처리 및 청킹
- [ ] **테스트**: `test_transcript_processor.py::test_clean_transcript_text` - 자막 텍스트 정제
- [ ] **테스트**: `test_transcript_processor.py::test_chunk_transcript_by_sentences` - 문장 단위 청킹
- [ ] **테스트**: `test_transcript_processor.py::test_chunk_with_overlap` - 오버랩을 포함한 청킹
- [ ] **테스트**: `test_transcript_processor.py::test_preserve_timestamp_metadata` - 타임스탬프 메타데이터 유지
- [ ] **리팩토링**: 청킹 전략 추상화 및 설정 가능하게 구조 개선

### Phase 4: 벡터 데이터베이스 (ChromaDB) 통합

#### ✅ 4.1 ChromaDB 연결 및 설정
- [ ] **테스트**: `test_vector_db.py::test_chromadb_connection` - ChromaDB 연결 테스트
- [ ] **테스트**: `test_vector_db.py::test_create_collection` - 컬렉션 생성
- [ ] **테스트**: `test_vector_db.py::test_collection_already_exists` - 기존 컬렉션 처리

#### ✅ 4.2 임베딩 및 벡터 저장
- [ ] **테스트**: `test_embeddings.py::test_generate_embeddings_openai` - OpenAI 임베딩 생성
- [ ] **테스트**: `test_embeddings.py::test_store_transcript_chunks` - 자막 청크 벡터 저장
- [ ] **테스트**: `test_embeddings.py::test_store_with_metadata` - 메타데이터와 함께 저장
- [ ] **테스트**: `test_embeddings.py::test_duplicate_chunk_handling` - 중복 청크 처리

#### ✅ 4.3 유사도 검색
- [ ] **테스트**: `test_similarity_search.py::test_search_by_query` - 쿼리 기반 유사도 검색
- [ ] **테스트**: `test_similarity_search.py::test_search_with_score_threshold` - 점수 임계값 필터링
- [ ] **테스트**: `test_similarity_search.py::test_search_top_k_results` - Top-K 결과 제한
- [ ] **테스트**: `test_similarity_search.py::test_search_with_video_filter` - 특정 영상 필터링
- [ ] **리팩토링**: 검색 파라미터 설정 클래스 도입

### Phase 5: PostgreSQL 데이터베이스 연동

#### ✅ 5.1 데이터베이스 스키마
- [ ] **테스트**: `test_database_schema.py::test_channel_table_creation` - 채널 테이블 생성
- [ ] **테스트**: `test_database_schema.py::test_video_table_creation` - 영상 테이블 생성
- [ ] **테스트**: `test_database_schema.py::test_transcript_table_creation` - 자막 테이블 생성
- [ ] **테스트**: `test_database_schema.py::test_foreign_key_constraints` - 외래키 제약조건 확인

#### ✅ 5.2 채널 관리 (CRUD)
- [ ] **테스트**: `test_channel_repository.py::test_add_channel_to_watchlist` - 와치리스트에 채널 추가
- [ ] **테스트**: `test_channel_repository.py::test_get_watchlist_channels` - 와치리스트 채널 조회
- [ ] **테스트**: `test_channel_repository.py::test_remove_channel_from_watchlist` - 와치리스트에서 채널 제거
- [ ] **테스트**: `test_channel_repository.py::test_update_channel_metadata` - 채널 메타데이터 업데이트

#### ✅ 5.3 영상 관리 (CRUD)
- [ ] **테스트**: `test_video_repository.py::test_save_video_info` - 영상 정보 저장
- [ ] **테스트**: `test_video_repository.py::test_get_latest_videos_by_channel` - 채널별 최신 영상 조회
- [ ] **테스트**: `test_video_repository.py::test_mark_video_as_processed` - 영상 처리 완료 마킹
- [ ] **테스트**: `test_video_repository.py::test_get_unprocessed_videos` - 미처리 영상 목록 조회

### Phase 6: LangGraph 에이전트 시스템

#### ✅ 6.1 기본 에이전트 구조
- [ ] **테스트**: `test_base_agent.py::test_agent_initialization` - 기본 에이전트 초기화
- [ ] **테스트**: `test_base_agent.py::test_agent_state_management` - 상태 관리 테스트
- [ ] **테스트**: `test_base_agent.py::test_agent_error_handling` - 에러 처리 테스트

#### ✅ 6.2 요약 에이전트
- [ ] **테스트**: `test_summary_agent.py::test_generate_summary_from_transcript` - 자막 기반 요약 생성
- [ ] **테스트**: `test_summary_agent.py::test_summary_length_control` - 요약 길이 제어
- [ ] **테스트**: `test_summary_agent.py::test_summary_with_key_timestamps` - 주요 타임스탬프 포함 요약
- [ ] **테스트**: `test_summary_agent.py::test_empty_transcript_handling` - 빈 자막 처리

#### ✅ 6.3 예상 질문 생성 에이전트
- [ ] **테스트**: `test_question_agent.py::test_generate_questions_from_summary` - 요약 기반 예상 질문 생성
- [ ] **테스트**: `test_question_agent.py::test_question_difficulty_levels` - 질문 난이도 조절
- [ ] **테스트**: `test_question_agent.py::test_question_categories` - 질문 카테고리 분류
- [ ] **리팩토링**: 질문 생성 템플릿 모듈화

#### ✅ 6.4 RAG 에이전트
- [ ] **테스트**: `test_rag_agent.py::test_answer_generation_with_context` - 컨텍스트 기반 답변 생성
- [ ] **테스트**: `test_rag_agent.py::test_source_citation_with_timestamps` - 출처 인용 및 타임스탬프
- [ ] **테스트**: `test_rag_agent.py::test_insufficient_context_handling` - 불충분한 컨텍스트 처리
- [ ] **테스트**: `test_rag_agent.py::test_confidence_score_calculation` - 신뢰도 점수 계산

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

### Phase 8: 채널 모니터링 시스템

#### ✅ 8.1 즉시 처리 시스템 (채널 추가 시)
- [ ] **테스트**: `test_immediate_processing.py::test_process_latest_video_on_channel_add` - 채널 추가 시 최신 영상 처리
- [ ] **테스트**: `test_immediate_processing.py::test_skip_if_video_already_processed` - 이미 처리된 영상 스킵
- [ ] **테스트**: `test_immediate_processing.py::test_processing_pipeline_completion` - 전체 처리 파이프라인 완료 확인

#### ✅ 8.2 주기적 모니터링
- [ ] **테스트**: `test_periodic_monitoring.py::test_scheduled_channel_check` - 스케줄된 채널 확인
- [ ] **테스트**: `test_periodic_monitoring.py::test_new_video_detection` - 새 영상 탐지
- [ ] **테스트**: `test_periodic_monitoring.py::test_monitoring_interval_configuration` - 모니터링 간격 설정
- [ ] **테스트**: `test_periodic_monitoring.py::test_error_recovery_in_monitoring` - 모니터링 중 에러 복구

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
- [ ] **테스트**: `test_qa_api.py::test_get_suggested_questions_endpoint` - 예상 질문 조회 엔드포인트
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
- [ ] **테스트**: `test_gradio_video_ui.py::test_suggested_questions_display` - 예상 질문 표시
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

**현재 진행 상황**: 6/83 테스트 완료 (7.2%)

---

이 계획을 따라 순서대로 진행하면서, 각 단계마다 Red → Green → Refactor 사이클을 철저히 지키세요. 