# YouTube Transcript PoC

## 📋 개요

YouTube 채널의 새 영상 업로드를 실시간으로 모니터링하고, 영상 자막을 기반으로 요약, 예상 질문 생성, 그리고 RAG 기반 Q&A 서비스를 제공하는 시스템입니다.

### 🎯 핵심 기능
- YouTube 채널 실시간 모니터링 및 알림
- 영상 자막 자동 다운로드 및 분석
- AI 기반 영상 요약 및 예상 질문 생성
- RAG 기반 Q&A (타임스탬프 링크 포함)

## 🏗️ 시스템 아키텍처

### 기술 스택
- **백엔드**: Python, FastAPI
- **UI**: Gradio
- **에이전트 프레임워크**: LangGraph
- **벡터 데이터베이스**: ChromaDB
- **데이터베이스**: PostgreSQL
- **YouTube API**: YouTube Data API v3, youtube_transcript_api
- **AI 모델**: OpenAI GPT-4/Claude 또는 로컬 LLM

### RAG 기술
- **CRAG**: 검색 결과 신뢰도 기반 응답 생성

## 🔄 시스템 플로우

```mermaid
graph TD
    A[사용자 채널 입력] --> B[YouTube API 채널 검색]
    B --> C[채널 선택 및 와치리스트 추가]
    C --> F[최신 영상 자막 다운로드]
    C --> D[주기적 모니터링 시작]
    D --> E{새 영상 발견?}
    E -->|Yes| F
    E -->|No| D
    F --> G[자막 청킹 및 벡터화]
    G --> H[ChromaDB 저장]
    H --> I[요약 에이전트]
    H --> J[예상질문 에이전트]
    I --> K[요약 완료]
    J --> L[예상질문 생성 완료]
    K --> M[사용자 알림]
    L --> M
    M --> N[사용자 질문 입력]
    N --> O[벡터 검색]
    O --> P[RAG 기반 답변 생성]
    P --> Q[타임스탬프 링크 제공]
```

## 🚀 주요 기능 상세

### 1. 채널 모니터링
- YouTube Data API를 통한 채널 검색 및 정보 조회
- 구독자 수, 채널 설명, 프로필 이미지 등 메타데이터 수집
- 와치리스트 관리 (추가/삭제/상태 관리)

### 2. 영상 자막 처리
- `youtube_transcript_api`를 통한 자동 자막 다운로드
- 다국어 자막 지원 (우선순위: 한국어 > 자동생성)
- 타임스탬프 메타데이터 보존

### 3. LangGraph 에이전트 시스템
```python
# 에이전트 구조 예시
- SummaryAgent: 영상 내용 요약
- QuestionAgent: 예상 질문 생성  
- RAGAgent: 검색 증강 생성
- HistoryManager: 대화 이력 관리
```

### 4. 고도화된 RAG 구현
- **Adaptive RAG**: 질문 유형별 검색 전략 최적화
- **Self-RAG**: 검색 결과 품질 자체 평가 및 재검색
- **CRAG**: 신뢰도 기반 응답 보정

### 5. 벡터 데이터베이스 (ChromaDB)
- 자막 청크 임베딩 저장
- 메타데이터: 타임스탬프, 영상 ID, 채널 정보
- 의미적 유사도 검색 최적화

## 📦 설치 및 실행

### 환경 요구사항
- Python 3.12
- PostgreSQL 13+
- Chrome/Chromium (Playwright용)

### 설치 방법
```bash
# 레포지토리 클론
git clone https://github.com/your-username/youtube-transcript-poc.git
cd youtube-transcript-poc

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일에 API 키 설정

# 데이터베이스 초기화
python scripts/init_db.py

# 서버 실행
python main.py
```

### 환경 변수 설정
```env
# YouTube API
YOUTUBE_API_KEY=your_youtube_api_key

# AI 모델 API
OPENAI_API_KEY=your_openai_api_key
# 또는
ANTHROPIC_API_KEY=your_anthropic_api_key

# 데이터베이스
DATABASE_URL=postgresql://user:password@localhost:5432/youtube_transcript

# 모니터링 설정
MONITORING_INTERVAL_MINUTES=5
```

## 🏃‍♂️ 사용 방법

### 1. 채널 추가
```bash
# CLI를 통한 채널 추가
python cli.py add-channel "채널명"

# 또는 웹 인터페이스에서 추가
```

### 2. 모니터링 시작
```bash
# 백그라운드 모니터링 시작
python monitor.py --daemon

# 특정 채널만 모니터링
python monitor.py --channel "채널ID"
```

### 3. Q&A 사용
```python
# API 사용 예시
import requests

response = requests.post('/api/chat', json={
    'question': '이 영상에서 가장 중요한 포인트가 뭔가요?',
    'video_id': 'VIDEO_ID'
})

print(response.json())
# {
#   'answer': '주요 포인트는...',
#   'sources': [
#     {
#       'text': '관련 자막 내용',
#       'timestamp': 120,
#       'url': 'https://youtube.com/watch?v=VIDEO_ID&t=120s'
#     }
#   ]
# }
```

## 🧪 테스트

### 테스트 실행
```bash
# 전체 테스트
pytest

# 특정 모듈 테스트
pytest tests/test_youtube_api.py

# 커버리지 포함
pytest --cov=src tests/
```

### TDD 개발 프로세스
1. 실패하는 테스트 작성
2. 최소한의 코드로 테스트 통과
3. 리팩토링 (구조적 변경)
4. 다음 기능 테스트 작성

## 📊 모니터링 및 로깅

### 로그 레벨
- `INFO`: 일반적인 시스템 동작
- `WARNING`: 주의가 필요한 상황
- `ERROR`: 오류 발생
- `DEBUG`: 상세 디버깅 정보

### 메트릭 수집
- 처리된 영상 수
- 평균 응답 시간
- RAG 검색 정확도
- API 호출 횟수 및 한도

## 🔧 설정 및 커스터마이징

### 청킹 전략
```python
CHUNK_CONFIG = {
    'size': 1000,          # 청크 크기 (토큰)
    'overlap': 200,        # 오버랩 크기
    'strategy': 'semantic' # semantic, fixed, sentence
}
```

### RAG 파라미터
```python
RAG_CONFIG = {
    'top_k': 5,           # 검색할 문서 수
    'score_threshold': 0.7, # 유사도 임계값
    'rerank': True,        # 재순위화 여부
    'adaptive_threshold': True # 적응적 임계값
}
```