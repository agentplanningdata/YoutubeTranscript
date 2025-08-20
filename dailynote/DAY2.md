# LangChain Day 2 학습 가이드

Day 2에서는 고급 프롬프트 엔지니어링 기법과 실용적인 시스템 구현 방법을 학습합니다.

## 📚 학습 목차

1. [Langfuse 프롬프트 관리](#1-langfuse-프롬프트-관리)
   - Langfuse 개요 (프롬프트 CMS, 버전 관리, 협업, 배포 관리)
   - 환경 설정
     - 콜백 핸들러 (`CallbackHandler()`)
     - 클라이언트 초기화 (`get_client()`, `auth_check()`)
   - 프롬프트 생성
     - 텍스트 프롬프트 (`create_prompt()`, `type="text"`, `labels`, `tags`)
     - 채팅 프롬프트 (`type="chat"`, `role`, `content` 구성)
     - 설정 옵션 (`config`, `model`, `temperature`, `max_tokens`)
   - 프롬프트 활용
     - 프롬프트 조회 (`get_prompt()`, 이름 기반 조회)
     - 변수 삽입 (`compile()`, 변수 값 전달)
     - LangChain 통합 (`get_langchain_prompt()`, 메타데이터 포함)
   - 버전 관리
     - 라벨 시스템 (환경별 배포: production, staging)
     - 태그 관리 (분류 및 검색)
     - 메타데이터 (`metadata` 추가 정보)
   - 체인 통합 (`PromptTemplate.from_template()`, 콜백 핸들러 연결)

2. [Few-shot 프롬프트 엔지니어링](#2-few-shot-프롬프트-엔지니어링)
   - Few-shot Learning 개념 (예시 기반 패턴 학습)
   - Few-shot 예제 구성
     - 예제 정의 (`examples` 딕셔너리 리스트)
     - 예제 템플릿 (`PromptTemplate`, `input_variables`, `template`)
     - Few-shot 템플릿 (`FewShotPromptTemplate`, `examples`, `example_prompt`, `prefix`, `suffix`)
   - 동적 Few-shot 예제 선택
     - 유사도 선택기 (`SemanticSimilarityExampleSelector`)
     - 임베딩 기반 선택 (`from_examples()`, `OpenAIEmbeddings`, `Chroma`)
     - 선택 파라미터 (`k` 개수, 유사도 기반 필터링)
     - 동적 프롬프트 (`example_selector` 활용)

3. [Chain-of-Thought (CoT) 프롬프트 엔지니어링](#3-chain-of-thought-cot-프롬프트-엔지니어링)
   - CoT 개념 (복잡한 문제의 단계별 사고 과정 명시)
   - 프롬프팅 기법 비교
     - Zero-shot (`zero_shot_template`, 직접 답변 요청)
     - One-shot/Few-shot (`one_shot_template`, 예시 포함 템플릿)
     - Chain of Thought (`cot_template`, 단계별 사고 유도)
   - 고급 CoT 기법
     - Self-Consistency (`self_consistency_template`, 다중 방법 해결)
     - Program-Aided Language (PAL)
       - 프로그래밍 접근 (`pal_template`, Python 코드 생성)
       - 단계별 계산 (`solve_problem()` 함수 구조)
     - Reflexion (`reflexion_template`)
       - 초기 답안 작성 (1단계)
       - 자체 평가 (2단계: 정확성 검토, 논리적 오류 확인)
       - 개선된 답안 (3단계: 평가 기반 개선)

4. [RunnableConfig & Fallback](#4-runnableconfig--fallback)
   - RunnableConfig 설정
     - 기본 설정 (`RunnableConfig`, `tags`, `metadata`)
     - 콜백 설정 (`callbacks`, 추적 핸들러)
     - 동시성 제어 (`max_concurrency`, 병렬 실행 제한)
     - 체인 실행 (`chain.invoke()`, config 매개변수)
   - Fallback 체인 구성
     - 메인/백업 체인 (`main_chain`, `fallback_chain`)
     - Fallback 설정 (`with_fallbacks()`, 대안 체인 리스트)
     - 자동 복구 (메인 체인 실패 시 백업 체인 실행)
   - 조건부 실행
     - 분기 체인 (`RunnableBranch`, 조건 함수)
     - 조건별 처리 (`lambda` 함수, 타입 기반 분기)
     - 기본 체인 (`default_chain`, 조건 불만족 시)
   - 오류 처리 및 복구 전략

5. [채팅 기록 관리](#5-채팅-기록-관리)
   - 메시지 전달 방식 (기본 대화 기록 관리)
     - 메시지 타입 (`HumanMessage`, `AIMessage`, `SystemMessage`)
     - 프롬프트 템플릿 (`ChatPromptTemplate.from_messages()`, `MessagesPlaceholder`)
     - 메시지 리스트 관리 (`messages` 리스트, 기록 추가)
   - RunnableWithMessageHistory (자동 메시지 기록)
     - 히스토리 래퍼 (`RunnableWithMessageHistory`)
     - 메시지 키 설정 (`input_messages_key`, `history_messages_key`)
     - 세션 관리 (`session_id` 기반 히스토리 분리)
   - 히스토리 저장소 구현
     - 기본 인터페이스 (`BaseChatMessageHistory`, `BaseModel`)
     - 메모리 기반 (`InMemoryHistory`)
       - 메시지 저장 (`messages`, `Field(default_factory=list)`)
       - 메시지 관리 (`add_messages()`, `clear()`)
     - SQLite 영구 저장 (`SQLiteChatMessageHistory`)
       - 데이터베이스 설정 (`sqlite3.connect()`, 테이블 생성)
       - 세션별 저장 (`session_id`, 메타데이터 포함)
   - 메시지 관리 기법
     - 메시지 트리밍 (`trim_messages()`)
       - 트리밍 전략 (`strategy="last"`, 최신 메시지 유지)
       - 토큰 제한 (`max_tokens`, `token_counter`)
       - 트리밍 히스토리 (`TrimmedInMemoryHistory`)
     - 대화 요약 저장 (`SummarizedInMemoryHistory`)
       - 요약 임계값 (`summary_threshold`, LLM 기반 요약)
       - 요약 프롬프트 (대화 압축, 세부사항 유지)
       - 메시지 재구성 (요약 + 최신 메시지)
     - 트리밍 + 요약 결합 (`TrimmedAndSummarizedHistory`)
       - 하이브리드 접근 (트리밍과 요약 동시 적용)
       - 통합 요약 (`combined_summary`, 요약 병합)
       - 최적화된 메모리 관리

6. [실전 RAG 시스템 구현](#6-실전-rag-시스템-구현)
   - 주택청약 FAQ 봇 구현 사례
   - 문서 전처리 파이프라인
     - QA 쌍 추출
       - 정규표현식 (`re.match()`, Q/A 패턴 매칭)
       - 추출 함수 (`extract_qa_pairs()`, 구조화된 데이터 생성)
     - 키워드 추출
       - 구조화 출력 (`with_structured_output()`, `BaseModel`, `Field`)
       - 키워드 체인 (`keyword_extractor`, LLM 기반 추출)
     - 문서 객체 생성 (`Document`, `page_content`, `metadata` 구성)
   - 고급 검색 기능
     - 메타데이터 필터링
       - 기본 필터 (`filter`, 키-값 매칭)
       - 비교 연산자 (`$eq`, `$ne`, `$in`, `$gt`, `$gte`, `$lt`, `$lte`)
       - 논리 연산자 (`$and`, `$or`, 복합 조건)
       - 문자열 연산 (`$contains`, 포함 관계)
     - 검색 전략
       - MMR 검색 (`search_type="mmr"`, `lambda_mult`, `fetch_k`)
       - 유사도 검색 (`search_type="similarity"`, `k` 매개변수)
       - 복합 조건 검색 (메타데이터 + 유사도)
   - 문서 관련성 평가
     - 관련성 모델 (`DocumentRelevance`, `Literal["yes", "no"]`)
     - 관련성 평가 체인 (`relevance_chain`, 컨텍스트-질문 매칭)
     - 평가 기준 (직접적 정보, 논리적 추론 가능성)
   - Gradio 챗봇 인터페이스
     - RAG 시스템 클래스 (`RAGSystem`)
       - 구성 요소 (`llm`, `eval_llm`, `retriever`)
       - 관련성 확인 (`_check_relevance()`, 문서 필터링)
       - 답변 생성 (`generate_answer()`, 소스 문서 포함)
     - 웹 인터페이스 (`gr.ChatInterface`)
       - 인터페이스 설정 (`title`, `description`, `examples`)
       - 실시간 채팅 (`fn` 함수, 메시지 히스토리)
       - 런칭 (`demo.launch()`, 웹 서버 시작)

---

## 1. Langfuse 프롬프트 관리

### 🎯 Langfuse 개요

Langfuse는 **프롬프트 CMS(Content Management System)** 기능을 제공하는 LLM 개발 플랫폼입니다.

**주요 기능:**
- **버전 관리**: 프롬프트의 모든 변경사항을 추적하고 롤백 가능
- **협업**: 팀원들과 함께 프롬프트를 편집하고 관리
- **배포 관리**: 라벨을 통해 코드 변경 없이 환경별 배포
- **성능 모니터링**: 프롬프트 버전별 성능 메트릭 비교

### 🔧 환경 설정

```python
from langfuse.langchain import CallbackHandler 
from langfuse import get_client

# LangChain 콜백 핸들러 생성
langfuse_handler = CallbackHandler()

# Langfuse 클라이언트 초기화
langfuse = get_client()
assert langfuse.auth_check()
```

### 📝 프롬프트 생성

#### 텍스트 프롬프트

```python
# 텍스트 프롬프트 생성
langfuse.create_prompt(
    name="movie-critic",
    type="text",          
    prompt="{{criticLevel}} 영화 평론가로서, {{movie}}를 어떻게 생각하시나요?",
    labels=["production"],
    tags=["movie", "qa", "text"],
    config={
        "model": "gpt-4.1-mini",
        "temperature": 0.7,
        "max_tokens": 500
    }
)
```

#### 챗 프롬프트

```python
# 챗 프롬프트 생성
langfuse.create_prompt(
    name="movie-critic-chat",
    type="chat",          
    prompt=[
        {
            "role": "system",
            "content": "당신은 {{criticLevel}} 영화 평론가입니다."
        },
        {
            "role": "user",
            "content": "영화 {{movie}}를 어떻게 생각하시나요?"
        }
    ],
    labels=["production"],
    tags=["movie", "qa", "chat"]
)
```

### 🔄 프롬프트 활용

```python
# 프롬프트 가져오기
prompt = langfuse.get_prompt("movie-critic")

# 변수 삽입
compiled_prompt = prompt.compile(criticLevel="전문가", movie="인셉션")

# LangChain과 통합
from langchain_core.prompts import PromptTemplate

langchain_prompt = PromptTemplate.from_template(
    prompt.get_langchain_prompt(),
    metadata={"langfuse_prompt": prompt}
)

# 모델과 연결
chain = langchain_prompt | model
response = chain.invoke(
    input={"criticLevel": "전문가", "movie": "인셉션"},
    config={"callbacks": [langfuse_handler]}
)
```

---

## 2. Few-shot 프롬프트 엔지니어링

### 🎯 Few-shot Learning 개념

Few-shot 프롬프트는 몇 개의 예시를 통해 모델이 패턴을 학습하고 비슷한 작업을 수행하도록 하는 기법입니다.

### 📊 Few-shot 예제 구성

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 예제 정의
examples = [
    {
        "question": "안녕하세요",
        "answer": "안녕하세요! 무엇을 도와드릴까요?"
    },
    {
        "question": "오늘 날씨가 어때요?",
        "answer": "죄송하지만 실시간 날씨 정보는 제공할 수 없습니다. 날씨 앱을 확인해주세요."
    }
]

# 예제 템플릿
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="질문: {question}\n답변: {answer}"
)

# Few-shot 프롬프트 템플릿
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="다음은 고객 서비스 대화의 예시입니다:",
    suffix="질문: {input}\n답변:",
    input_variables=["input"]
)
```

### 🔄 동적 Few-shot 예제 선택

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 임베딩 기반 예제 선택기
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2  # 가장 유사한 2개 예제 선택
)

# 동적 Few-shot 프롬프트
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="다음은 관련된 대화 예시입니다:",
    suffix="질문: {input}\n답변:",
    input_variables=["input"]
)
```

---

## 3. Chain-of-Thought (CoT) 프롬프트 엔지니어링

### 🧠 CoT 개념

Chain of Thought는 AI 모델이 복잡한 문제를 해결할 때 각 단계별 사고 과정을 명시적으로 보여주도록 하는 프롬프팅 기법입니다.

### 📈 프롬프팅 기법 비교

#### 1) Zero-shot 프롬프팅

```python
zero_shot_template = """
다음 문제를 해결하시오:

문제: {question}

답안:
"""
```

#### 2) One-shot/Few-shot 프롬프팅

```python
one_shot_template = """
다음은 수학 문제를 해결하는 예시입니다:

예시 문제: 한 학급에 30명의 학생이 있습니다. 이 중 40%가 남학생이라면, 여학생은 몇 명인가요?

예시 풀이:
1) 먼저 남학생 수를 계산합니다:
   - 전체 학생의 40% = 30 x 0.4 = 12명이 남학생

2) 여학생 수를 계산합니다:
   - 전체 학생 수 - 남학생 수 = 30 - 12 = 18명이 여학생

따라서 여학생은 18명입니다.

이제 아래 문제를 같은 방식으로 해결하시오:

새로운 문제: {question}

답안:
"""
```

#### 3) Chain of Thought (CoT) 프롬프팅

```python
cot_template = """
수학 문제를 해결하기 위해 다음 단계를 따라하세요:
문제: {question}

해결 전략: 백분율 계산 문제는 전체에서 부분을 구하는 문제입니다.

📝 1단계: 주어진 정보를 표로 정리하기
📊 2단계: 백분율을 소수로 변환하기
🧮 3단계: 각 단계별 계산하기
✅ 4단계: 답안 검증하기

최종 답안: ___
"""
```

### 🔍 고급 CoT 기법

#### Self-Consistency

```python
self_consistency_template = """
다음 문제를 세 가지 다른 방법으로 해결하시오:

문제: {question}

세 가지 풀이 방법:
1) 직접 계산 방법
2) 비율 활용 방법  
3) 단계별 분해 방법

각 방법의 답안을 제시하고, 결과가 일치하는지 확인하시오.
"""
```

#### Program-Aided Language (PAL)

```python
pal_template = """
다음 문제를 Python 프로그래밍 방식으로 해결하시오:

문제: {question}

def solve_problem():
    # 1. 변수 정의
    # 2. 계산 과정
    # 3. 결과 반환
    
답안:
"""
```

#### Reflexion

```python
reflexion_template = """
다음 문제에 대해 단계적으로 해결하여 초기 답안을 작성하고, 자체 평가 후 개선하시오:

문제: {question}

1단계: 초기 답안
---
[여기에 첫 번째 답안 작성]

2단계: 자체 평가
---
- 정확성 검토
- 논리적 오류 확인
- 개선이 필요한 부분 식별

3단계: 개선된 답안
---
[평가를 바탕으로 개선된 답안 작성]
"""
```

---

## 4. RunnableConfig & Fallback

### ⚙️ RunnableConfig 설정

RunnableConfig는 체인 실행 시 추가적인 설정을 제공하는 메커니즘입니다.

```python
from langchain_core.runnables import RunnableConfig

# 기본 설정
config = RunnableConfig(
    tags=["experiment", "test"],
    metadata={"version": "1.0", "user": "admin"},
    callbacks=[langfuse_handler],
    max_concurrency=3
)

# 체인 실행 시 설정 적용
response = chain.invoke(input_data, config=config)
```

### 🛡️ Fallback 체인 구성

모델 실패나 오류 상황에 대비한 대안 체인을 구성할 수 있습니다.

```python
from langchain_core.runnables import RunnableWithFallbacks

# 메인 체인과 백업 체인 정의
main_chain = prompt | primary_llm | parser
fallback_chain = simple_prompt | backup_llm | simple_parser

# Fallback 체인 구성
chain_with_fallback = main_chain.with_fallbacks([fallback_chain])

# 실행 - 메인 체인 실패 시 자동으로 백업 체인 실행
result = chain_with_fallback.invoke(input_data)
```

### 🔄 조건부 실행

```python
from langchain_core.runnables import RunnableBranch

# 조건부 체인 실행
conditional_chain = RunnableBranch(
    (lambda x: x["type"] == "question", qa_chain),
    (lambda x: x["type"] == "translation", translation_chain),
    default_chain  # 기본 체인
)
```

---

## 5. 채팅 기록 관리

### 💬 메시지 전달 방식

기본적인 대화 기록 관리 방법입니다.

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="messages"),
])

# 메시지 목록
messages = [
    HumanMessage(content="안녕하세요. 제 이름은 홍길동입니다."),
    AIMessage(content="안녕하세요! 어떻게 도와드릴까요?"),
]

# 체인 실행
chain = prompt | llm
response = chain.invoke({
    "messages": messages + [HumanMessage(content="제 이름을 기억하나요?")]
})
```

### 🗄️ RunnableWithMessageHistory

자동 메시지 기록 관리를 위한 고급 기능입니다.

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# 메모리 기반 히스토리 구현
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    
    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)
    
    def clear(self) -> None:
        self.messages = []

# 세션 저장소
store = {}

def get_session_history(session_id: str) -> InMemoryHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# 히스토리 관리 체인
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

### 🗃️ SQLite 영구 저장소

```python
import sqlite3
from langchain_core.chat_history import BaseChatMessageHistory

class SQLiteChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, db_path: str = "chat_history.db"):
        self.session_id = session_id
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                message_type TEXT,
                content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
```

### ✂️ 메시지 관리 기법

#### 메시지 트리밍

```python
from langchain_core.messages import trim_messages

class TrimmedInMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    max_tokens: int = Field(default=4)
    
    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)
        # 메시지 추가 후 트리밍 수행
        trimmer = trim_messages(
            strategy="last",
            max_tokens=self.max_tokens,
            token_counter=len
        )
        self.messages = trimmer.invoke(self.messages)
```

#### 대화 요약 저장

```python
class SummarizedInMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    summary_threshold: int = Field(default=6)
    llm: ChatOpenAI = Field(default_factory=lambda: ChatOpenAI(model="gpt-4.1-mini"))
    
    def add_messages(self, new_messages: List[BaseMessage]) -> None:
        self.messages.extend(new_messages)
        
        if len(self.messages) >= self.summary_threshold:
            # 마지막 사용자-AI 메시지 쌍 저장
            last_user_message = self.messages[-2]
            last_ai_message = self.messages[-1]
            
            # 이전 메시지들 요약
            summary_prompt = (
                "Distill the above chat messages into a single summary message. "
                "Include as many specific details as you can."
            )
            
            summary_chain_messages = [
                SystemMessage(content="You are a helpful assistant."),
                *self.messages[:-2],
                HumanMessage(content=summary_prompt)
            ]
            
            summary = self.llm.invoke(summary_chain_messages)
            
            # 요약 + 최신 메시지로 재구성
            self.messages = [summary, last_user_message, last_ai_message]
```

---

## 6. 실전 RAG 시스템 구현

### 🏠 주택청약 FAQ 봇 사례

실제 문서를 활용한 완전한 RAG 시스템 구현 예제입니다.

#### 📄 문서 전처리 파이프라인

```python
import re
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# 출력 형식 정의
class KeywordOutput(BaseModel):
    keyword: str = Field(description="텍스트에서 추출한 가장 중요한 키워드")
    summary: str = Field(description="텍스트의 간단한 요약")

# QA 쌍 추출 함수
def extract_qa_pairs(text):
    qa_pairs = []
    lines = [line.strip() for line in text.split('\n')]
    current_question = None
    current_answer = []
    
    for line in lines:
        # Q로 시작하는 질문 확인
        q_match = re.match(r'Q(\d+)\s+(.*)', line)
        if q_match:
            if current_question and current_answer:
                qa_pairs.append({
                    'number': current_number,
                    'question': current_question,
                    'answer': ' '.join(current_answer).strip()
                })
            
            current_number = int(q_match.group(1))
            current_question = q_match.group(2).strip()
            current_answer = []
        
        # A로 시작하는 답변 처리
        elif line.startswith('A '):
            current_answer.append(line.lstrip('A '))
    
    return qa_pairs

# 키워드 추출 체인
keyword_extractor = ChatPromptTemplate.from_template("""
텍스트에서 중요한 키워드를 추출하고 요약을 작성합니다.

텍스트: {input_text}

JSON 형식으로 반환:
- keyword: 핵심 키워드 (1개)
- summary: 텍스트 요약 (1-2문장)
""") | llm.with_structured_output(KeywordOutput)
```

#### 🔍 고급 검색 기능

```python
# 메타데이터 필터링
retriever = vector_store.as_retriever(
    search_kwargs={"filter": {"keyword": "해당 주택건설지역"}},
)

# MMR 검색 (다양성 기반)
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
)

# 복합 조건 필터링
retriever = vector_store.as_retriever(
    search_kwargs={"filter": {"$and": [
        {"keyword": "주택건설지역"}, 
        {"question_id": {"$lt": 10}}
    ]}},
)
```

#### 🎯 문서 관련성 평가

```python
from pydantic import BaseModel, Field
from typing import Literal

class DocumentRelevance(BaseModel):
    is_relevant: Literal["yes", "no"] = Field(
        description="문서가 질문에 답변하는데 필요한 정보를 포함하는지 여부"
    )

# 관련성 평가 체인
relevance_prompt = ChatPromptTemplate.from_messages([
    ("system", """주어진 컨텍스트가 질문에 답변하는데 필요한 정보를 포함하고 있는지 평가하세요.

다음 기준 중 하나 이상을 충족할 경우 'yes'로, 모두 충족하지 못하면 'no'로 답변하세요:
1. 컨텍스트가 질문에 답변하는데 필요한 정보를 직접적으로 포함하는가?
2. 컨텍스트의 정보로부터 답변에 필요한 내용을 논리적으로 추론할 수 있는가?"""),
    ("human", "[컨텍스트]\n{context}\n\n[질문]\n{question}")
])

relevance_chain = relevance_prompt | llm.with_structured_output(DocumentRelevance)
```

#### 🎨 Gradio 챗봇 인터페이스

```python
import gradio as gr
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SearchResult:
    context: str
    source_documents: Optional[List]

class RAGSystem:
    def __init__(self, llm, eval_llm, retriever):
        self.llm = llm
        self.eval_llm = eval_llm
        self.retriever = retriever
    
    def _check_relevance(self, docs: List, question: str) -> List:
        """문서의 관련성 확인"""
        relevant_docs = []
        for doc in docs:
            result = relevance_chain.invoke({
                "context": doc.page_content,
                "question": question
            })
            if result.is_relevant == "yes":
                relevant_docs.append(doc)
        return relevant_docs
    
    def generate_answer(self, message: str, history: List) -> str:
        # 문서 검색
        docs = self.retriever.invoke(message)
        relevant_docs = self._check_relevance(docs, message)
        
        if not relevant_docs:
            return "관련 문서를 찾을 수 없어 답변하기 어렵습니다."
        
        # 답변 생성
        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """주어진 문서의 내용만을 기반으로 답변하세요.
문서에 명확한 근거가 없는 내용은 "근거 없음"이라고 답변하세요."""),
            ("human", "문서들:\n{context}\n\n질문: {question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": message})
        
        # 참조 문서 정보 추가
        sources = self._format_source_documents(relevant_docs)
        return f"{answer}\n{sources}"

# Gradio 인터페이스
rag_system = RAGSystem(
    llm=ChatOpenAI(model="gpt-4.1-mini"),
    eval_llm=ChatOpenAI(model="gpt-4.1"),
    retriever=vector_store.as_retriever(search_kwargs={"k": 3})
)

demo = gr.ChatInterface(
    fn=rag_system.generate_answer,
    title="주택청약 FAQ 봇",
    description="주택청약 관련 질문에 답변해드립니다.",
    examples=[
        "수원시의 주택건설지역은 어디에 해당하나요?",
        "무주택 세대에 대해서 설명해주세요.",
        "2순위로 당첨된 사람이 청약통장을 다시 사용할 수 있나요?",
    ]
)

demo.launch()
```

---

## 🎯 주요 학습 포인트

### Day 2에서 배운 핵심 개념

1. **프롬프트 관리**: Langfuse를 통한 체계적인 프롬프트 버전 관리와 협업
2. **고급 프롬프트 기법**: Zero-shot, Few-shot, CoT 등 다양한 프롬프팅 전략
3. **시스템 안정성**: RunnableConfig와 Fallback을 통한 견고한 시스템 구축
4. **대화 관리**: 메시지 트리밍, 요약 등을 통한 효율적인 채팅 기록 관리
5. **실전 RAG**: 문서 전처리부터 UI까지 완전한 RAG 시스템 구현

### 💡 실무 적용 팁

- **프롬프트 관리**: 개발/스테이징/프로덕션 환경별로 프롬프트 버전을 관리하세요
- **CoT 활용**: 복잡한 추론이 필요한 작업에서는 단계별 사고 과정을 명시하세요
- **메모리 최적화**: 긴 대화에서는 요약과 트리밍을 조합하여 토큰 사용량을 관리하세요
- **품질 관리**: 문서 관련성 평가를 통해 RAG 시스템의 답변 품질을 높이세요

## 📁 관련 파일

- `Day2/DAY02_001_Langfuse_Pompt_Management.ipynb`
- `Day2/DAY02_002_Prompt_Engineering_Fewshot.ipynb`
- `Day2/DAY02_003_Prompt_Engineering_CoT.ipynb`
- `Day2/DAY02_004_RunnableConfig_Fallback.ipynb`
- `Day2/DAY02_005_Chat_History.ipynb`
- `Day2/DAY02_006_Housing_FAQ_Bot.ipynb` 