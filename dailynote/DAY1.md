

# LangChain Day 1 학습 가이드

Day 1에서는 LangChain의 기본 개념과 주요 컴포넌트들을 학습하고, RAG 시스템을 구현하는 방법을 익힙니다.

## 📚 학습 목차

1. [LangChain 컴포넌트](#1-langchain-컴포넌트)
   - LangChain 개요 (Chain, Agent, 모듈성)
   - Models (언어 모델)
     - OpenAI 모델 (`ChatOpenAI`, `model`, `temperature`)
     - 모델 호출 (`invoke()`, `response.content`, `response.response_metadata`)
     - 스트리밍 (`stream()`, `astream()`)
   - Messages (메시지 시스템)
     - 메시지 타입 (`HumanMessage`, `AIMessage`, `SystemMessage`)
     - 메시지 구성 (`content`, `additional_kwargs`)
     - 대화 관리 (`messages` 리스트)
   - Prompt Templates (프롬프트 템플릿)
     - 기본 템플릿 (`PromptTemplate`, `from_template()`, `input_variables`)
     - 채팅 템플릿 (`ChatPromptTemplate`, `from_messages()`)
     - 템플릿 호출 (`invoke()`, `format_prompt()`)
   - Output Parsers (출력 파서)
     - 문자열 파서 (`StrOutputParser`)
     - 구조화 출력 (`with_structured_output()`, `BaseModel`, `Field`)
     - 커스텀 파서 (`BaseOutputParser`, `parse()`)

2. [LangSmith & LCEL](#2-langsmith--lcel)
   - LangSmith 모니터링
     - 환경 설정 (`LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT`)
     - 추적 및 로깅 (자동 체인 추적)
     - 디버깅 도구 (프롬프트 디버깅, 성능 측정)
   - LCEL (LangChain Expression Language)
     - 파이프 연산자 (`|`, 체인 구성)
     - 체인 실행 (`invoke()`, `batch()`, `stream()`)
   - Runnable 인터페이스
     - 순차 실행 (`RunnableSequence`, `first`, `middle`, `last`)
     - 병렬 실행 (`RunnableParallel`, 딕셔너리 구성)
     - 입력 전달 (`RunnablePassthrough`, `assign()`)
     - 커스텀 함수 (`RunnableLambda`, 함수 래핑)
     - 조건부 실행 (`RunnableBranch`, 조건 분기)

3. [RAG 시스템 구현](#3-rag-시스템-구현)
   - RAG 개요 (Retrieval-Augmented Generation)
     - 워크플로우 (질문 → 검색 → 컨텍스트 전달 → 답변 생성)
     - 핵심 구성 요소 (문서 로더, 분할기, 임베딩, 벡터 저장소, 검색기)
   - Document Loaders (문서 로더)
     - 웹 문서 (`WebBaseLoader`, URL 리스트 처리)
     - PDF 파일 (`PyPDFLoader`, `load()`)
     - CSV 파일 (`CSVLoader`, `source_column`, `content_columns`, `metadata_columns`)
     - 텍스트 파일 (`TextLoader`, `encoding` 설정)
   - Text Splitting (텍스트 분할)
     - 재귀적 분할기 (`RecursiveCharacterTextSplitter`)
       - 기본 설정 (`chunk_size`, `chunk_overlap`, `length_function`, `separators`)
       - 토큰 기반 (`from_tiktoken_encoder()`, `encoding_name`)
       - HF 토크나이저 (`from_huggingface_tokenizer()`)
     - 의미 기반 분할 (`SemanticChunker`, `breakpoint_threshold_type`)
     - 분할 실행 (`split_documents()`, `split_text()`)
   - Document Embedding (문서 임베딩)
     - OpenAI 임베딩
       - 모델 설정 (`OpenAIEmbeddings`, `text-embedding-3-small`, `dimensions`)
       - 임베딩 실행 (`embed_documents()`, `embed_query()`)
       - 진행률 표시 (`show_progress_bar`)
     - HuggingFace 임베딩
       - 모델 초기화 (`HuggingFaceEmbeddings`, `BAAI/bge-m3`)
       - 설정 옵션 (`model_kwargs`, `encode_kwargs`, `normalize_embeddings`)
     - 유사도 계산 (`cosine_similarity`, `find_most_similar()`)
   - Vector Store (벡터 저장소)
     - Chroma
       - 생성 (`Chroma.from_documents()`, `embedding`, `collection_name`)
       - 저장 설정 (`persist_directory`, `collection_metadata`)
       - 검색 (`similarity_search()`, `similarity_search_with_score()`)
     - FAISS  
       - 생성 (`FAISS.from_documents()`)
       - 저장/로드 (`save_local()`, `load_local()`, `allow_dangerous_deserialization`)
   - Retriever (검색기)
     - 기본 검색기 (`as_retriever()`, `search_kwargs`)
     - 검색 전략
       - 유사도 검색 (`similarity`, `k` 파라미터)
       - 임계값 검색 (`similarity_score_threshold`, `score_threshold`)
       - MMR 검색 (`mmr`, `fetch_k`, `lambda_mult`)
     - 검색 실행 (`invoke()`, 검색 결과 반환)
   - RAG Chain (RAG 체인)
     - 프롬프트 설계 (컨텍스트 기반 답변 생성)
     - 체인 구성
       - 병렬 처리 (`RunnableParallel`, 컨텍스트와 질문 동시 처리)
       - 문서 포맷팅 (`format_docs()`, 문서 리스트 → 문자열)
       - 완전한 파이프라인 (검색기 | 프롬프트 | LLM | 파서)

---

## 1. LangChain 컴포넌트

### 🎯 LangChain 개요

LangChain은 **LLM 기반 애플리케이션 개발을 위한 프레임워크**입니다.

- **Chain**: 작업을 순차적으로 실행하는 파이프라인 구조
- **Agent**: 자율적 의사결정이 가능한 실행 단위
- **모듈성**: 독립적인 컴포넌트들을 조합해 복잡한 시스템 구현 가능

### 📋 주요 컴포넌트

#### 1) 모델 (Models)

```python
from langchain_openai import ChatOpenAI

# OpenAI 모델 설정
model = ChatOpenAI(model="gpt-4.1-mini")

# 메시지 전송 및 응답 받기
response = model.invoke("안녕하세요!")
print(response.content)  # 응답 내용
print(response.response_metadata)  # 메타데이터
```

#### 2) 메시지 (Messages)

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 시스템 메시지 (AI 동작 정의)
system_msg = SystemMessage(content="당신은 영어를 한국어로 번역하는 AI 어시스턴트입니다.")

# 사용자 메시지
human_msg = HumanMessage(content="Glory")

# 메시지 조합하여 모델에 전달
messages = [system_msg, human_msg]
response = model.invoke(messages)
```

#### 3) 프롬프트 템플릿 (Prompt Templates)

```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# 기본 프롬프트 템플릿
template = PromptTemplate.from_template("{topic}에 대한 이야기를 해줘")
prompt = template.invoke({"topic": "고양이"})

# 채팅 프롬프트 템플릿
chat_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 도움이 되는 비서입니다"),
    ("user", "{subject}에 대해 설명해주세요")
])
prompt = chat_template.invoke({"subject": "인공지능"})
```

#### 4) 출력 파서 (Output Parsers)

```python
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# 문자열 출력 파서
parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({"city": "서울"})

# 구조화된 출력
class CityInfo(BaseModel):
    name: str = Field(description="도시 이름")
    description: str = Field(description="도시의 특징")

structured_model = model.with_structured_output(CityInfo)
chain = prompt | structured_model
result = chain.invoke({"city": "서울"})
```

---

## 2. LangSmith & LCEL

### 🔍 LangSmith 모니터링

LangSmith는 **LLM 애플리케이션의 관찰성(Observability)**을 제공하는 도구입니다.

- 체인 실행 **로깅 및 추적**
- **프롬프트 디버깅**과 성능 측정
- 개발자가 LLM 체인과 에이전트를 효과적으로 디버깅 가능

```python
# .env 파일 설정
# LANGSMITH_API_KEY=your_langsmith_api_key
# LANGSMITH_TRACING=true
# LANGSMITH_PROJECT=your_project_name
```

### ⚡ LCEL (LangChain Expression Language)

LCEL은 `|` 연산자를 사용해 컴포넌트들을 순차적으로 연결하는 선언적 체이닝을 지원합니다.

#### 기본 체인 구성

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 컴포넌트 정의
prompt = PromptTemplate.from_template("'{text}'를 영어로 번역해주세요.")
model = ChatOpenAI(model="gpt-4.1-mini")
output_parser = StrOutputParser()

# 체인 구성 (Prompt | LLM | OutputParser)
chain = prompt | model | output_parser

# 실행
result = chain.invoke({"text": "안녕하세요"})
print(result)  # "Hello"
```

#### Runnable 인터페이스

**1) RunnableSequence**: 순차적 실행
```python
from langchain_core.runnables import RunnableSequence

# 파이프 연산자와 동일
chain = prompt | model | output_parser
# 또는
chain = RunnableSequence(first=prompt, middle=[model], last=output_parser)
```

**2) RunnableParallel**: 병렬 실행
```python
from langchain_core.runnables import RunnableParallel

# 병렬 처리로 주제 분류와 언어 감지 동시 실행
parallel_chain = RunnableParallel({
    "topic": question_chain,
    "language": language_chain,
    "question": itemgetter("question")
})
```

**3) RunnablePassthrough**: 입력 전달
```python
from langchain_core.runnables import RunnablePassthrough

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    modified=lambda x: process_text(x)
)
```

**4) RunnableLambda**: 커스텀 함수 통합
```python
from langchain_core.runnables import RunnableLambda

def preprocess_text(text: str) -> str:
    return text.strip().lower()

chain = (
    RunnableLambda(preprocess_text) |
    prompt |
    model |
    RunnableLambda(postprocess_response)
)
```

---

## 3. RAG 시스템 구현

### 🎯 RAG (Retrieval Augmented Generation) 개요

RAG는 **대규모 언어 모델(LLM)에 외부 지식을 연결하여 더 정확하고 최신의 정보를 제공**하는 AI 프레임워크입니다.

**워크플로우**: 사용자 질문 → 관련 문서 검색 → 컨텍스트와 함께 LLM에 전달 → 답변 생성

### 📄 문서 로더 (Document Loaders)

#### 웹 문서 로더
```python
from langchain_community.document_loaders import WebBaseLoader

web_loader = WebBaseLoader([
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://js.langchain.com/docs/tutorials/rag/"
])
web_docs = web_loader.load()
```

#### PDF 파일 로더
```python
from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader('./data/labor_law.pdf')
pdf_docs = pdf_loader.load()
print(f'PDF 문서 개수: {len(pdf_docs)}')
```

#### CSV 파일 로더
```python
from langchain_community.document_loaders.csv_loader import CSVLoader

csv_loader = CSVLoader(
    file_path="./data/kbo_teams_2023.csv",
    source_column="Team",
    content_columns=["Introduction"],
    metadata_columns=["Founded"],
    encoding="utf-8"
)
csv_docs = csv_loader.load()
```

### ✂️ 텍스트 분할 (Text Splitting)

#### RecursiveCharacterTextSplitter (권장)
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(pdf_docs)
print(f"생성된 청크 수: {len(chunks)}")
```

#### 토큰 기반 분할
```python
# TikToken 사용
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=300,
    chunk_overlap=0
)

# Hugging Face 토크나이저 사용
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=300,
    chunk_overlap=0
)
```

#### 의미 기반 분할
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

text_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    breakpoint_threshold_type="gradient"
)
chunks = text_splitter.split_documents([pdf_docs[0]])
```

### 🧠 문서 임베딩 (Document Embedding)

#### OpenAI 임베딩
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
    show_progress_bar=True
)

# 문서 임베딩
doc_embeddings = embeddings.embed_documents(documents)
# 쿼리 임베딩
query_embedding = embeddings.embed_query(query)
```

#### Hugging Face 임베딩
```python
from langchain_huggingface import HuggingFaceEmbeddings

# BGE-M3 모델 (한국어 성능 우수)
embeddings_bge = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

#### 유사도 계산
```python
from langchain_community.utils.math import cosine_similarity

def find_most_similar(query, doc_embeddings, documents, embeddings_model):
    query_embedding = embeddings_model.embed_query(query)
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    most_similar_idx = np.argmax(similarities)
    
    return {
        "document": documents[most_similar_idx],
        "similarity": similarities[most_similar_idx],
        "index": most_similar_idx
    }
```

### 🗄️ 벡터 저장소 (Vector Store)

#### Chroma
```python
from langchain_chroma import Chroma

# 벡터 저장소 생성
chroma_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="labor_law",
    persist_directory="./chroma_db",
    collection_metadata={'hnsw:space': 'cosine'}
)

# 유사도 검색
results = chroma_db.similarity_search(query, k=5)

# 점수가 포함된 검색
results_with_scores = chroma_db.similarity_search_with_score(query, k=5)
```

#### FAISS
```python
from langchain_community.vectorstores import FAISS

# FAISS 벡터 저장소 생성
faiss_db = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 저장 및 로드
faiss_db.save_local("./faiss_index")
faiss_db = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
```

### 🔍 검색기 (Retriever)

#### 기본 유사도 검색
```python
retriever = chroma_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
retrieved_docs = retriever.invoke(query)
```

#### 임계값 기반 검색
```python
threshold_retriever = chroma_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.25,
        "k": 10
    }
)
```

#### MMR (Maximal Marginal Relevance) 검색
```python
mmr_retriever = chroma_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5  # 관련성 vs 다양성 균형
    }
)
```

### 🔗 RAG 체인 구현

#### 프롬프트 템플릿
```python
from langchain_core.prompts import ChatPromptTemplate

template = """당신은 전문적인 문서 분석 AI입니다. 주어진 컨텍스트를 바탕으로 정확하고 유용한 답변을 제공하세요.

## 답변 지침
- 컨텍스트에 있는 정보만을 사용하여 답변하세요
- 확실하지 않은 정보는 "명확하지 않습니다"라고 명시하세요
- 답변은 논리적이고 구조화된 형태로 제공하세요

## 컨텍스트
{context}

## 질문
{question}

## 답변 형식
**핵심 답변:** (질문에 대한 직접적인 답변)
**세부 설명:** (추가적인 설명이나 배경 정보)
**관련 정보:** (컨텍스트에서 발견된 연관 정보)
"""

prompt = ChatPromptTemplate.from_template(template)
```

#### 완전한 RAG 체인
```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# LLM 설정
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join([f"{doc.page_content}" for doc in docs])

# RAG 체인 생성
rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

# 실행
query = "탄력 근로에 대해서 설명해주세요"
output = rag_chain.invoke(query)
print(f"쿼리: {query}")
print(f"답변: {output}")
```

---

## 🛠️ 설치 및 설정

### 필수 패키지 설치
```bash
# 기본 패키지
uv add langchain langchain-openai langchain-community

# 텍스트 분할
uv add langchain-text-splitters

# 벡터 저장소
uv add langchain-chroma faiss-cpu

# 문서 로더
uv add bs4 pypdf

# 임베딩
uv add langchain-huggingface sentence-transformers

# 실험적 기능
uv add langchain-experimental

# LangSmith (선택사항)
uv add langsmith
```

### 환경 변수 설정
```bash
# .env 파일
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key  # 선택사항
LANGSMITH_TRACING=true  # 선택사항
LANGSMITH_PROJECT=your_project_name  # 선택사항
```

---

## 📊 성능 최적화 팁

### 임베딩 모델 선택
| 모델 | 차원 | 언어 지원 | 비용 | 성능 | 사용 사례 |
|------|------|----------|------|------|----------|
| OpenAI text-embedding-3-small | 1536 | 다국어 | 유료 | 높음 | 프로덕션 |
| OpenAI text-embedding-3-large | 3072 | 다국어 | 유료 | 최고 | 고성능 요구 |
| BAAI/bge-m3 | 1024 | 다국어 | 무료 | 높음 | 한국어 특화 |

### 벡터 저장소 선택
| 종류 | 장점 | 단점 | 사용 사례 |
|------|------|------|----------|
| Chroma | 설치 간단, 로컬 친화적 | 대용량 처리 한계 | 개발, 프로토타입 |
| FAISS | 매우 빠름, 확장성 우수 | 설정 복잡 | 대용량 검색 |
| Pinecone | 완전 관리형, 고성능 | 유료, 클라우드 의존 | 프로덕션 |

### 검색 전략 선택
| 전략 | 설명 | 장점 | 단점 | 사용 사례 |
|------|------|------|------|----------|
| similarity | 단순 유사도 검색 | 빠름, 직관적 | 다양성 부족 | 일반적인 검색 |
| similarity_score_threshold | 임계값 기반 검색 | 품질 보장 | 결과 수 불안정 | 고품질 결과 필요 |
| mmr | 최대 한계 관련성 | 다양성 우수 | 느림 | 포괄적 정보 필요 |

---

## 🎯 다음 단계

Day 1에서 LangChain의 기본기를 익혔다면, 다음 내용들을 학습해보세요:

- **고급 RAG 기법**: 하이브리드 검색, 쿼리 변환, 컨텍스트 압축
- **에이전트 시스템**: 도구 호출, 멀티 에이전트, 워크플로우
- **메모리 관리**: 대화 기록, 세션 관리
- **성능 최적화**: 캐싱, 배치 처리, 스트리밍

---

## 📚 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/)
- [LangSmith 가이드](https://docs.smith.langchain.com/)
- [RAG 튜토리얼](https://python.langchain.com/docs/tutorials/rag/)
- [LCEL 가이드](https://python.langchain.com/docs/concepts/#langchain-expression-language-lcel)
