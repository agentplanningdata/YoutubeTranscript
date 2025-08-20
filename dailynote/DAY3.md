# LangChain Day 3 학습 가이드

Day 3에서는 RAG 시스템의 고급 기법과 체계적인 평가 방법론을 학습합니다.

## 📚 학습 목차

1. [RAG 성능 평가](#1-rag-성능-평가)
   - RAGAS 프레임워크
     - 평가 메트릭 (`LLMContextRecall`, `Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, `FactualCorrectness`)
     - 래퍼 클래스 (`LangchainLLMWrapper`, `LangchainEmbeddingsWrapper`)
     - 평가 실행 (`evaluate()`, `EvaluationDataset`)
   - 평가 데이터셋 생성
     - 테스트셋 생성기 (`TestsetGenerator`, `Persona`)
     - 쿼리 신시사이저 (`SingleHopSpecificQuerySynthesizer`, `MultiHopSpecificQuerySynthesizer`, `MultiHopAbstractQuerySynthesizer`)
   - Reference-Free vs Reference-Based 평가
   - A/B 테스트 (`compare_rag_systems()`)

2. [정보 검색 평가지표](#2-정보-검색-평가지표)
   - 기본 평가 지표
     - Hit Rate (`calculate_hit_rate()`, 적중률 계산)
     - MRR (`calculate_mrr()`, Mean Reciprocal Rank)
     - MAP (`calculate_map_at_k()`, Mean Average Precision)
     - NDCG (`calculate_ndcg_at_k()`, Normalized Discounted Cumulative Gain)
   - ranx 라이브러리
     - 핵심 클래스 (`Qrels`, `Run`)
     - 평가 함수 (`evaluate()`, `compare()`)
     - 데이터 변환 (`convert_to_ranx_format()`)
   - 평가 라이브러리 비교 (ranx, pytrec_eval, ir-measures)
   - 지표별 특성 비교 및 활용 시나리오

3. [하이브리드 검색](#3-하이브리드-검색)
   - 검색 방식 분류 (의미론적, 키워드, 하이브리드)
   - BM25 알고리즘
     - BM25 검색기 (`BM25Retriever`)
     - 점수 분석 (`get_scores()`, `analyze_bm25_scores()`)
   - 한국어 처리
     - 토크나이저 (`Kiwi`, `add_user_word()`, `korean_tokenizer()`)
     - 전처리 함수 (`preprocess_func`)
   - 하이브리드 구현
     - 앙상블 검색기 (`EnsembleRetriever`, `weights`)
     - 성능 비교 (`compare_retrieval_methods()`)
   - ranx-k 평가
     - 유사도 기반 평가 (`evaluate_with_ranx_similarity`)
     - 평가 방법 (`kiwi_rouge`, `embedding`)
   - 시각화 (`matplotlib`, `seaborn`)

4. [쿼리 확장](#4-쿼리-확장)
   - Query Reformulation
     - 프롬프트 템플릿 (`ChatPromptTemplate`)
     - 체인 구성 (`reformulation_chain`)
   - Multi Query
     - 멀티쿼리 검색기 (`MultiQueryRetriever`)
     - 출력 파서 (`LineListOutputParser`, `BaseOutputParser`)
     - 커스텀 체인 (`llm_chain`, `parser_key`)
   - Decomposition
     - 질문 분해 프롬프트 (`QUERY_PROMPT`)
     - 서브 질문 생성 체인
   - Step-Back Prompting
     - Few-shot 템플릿 (`FewShotChatMessagePromptTemplate`)
     - 예제 프롬프트 (`example_prompt`)
     - 컨텍스트 통합 (`RunnablePassthrough`)
   - HyDE (Hypothetical Document Embedding)
     - 가상 문서 생성 (`hyde_chain`)
     - 검색기 연결 (`hyde_retriever`)
   - 각 기법별 성능 비교

5. [재순위 및 압축](#5-재순위-및-압축)
   - 검색 결과 재순위 (Reranking)
     - 재순위 모델 (`CrossEncoder`, `SentenceTransformer`)
     - 점수 기반 재정렬 (`rerank_documents()`)
   - 컨텍스트 압축 (Context Compression)
     - 압축기 (`LLMChainExtractor`, `LLMChainFilter`)
     - 압축 검색기 (`ContextualCompressionRetriever`)
     - 관련성 필터 (`EmbeddingsFilter`, `SimilarityThresholdFilter`)
   - 문서 변환기 (`DocumentTransformer`)
   - 토큰 효율성 최적화 (`get_stateful_documents()`)

6. [생성 평가 지표](#6-생성-평가-지표)
   - 전통적 지표
     - BLEU (`load_metric("bleu")`, `sacrebleu`)
     - ROUGE (`load_metric("rouge")`, `rouge_score`)
     - METEOR (`nltk.translate.meteor_score`)
   - 임베딩 기반 지표
     - BERTScore (`load_metric("bertscore")`)
     - Semantic Similarity (`sentence_transformers`)
   - 환각 탐지 (Hallucination Detection)
     - 팩트 체킹 (`factcheck_chain`)
     - 일관성 검사 (`consistency_checker`)
   - 생성 품질 자동 평가 (`evaluate_generation()`)

7. [LLM-as-Judge](#7-llm-as-judge)
   - 평가 프롬프트 설계
     - 평가 템플릿 (`judge_template`, `scoring_rubric`)
     - 평가 체인 (`judge_chain`, `multi_aspect_evaluator`)
   - 평가 기준 정의
     - 점수 추출기 (`ScoreExtractor`, `parse_evaluation_result()`)
     - 다중 평가자 (`MultiJudgeEvaluator`)
   - 일관성 및 신뢰성
     - 일관성 측정 (`calculate_inter_rater_reliability()`)
     - 신뢰도 계산 (`confidence_interval()`)
   - LangChain 평가 도구 (`LangChainStringEvaluator`)

8. [Langfuse 평가 시스템](#8-langfuse-평가-시스템)
   - Langfuse 핵심 클래스
     - 클라이언트 (`Langfuse`, `get_client()`)
     - 콜백 핸들러 (`CallbackHandler`)
     - 평가 데코레이터 (`@observe`, `@langfuse_context`)
   - 평가 기능
     - 점수 추가 (`score()`, `ScoreType`)
     - 추적 관리 (`get_traces()`, `TraceClient`)
   - 자동 평가 파이프라인
     - 배치 평가 (`run_batch_evaluation()`)
     - 평가 결과 분석 (`analyze_evaluation_results()`)
   - 팀 협업
     - 프로젝트 관리 (`project_name`, `session_id`)
     - 결과 공유 (`export_traces()`, `evaluation_dashboard`)

---

## 1. RAG 성능 평가

### 🎯 RAG 평가의 중요성

RAG 시스템은 검색과 생성이 결합된 복합 시스템으로, 각 구성 요소의 성능이 최종 품질에 영향을 미칩니다.

**주요 평가 목표:**
- **검색 품질**: 관련성 높은 문서를 얼마나 잘 찾는가?
- **생성 품질**: 검색된 정보를 얼마나 잘 활용하는가?
- **종합 성능**: 사용자에게 얼마나 유용한 답변을 제공하는가?

### 📊 평가 차원 (Evaluation Dimensions)

| 평가 영역 | 세부 지표 | 설명 |
|-----------|-----------|------|
| **검색 (Retrieval)** | Context Relevancy | 검색된 문서가 질문과 얼마나 관련있는가? |
|  | Context Recall | 정답에 필요한 모든 정보가 검색되었는가? |
| **생성 (Generation)** | Faithfulness | 생성된 답변이 검색된 문서에 충실한가? |
|  | Answer Relevancy | 생성된 답변이 질문과 관련있는가? |
| **종합** | Answer Correctness | 생성된 답변이 정답과 일치하는가? |

### 🔧 RAGAS 프레임워크

RAGAS는 RAG 시스템을 위한 오픈소스 평가 프레임워크입니다.

#### 환경 설정

```python
from ragas.metrics import (
    context_relevancy,      # 컨텍스트 관련성
    context_recall,         # 컨텍스트 회상률
    faithfulness,          # 충실도
    answer_relevancy,      # 답변 관련성
    answer_correctness     # 답변 정확성
)

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
```

#### 기본 평가 실행

```python
from ragas import EvaluationDataset
from langchain_openai import ChatOpenAI

# 평가용 LLM 설정
evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4.1-mini", temperature=0)
)

# 평가 데이터셋 생성
def create_evaluation_dataset(questions_data, rag_chain_func):
    evaluation_data = []
    
    for item in questions_data:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        
        # RAG 체인 실행
        result = rag_chain_func(question)
        
        # 평가 데이터 구성
        eval_sample = {
            "user_input": question,
            "response": result["answer"],
            "retrieved_contexts": [doc.page_content for doc in result["retrieved_docs"]],
            "reference": ground_truth
        }
        
        evaluation_data.append(eval_sample)
    
    return EvaluationDataset.from_list(evaluation_data)

# 평가 실행
metrics = [
    LLMContextRecall(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm),
    ContextPrecision(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm)
]

results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    llm=evaluator_llm
)
```

### 📈 자동 데이터셋 생성

```python
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

# 페르소나 정의
personas = [
    Persona(
        name="graduate_researcher",
        role_description="미국 전기차 시장을 연구하는 한국인 박사과정 연구원입니다."
    ),
    Persona(
        name="industry_analyst", 
        role_description="한국 자동차 회사에서 미국 전기차 시장을 분석하는 주니어 연구원입니다."
    )
]

# 테스트셋 생성기
testset_generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    persona_list=personas
)

# 합성 데이터셋 생성
synthetic_dataset = testset_generator.generate_with_langchain_docs(
    documents=split_docs,
    testset_size=50
)
```

---

## 2. 정보 검색 평가지표

### 📊 주요 평가 지표

#### Hit Rate (적중률)

상위 k개 검색 결과에 관련 문서가 **하나라도** 포함되어 있는지 측정하는 지표입니다.

```python
def calculate_hit_rate(ground_truth: List[List[str]], 
                      predictions: List[List[str]], 
                      k: int) -> float:
    hits = 0
    total_queries = len(ground_truth)
    
    for gt, pred in zip(ground_truth, predictions):
        top_k_pred = pred[:k]
        hit = any(doc in gt for doc in top_k_pred)
        if hit:
            hits += 1
    
    return hits / total_queries
```

#### MRR (Mean Reciprocal Rank)

**첫 번째 관련 문서의 순위**에 기반한 평가 지표입니다.

```python
def calculate_mrr(ground_truth: List[List[str]], 
                  predictions: List[List[str]]) -> float:
    reciprocal_ranks = []
    
    for gt, pred in zip(ground_truth, predictions):
        first_relevant_rank = None
        for rank, doc_id in enumerate(pred, 1):
            if doc_id in gt:
                first_relevant_rank = rank
                break
        
        rr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        reciprocal_ranks.append(rr)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

#### NDCG (Normalized Discounted Cumulative Gain)

관련성 점수와 순위를 모두 고려한 **가장 정교한 평가 지표**입니다.

```python
import math

def calculate_ndcg_at_k(ground_truth: List[List[str]], 
                        predictions: List[List[str]], 
                        k: int) -> float:
    ndcg_scores = []
    
    for gt, pred in zip(ground_truth, predictions):
        top_k_pred = pred[:k]
        
        # DCG 계산
        dcg = 0.0
        for rank, doc_id in enumerate(top_k_pred, 1):
            rel_score = 1 if doc_id in gt else 0
            if rel_score > 0:
                gain = (2**rel_score - 1) / math.log2(rank + 1)
                dcg += gain
        
        # IDCG 계산 (이상적인 순위)
        ideal_scores = [1] * min(len(gt), k)
        idcg = sum((2**rel - 1) / math.log2(rank + 1) 
                  for rank, rel in enumerate(ideal_scores, 1))
        
        # NDCG 계산
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    return sum(ndcg_scores) / len(ndcg_scores)
```

### 🛠️ ranx 라이브러리 활용

```python
from ranx import Qrels, Run, evaluate

def evaluate_with_ranx(ground_truth, system_results):
    # 데이터 변환
    qrels_dict = {}
    run_dict = {}
    
    for i, (gt, pred) in enumerate(zip(ground_truth, system_results)):
        query_id = f"q_{i+1}"
        qrels_dict[query_id] = {doc: 1 for doc in gt}
        run_dict[query_id] = {doc: len(pred) - rank for rank, doc in enumerate(pred)}
    
    # ranx 객체 생성
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)
    
    # 평가 실행
    metrics = ["hit_rate@5", "mrr", "map@5", "ndcg@5"]
    results = evaluate(qrels, run, metrics)
    
    return results
```

---

## 3. 하이브리드 검색

### 🔍 검색 방식 분류

#### 의미론적 검색 (Semantic Search)
- 벡터 임베딩을 활용한 의미 기반 검색
- 동의어와 문맥적 의미 파악 가능
- 정확한 키워드 매칭에 약함

#### 키워드 검색 (Keyword Search)
- BM25 알고리즘 기반 키워드 매칭 검색
- 정확한 키워드 매칭, 빠른 처리 속도
- 의미적 유사성 파악 제한적

#### 하이브리드 검색 (Hybrid Search)
- 키워드 검색과 의미론적 검색의 결합
- EnsembleRetriever를 통한 구현

### 🔧 BM25 알고리즘

```python
from langchain_community.retrievers import BM25Retriever
from kiwipiepy import Kiwi

# 한국어 토크나이저 설정
def setup_korean_tokenizer():
    kiwi = Kiwi()
    custom_words = [
        ('리비안', 'NNP'),
        ('테슬라', 'NNP'),
        ('전기차', 'NNG'),
    ]
    
    for word, pos in custom_words:
        kiwi.add_user_word(word, pos)
    
    return kiwi

def korean_tokenizer(text, kiwi_model):
    return [token.form for token in kiwi_model.tokenize(text)]

# BM25 검색기 생성
def create_bm25_retriever(documents, kiwi_model, k=5):
    def preprocess_func(text):
        return korean_tokenizer(text, kiwi_model)
    
    bm25_retriever = BM25Retriever.from_documents(
        documents=documents,
        preprocess_func=preprocess_func,
        k=k
    )
    
    return bm25_retriever
```

### ⚖️ 하이브리드 검색기 구현

```python
from langchain.retrievers import EnsembleRetriever

def create_hybrid_retriever(vector_store, bm25_retriever, weights=None):
    if weights is None:
        weights = [0.5, 0.5]  # 동일한 가중치
    
    # 의미론적 검색기
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # 앙상블 검색기 생성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=weights
    )
    
    return ensemble_retriever

# 다양한 가중치로 성능 비교
retrievers = {
    "의미론적 검색": vector_store.as_retriever(search_kwargs={"k": 5}),
    "키워드 검색": bm25_retriever,
    "하이브리드 (5:5)": create_hybrid_retriever(vector_store, bm25_retriever, [0.5, 0.5]),
    "하이브리드 (7:3)": create_hybrid_retriever(vector_store, bm25_retriever, [0.7, 0.3]),
    "하이브리드 (3:7)": create_hybrid_retriever(vector_store, bm25_retriever, [0.3, 0.7]),
}
```

### 📊 ranx-k 라이브러리를 활용한 평가

```python
from ranx_k.evaluation import evaluate_with_ranx_similarity

# ROUGE 점수 기반 평가 (문자열 유사도)
results_rouge = evaluate_with_ranx_similarity(
    retriever=hybrid_retriever,
    questions=questions,
    reference_contexts=reference_contexts,
    k=5,
    method='kiwi_rouge',
    similarity_threshold=0.8
)

# 임베딩 기반 평가 (의미적 유사도)
results_embedding = evaluate_with_ranx_similarity(
    retriever=hybrid_retriever,
    questions=questions,
    reference_contexts=reference_contexts,
    k=5,
    method='embedding',
    embedding_model="BAAI/bge-m3",
    similarity_threshold=0.9
)
```

---

## 4. 쿼리 확장

### 🔄 쿼리 확장 방법론

| 방법론 | 핵심 아이디어 | 장점 | 단점 | 적용 상황 |
|--------|---------------|------|------|-----------| 
| **Query Reformulation** | LLM으로 질문 재작성 | 구현 간단, 즉시 적용 | 단일 변형만 생성 | 일반적인 질문 개선 |
| **Multi Query** | 다양한 관점의 질문 생성 | 검색 다양성 증가 | 계산 비용 증가 | 모호한 질문 처리 |
| **Decomposition** | 복잡한 질문을 하위 질문으로 분해 | 체계적 접근 | 분해 정확도 의존 | 복합적 질문 |
| **Step-Back Prompting** | 일반적 맥락에서 구체적 답변으로 | 포괄적 이해 | 추가 검색 필요 | 전문적/복잡한 질문 |
| **HyDE** | 가상 답변 문서 생성 | 의미적 정렬 우수 | 환각 위험 | Zero-shot 상황 |

### 1️⃣ Query Reformulation

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 쿼리 리포뮬레이션을 위한 프롬프트
reformulation_template = """다음 질문을 검색 성능을 향상시키기 위해 다시 작성해주세요:

[질문]
{question}

다음 방식으로 질문을 재작성하세요:
1. 동의어 추가
2. 더 구체적인 키워드 포함
3. 관련된 개념 확장

[재작성된 질문]
"""

# 쿼리 리포뮬레이션 체인
prompt = ChatPromptTemplate.from_template(reformulation_template)
llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0)
reformulation_chain = prompt | llm | StrOutputParser()

# 검색기와 연결
reformulation_retriever = reformulation_chain | chroma_k_retriever
```

### 2️⃣ Multi Query

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import BaseOutputParser
from typing import List

class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        return [line.strip() for line in text.strip().split("\n") if line.strip()]

# 커스텀 프롬프트
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Generate three different versions of the given user question to retrieve relevant documents from a vector database.

[Original question]
{question}

[Alternative questions]
""",
)

# 멀티쿼리 체인
multiquery_chain = QUERY_PROMPT | llm | LineListOutputParser()

# 멀티쿼리 검색기
multi_query_retriever = MultiQueryRetriever(
    retriever=chroma_k_retriever,
    llm_chain=multiquery_chain,
    parser_key="lines"
)
```

### 3️⃣ Decomposition

```python
# 질문 분해를 위한 프롬프트
DECOMPOSITION_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant. Decompose the given question into multiple sub-questions.

[Input question] 
{question}

[Sub-questions (5)]
""",
)

# 질문 분해 체인
decomposition_chain = DECOMPOSITION_PROMPT | llm | LineListOutputParser()

# 분해 기반 검색기
decomposition_retriever = MultiQueryRetriever(
    retriever=chroma_k_retriever,
    llm_chain=decomposition_chain,
    parser_key="lines"
)
```

### 4️⃣ Step-Back Prompting

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

# Few-shot 예제
examples = [
    {
        "input": "애플의 M1 칩 개발이 기업 가치에 미친 영향은?",
        "output": "기업의 핵심 기술 내재화가 경쟁우위에 미치는 영향은 무엇인가?",
    },
    {
        "input": "아마존의 AWS가 수익성에 기여하는 방식은?",
        "output": "기업의 새로운 사업 영역 확장이 수익 구조에 미치는 영향은 무엇인가?",
    }
]

# Step-Back 프롬프트
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

step_back_prompt = ChatPromptTemplate.from_messages([
    ("system", """구체적인 질문을 일반적인 관점에서 재해석하는 것이 임무입니다."""),
    few_shot_prompt,
    ("user", "{question}"),
])

# Step-Back 체인
step_back_chain = step_back_prompt | llm | StrOutputParser()
step_back_retriever = step_back_chain | chroma_k_retriever

# 최종 답변 생성
response_prompt = ChatPromptTemplate.from_template(
    """다음 컨텍스트를 바탕으로 포괄적인 답변을 제공해주세요.

일반 컨텍스트:
{normal_context}

기본 개념 컨텍스트:
{step_back_context}

원래 질문: {question}

답변:"""
)

answer_chain = (
    {
        "normal_context": chroma_k_retriever,
        "step_back_context": step_back_retriever,
        "question": RunnablePassthrough(),
    }
    | response_prompt
    | llm
    | StrOutputParser()
)
```

### 5️⃣ HyDE (Hypothetical Document Embedding)

```python
# HyDE를 위한 프롬프트
hyde_template = """주어진 질문에 대한 이상적인 문서 내용을 생성해주세요.
문서는 학술적이고 전문적인 톤으로 작성되어야 합니다.

질문: {question}

문서 내용:"""

hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
hyde_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 가상 문서 생성 체인
hyde_chain = hyde_prompt | hyde_llm | StrOutputParser()

# HyDE 검색기
hyde_retriever = hyde_chain | chroma_k_retriever

# 최종 RAG 체인
rag_template = """다음 컨텍스트를 바탕으로 질문에 답변해주세요:

컨텍스트:
{context}

질문: {question}

답변:"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = rag_prompt | llm | StrOutputParser()

def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

# HyDE 전체 체인
query = "리비안의 사업 경쟁력은 어디서 나오나요?"

# Step 1: 가상 문서 생성 및 검색
retrieved_docs = hyde_retriever.invoke({"question": query})

# Step 2: 최종 답변 생성
final_answer = rag_chain.invoke({
    "context": format_docs(retrieved_docs), 
    "question": query
})
```

---

## 5. 재순위 및 압축

### 🔄 검색 결과 재순위 (Reranking)

검색된 문서들을 질문과의 관련성에 따라 재정렬하는 기법입니다.

**주요 특징:**
- **정밀 검색**: 1차 검색 결과를 더 정교하게 평가
- **Cross-encoder 활용**: 질문-문서 쌍을 직접 비교
- **관련성 점수 재계산**: 더 정확한 순위 매기기

```python
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

# 압축 체인 생성
compressor = LLMChainExtractor.from_llm(llm)

# 압축 검색기
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
```

### 🗜️ 컨텍스트 압축 (Context Compression)

검색된 문서에서 질문과 관련된 부분만 추출하는 기법입니다.

**주요 이점:**
- **토큰 효율성**: 불필요한 내용 제거로 토큰 사용량 감소
- **정확도 향상**: 핵심 정보에 집중하여 답변 품질 개선
- **처리 속도**: 컨텍스트 길이 단축으로 응답 시간 단축

---

## 6. 생성 평가 지표

### 📊 텍스트 생성 품질 측정

#### BLEU (Bilingual Evaluation Understudy)
- n-gram 정확도 기반 평가
- 기계 번역에서 주로 사용
- 정확한 단어 매칭에 집중

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- 요약 품질 평가에 특화
- Recall 중심의 평가 방식
- ROUGE-1, ROUGE-2, ROUGE-L 변형

#### BERTScore
- 문맥적 임베딩 기반 평가
- 의미적 유사성 고려
- BERT 모델을 활용한 측정

```python
from datasets import load_metric

# BLEU 계산
bleu = load_metric("bleu")
bleu_score = bleu.compute(predictions=predictions, references=references)

# ROUGE 계산
rouge = load_metric("rouge")
rouge_scores = rouge.compute(predictions=predictions, references=references)

# BERTScore 계산
bertscore = load_metric("bertscore")
bert_scores = bertscore.compute(
    predictions=predictions, 
    references=references,
    model_type="distilbert-base-uncased"
)
```

### 🔍 환각 탐지 (Hallucination Detection)

```python
def detect_hallucination(context, generated_text, llm):
    hallucination_prompt = """
    주어진 컨텍스트와 생성된 텍스트를 비교하여 환각(hallucination)이 있는지 확인해주세요.
    
    컨텍스트: {context}
    생성된 텍스트: {generated_text}
    
    환각이 있으면 "YES", 없으면 "NO"로 답하고 이유를 설명해주세요.
    """
    
    result = llm.invoke(hallucination_prompt.format(
        context=context,
        generated_text=generated_text
    ))
    
    return result.content
```

---

## 7. LLM-as-Judge

### ⚖️ LLM을 평가자로 활용

LLM을 사용하여 생성된 답변의 품질을 자동으로 평가하는 방법입니다.

**주요 장점:**
- **확장성**: 대량의 데이터 자동 평가 가능
- **일관성**: 동일한 기준으로 일관된 평가
- **비용 효율성**: 인간 평가 대비 저비용

#### 평가 프롬프트 설계

```python
judge_template = """다음 기준으로 답변을 평가해주세요:

1. 관련성 (1-5점): 질문과 얼마나 관련있는가?
2. 정확성 (1-5점): 제공된 정보가 얼마나 정확한가?
3. 완성도 (1-5점): 답변이 얼마나 완전한가?
4. 명확성 (1-5점): 답변이 얼마나 이해하기 쉬운가?

질문: {question}
컨텍스트: {context}
답변: {answer}

평가 결과:
관련성: [점수] - [이유]
정확성: [점수] - [이유]
완성도: [점수] - [이유]  
명확성: [점수] - [이유]
총점: [총점/20]
"""

judge_prompt = ChatPromptTemplate.from_template(judge_template)
judge_chain = judge_prompt | llm | StrOutputParser()

# 평가 실행
evaluation_result = judge_chain.invoke({
    "question": question,
    "context": context,
    "answer": answer
})
```

#### 일관성 확보 방법

```python
# 다중 평가를 통한 일관성 확보
def multi_judge_evaluation(question, context, answer, num_evaluations=3):
    scores = []
    
    for i in range(num_evaluations):
        result = judge_chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        })
        
        # 점수 추출 및 저장
        score = extract_score(result)
        scores.append(score)
    
    # 평균 점수 계산
    average_score = sum(scores) / len(scores)
    confidence = calculate_confidence(scores)
    
    return {
        "average_score": average_score,
        "confidence": confidence,
        "individual_scores": scores
    }
```

---

## 8. Langfuse 평가 시스템

### 📊 Langfuse 평가 기능

Langfuse는 LLM 애플리케이션을 위한 종합적인 평가 및 모니터링 플랫폼입니다.

**주요 기능:**
- **자동 평가**: 다양한 메트릭을 통한 자동 평가
- **시각화**: 평가 결과의 직관적 시각화
- **추적**: 실험 및 평가 결과 추적
- **협업**: 팀 단위 평가 결과 공유

#### 평가 파이프라인 구축

```python
from langfuse import Langfuse
from langfuse.decorators import observe

# Langfuse 클라이언트 초기화
langfuse = Langfuse()

@observe()
def evaluate_rag_system(question, expected_answer):
    # RAG 시스템 실행
    result = rag_chain.invoke(question)
    
    # 평가 메트릭 계산
    relevance_score = calculate_relevance(question, result["answer"])
    faithfulness_score = calculate_faithfulness(result["context"], result["answer"])
    
    # Langfuse에 결과 로깅
    langfuse.score(
        name="relevance",
        value=relevance_score,
        comment="Question-answer relevance"
    )
    
    langfuse.score(
        name="faithfulness",
        value=faithfulness_score,
        comment="Context faithfulness"
    )
    
    return {
        "answer": result["answer"],
        "relevance": relevance_score,
        "faithfulness": faithfulness_score
    }

# 배치 평가 실행
def run_batch_evaluation(test_dataset):
    results = []
    
    for item in test_dataset:
        result = evaluate_rag_system(
            item["question"], 
            item["expected_answer"]
        )
        results.append(result)
    
    return results
```

#### 평가 결과 분석

```python
# 평가 결과 조회 및 분석
def analyze_evaluation_results():
    # Langfuse에서 평가 결과 조회
    traces = langfuse.get_traces(
        name="rag_evaluation",
        limit=100
    )
    
    # 평가 지표별 통계
    relevance_scores = [trace.scores["relevance"] for trace in traces]
    faithfulness_scores = [trace.scores["faithfulness"] for trace in traces]
    
    analysis = {
        "avg_relevance": sum(relevance_scores) / len(relevance_scores),
        "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "total_evaluations": len(traces),
        "score_distribution": calculate_distribution(relevance_scores)
    }
    
    return analysis
```

---

## 🎯 Day 3 학습 포인트

### 핵심 개념 요약

1. **체계적 평가**: RAGAS를 통한 RAG 시스템의 정량적 평가
2. **검색 지표**: Hit Rate, MRR, MAP, NDCG를 활용한 검색 성능 측정
3. **하이브리드 검색**: 의미론적 + 키워드 검색의 시너지 효과
4. **쿼리 최적화**: 다양한 쿼리 확장 기법을 통한 검색 성능 향상
5. **생성 품질**: BLEU, ROUGE, BERTScore를 통한 생성 품질 평가
6. **자동 평가**: LLM-as-Judge를 통한 확장 가능한 평가 시스템
7. **통합 관리**: Langfuse를 통한 평가 결과 추적 및 관리

### 💡 실무 적용 팁

- **단계별 평가**: 검색과 생성을 분리하여 각각 평가하세요
- **적절한 지표 선택**: 시스템 특성에 맞는 평가 지표를 선택하세요
- **자동화**: 평가 파이프라인을 자동화하여 지속적인 모니터링을 구축하세요
- **A/B 테스트**: 다양한 기법을 비교 실험하여 최적의 조합을 찾으세요
- **사용자 피드백**: 자동 평가와 함께 실제 사용자 피드백을 수집하세요

### 🔗 다음 단계

Day 3에서 RAG 평가와 고급 기법을 익혔다면, 다음 내용들을 학습해보세요:

- **프로덕션 배포**: 실서비스를 위한 RAG 시스템 배포 전략
- **성능 최적화**: 지연 시간 최소화 및 처리량 최대화 기법
- **도메인 특화**: 특정 도메인에 최적화된 RAG 시스템 구축
- **멀티모달**: 텍스트 외 이미지, 음성 등을 포함한 RAG 시스템

---

## 📚 참고 자료

- [RAGAS Documentation](https://docs.ragas.io/)
- [ranx Library](https://github.com/AmenRa/ranx)
- [LangChain Evaluation](https://python.langchain.com/docs/guides/evaluation/)
- [Langfuse Evaluation](https://langfuse.com/docs/scores/model-based-evals) 