"""
LangChain 기반 VectorDatabase 데모

ChromaDB와 다양한 임베딩 모델(OpenAI, HuggingFace)을 사용한
벡터 데이터베이스 기능을 데모합니다.
"""

import os
from langchain_core.documents import Document

from src.database.vector_db import (
    VectorDatabase, 
    ChromaDBConfig,
    create_openai_vector_db,
    create_huggingface_vector_db
)


def demo_huggingface_vector_db():
    """HuggingFace 임베딩을 사용한 VectorDatabase 데모"""
    print("\n🤗 === HuggingFace Embeddings 데모 ===")
    
    # 1. 설정 및 초기화
    vector_db = create_huggingface_vector_db(
        collection_name="hf_demo_collection",
        persist_directory="./demo_chromadb_hf",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(f"✅ VectorDatabase 초기화 완료")
    
    # 2. 문서 추가
    documents = [
        Document(
            page_content="LangChain은 대규모 언어 모델(LLM) 애플리케이션을 구축하기 위한 프레임워크입니다.",
            metadata={"source": "docs", "category": "framework", "language": "ko"}
        ),
        Document(
            page_content="ChromaDB는 AI 애플리케이션을 위한 오픈소스 임베딩 데이터베이스입니다.",
            metadata={"source": "docs", "category": "database", "language": "ko"}
        ),
        Document(
            page_content="HuggingFace Transformers provides easy-to-use machine learning models.",
            metadata={"source": "docs", "category": "ml", "language": "en"}
        ),
        Document(
            page_content="벡터 검색은 의미론적 유사성을 기반으로 관련 문서를 찾는 기술입니다.",
            metadata={"source": "docs", "category": "search", "language": "ko"}
        )
    ]
    
    doc_ids = vector_db.add_documents(documents)
    print(f"📄 {len(documents)}개 문서 추가 완료: {doc_ids}")
    
    # 3. 컬렉션 정보 확인
    info = vector_db.get_collection_info()
    print(f"📊 컬렉션 정보: {info}")
    
    # 4. 유사도 검색
    print("\n🔍 === 검색 테스트 ===")
    
    queries = [
        "프레임워크에 대해 알려주세요",
        "데이터베이스 관련 정보",
        "machine learning models"
    ]
    
    for query in queries:
        print(f"\n쿼리: '{query}'")
        results = vector_db.similarity_search(query, k=2)
        
        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc.page_content[:50]}... (메타데이터: {doc.metadata})")
    
    # 5. 점수와 함께 검색
    print("\n🎯 === 점수 포함 검색 ===")
    query = "벡터 검색"
    scored_results = vector_db.similarity_search_with_score(query, k=3)
    
    print(f"쿼리: '{query}'")
    for doc, score in scored_results:
        print(f"  점수: {score:.3f} | {doc.page_content[:60]}...")
    
    # 6. Retriever로 변환하여 사용
    print("\n🔗 === Retriever 변환 테스트 ===")
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    
    retrieved_docs = retriever.invoke("AI 기술")
    print(f"Retriever 결과 ({len(retrieved_docs)}개):")
    for i, doc in enumerate(retrieved_docs):
        print(f"  {i+1}. {doc.page_content[:50]}...")
    
    # 7. 정리
    vector_db.close()
    print("\n✅ HuggingFace VectorDatabase 데모 완료!")


def demo_openai_vector_db():
    """OpenAI 임베딩을 사용한 VectorDatabase 데모 (API 키 필요)"""
    print("\n🤖 === OpenAI Embeddings 데모 ===")
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   OpenAI 데모를 건너뜁니다.")
        return
    
    try:
        # 1. OpenAI VectorDatabase 생성
        vector_db = create_openai_vector_db(
            collection_name="openai_demo_collection",
            persist_directory="./demo_chromadb_openai",
            openai_model="text-embedding-3-large"
        )
        
        print(f"✅ OpenAI VectorDatabase 초기화 완료")
        
        # 2. 텍스트 추가 (간단한 예제)
        texts = [
            "OpenAI는 인공지능 연구를 수행하는 회사입니다.",
            "GPT 모델은 텍스트 생성에 특화된 언어 모델입니다.",
            "임베딩 모델은 텍스트를 벡터로 변환합니다."
        ]
        
        metadatas = [
            {"source": "wiki", "topic": "company"},
            {"source": "wiki", "topic": "model"},
            {"source": "wiki", "topic": "embedding"}
        ]
        
        text_ids = vector_db.add_texts(texts, metadatas=metadatas)
        print(f"📝 {len(texts)}개 텍스트 추가 완료: {text_ids}")
        
        # 3. 검색 테스트
        query = "AI 회사는 어디인가요?"
        results = vector_db.similarity_search(query, k=2)
        
        print(f"\n🔍 검색 결과 (쿼리: '{query}'):")
        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc.page_content}")
        
        # 4. 정리
        vector_db.close()
        print("\n✅ OpenAI VectorDatabase 데모 완료!")
        
    except Exception as e:
        print(f"❌ OpenAI 데모 실행 중 오류: {e}")


def main():
    """메인 데모 함수"""
    print("🚀 === LangChain VectorDatabase 데모 시작 ===")
    print("이 데모는 다음을 보여줍니다:")
    print("  • HuggingFace 임베딩을 사용한 벡터 검색")
    print("  • OpenAI 임베딩을 사용한 벡터 검색 (API 키 필요)")
    print("  • 문서/텍스트 추가, 검색, Retriever 변환")
    
    # 1. HuggingFace 데모 (항상 실행)
    demo_huggingface_vector_db()
    
    # 2. OpenAI 데모 (API 키가 있는 경우만)
    demo_openai_vector_db()
    
    print("\n🎉 === 모든 데모 완료 ===")
    print("LangChain 기반 VectorDatabase가 성공적으로 작동합니다!")


if __name__ == "__main__":
    main() 