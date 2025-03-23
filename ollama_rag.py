"""
Ollama와 ChromaDB를 활용한 간단한 RAG 시스템
"""

import os
import argparse
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from prompts.rag_prompt import DEFAULT_RAG_PROMPT, EXPERT_RAG_PROMPT, CONCISE_RAG_PROMPT, create_custom_prompt

# 환경 변수 설정
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# 문서 로드 및 분할 함수
def load_and_split_documents(directory_path="./data") -> List:
    """
    지정된 디렉토리에서 텍스트 문서를 로드하고 청크로 분할합니다.
    """
    # 디렉토리가 없으면 생성
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"'{directory_path}' 디렉토리가 생성되었습니다. 이 디렉토리에 텍스트 파일을 추가하세요.")
        return []
    
    # 디렉토리에서 모든 텍스트 파일 로드
    loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print(f"'{directory_path}' 디렉토리에 텍스트 파일이 없습니다.")
        return []
    
    # 문서를 작은 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"{len(documents)} 문서를 {len(chunks)} 청크로 분할했습니다.")
    
    return chunks

# 벡터 저장소 생성 함수
def create_vector_store(chunks, persist_directory="./chroma_db", model_name="llama3:8b"):
    """
    문서 청크로부터 벡터 저장소를 생성합니다.
    """
    # 임베딩 모델 초기화 - Ollama 또는 로컬 HuggingFace 모델 중 선택
    try:
        print(f"Ollama 임베딩 모델({model_name})을 초기화합니다...")
        embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=OLLAMA_HOST
        )
    except Exception as e:
        print(f"Ollama 임베딩 모델 초기화 실패: {str(e)}")
        print("대체 HuggingFace 임베딩 모델을 사용합니다...")
        # 대체 임베딩 모델로 HuggingFace 사용
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    
    # 벡터 저장소 생성
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # persist() 메서드 호출 제거 - 최신 버전의 Chroma에서는 필요 없음
    # 또는 조건부로 처리
    try:
        # 이전 버전 호환성을 위해 시도
        if hasattr(vector_store, 'persist'):
            vector_store.persist()
    except Exception as e:
        print(f"참고: 벡터 저장소 persist 호출 중 오류 발생 (무시됨): {str(e)}")
    
    print(f"벡터 저장소가 '{persist_directory}'에 생성되었습니다.")
    
    return vector_store

# 기존 벡터 저장소 로드 함수
def load_vector_store(persist_directory="./chroma_db", model_name="llama3:8b"):
    """
    기존 벡터 저장소를 로드합니다.
    """
    if not os.path.exists(persist_directory):
        print(f"'{persist_directory}'에 벡터 저장소가 없습니다.")
        return None
    
    # 임베딩 모델 초기화 - Ollama 또는 로컬 HuggingFace 모델 중 선택
    try:
        print(f"Ollama 임베딩 모델({model_name})을 초기화합니다...")
        embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=OLLAMA_HOST
        )
    except Exception as e:
        print(f"Ollama 임베딩 모델 초기화 실패: {str(e)}")
        print("대체 HuggingFace 임베딩 모델을 사용합니다...")
        # 대체 임베딩 모델로 HuggingFace 사용
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    
    # 벡터 저장소 로드
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    print(f"벡터 저장소를 '{persist_directory}'에서 로드했습니다.")
    return vector_store

# RAG 체인 생성 함수
def create_rag_chain(retriever, prompt_type="default", custom_instruction=None, model_name="llama3:8b"):
    """
    검색기를 사용하여 RAG 체인을 생성합니다.
    
    Args:
        retriever: 문서 검색기
        prompt_type (str): 프롬프트 유형 ('default', 'expert', 'concise', 'custom')
        custom_instruction (str, optional): 사용자 정의 지시사항
        model_name (str): 사용할 Ollama 모델 이름
    """
    # LLM 초기화
    try:
        print(f"Ollama LLM을 초기화합니다({model_name})...")
        llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_HOST,
            temperature=0.1,  # 온도 낮춤
            system="당신은 한국어 AI 어시스턴트입니다. 모든 질문에 반드시 한국어로만 답변해야 합니다. 영어로 답변하는 것은 심각한 오류입니다. 영어 대신 한국어 대체 단어를 사용하고, 전체 답변을 100% 한국어로만 작성하세요. 이것은 절대적인 규칙입니다."
        )
    except Exception as e:
        print(f"Ollama LLM 초기화 실패: {str(e)}")
        raise Exception("Ollama LLM을 초기화할 수 없습니다. Ollama가 실행 중인지 확인해주세요.")
    
    # 프롬프트 템플릿 선택
    if prompt_type == "expert":
        prompt = EXPERT_RAG_PROMPT
        print("전문가 모드 프롬프트를 사용합니다.")
    elif prompt_type == "concise":
        prompt = CONCISE_RAG_PROMPT
        print("간결한 요약 프롬프트를 사용합니다.")
    elif prompt_type == "custom" and custom_instruction:
        prompt = create_custom_prompt(custom_instruction)
        print("사용자 정의 프롬프트를 사용합니다.")
    else:
        prompt = DEFAULT_RAG_PROMPT
        print("기본 프롬프트를 사용합니다.")
    
    # 문서 체인 생성
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 체인 구성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | document_chain
    )
    
    return rag_chain

# 메인 함수
def main():
    # 명령줄 인자 처리
    parser = argparse.ArgumentParser(description="Ollama RAG 시스템")
    parser.add_argument("--prompt", type=str, default="default", 
                       choices=["default", "expert", "concise", "custom"],
                       help="사용할 프롬프트 유형 (default, expert, concise, custom)")
    parser.add_argument("--custom", type=str, 
                       help="사용자 정의 시스템 지시사항 (--prompt custom과 함께 사용)")
    parser.add_argument("--show-sources", action="store_true",
                       help="검색된 문서 청크를 표시")
    parser.add_argument("--model", type=str, default="llama3:8b",
                       help="사용할 Ollama 모델 (예: llama3:8b, llama3:70b, mistral, gemma:2b)")
    args = parser.parse_args()
    
    # 모델 이름 설정
    model_name = args.model
    print(f"사용할 모델: {model_name}")
    
    # Ollama 서버 연결 확인
    print(f"Ollama 서버에 연결 중... ({OLLAMA_HOST})")
    
    # 벡터 저장소 초기화 (이미 존재하면 로드, 없으면 생성)
    chroma_dir = "./chroma_db"
    vector_store = load_vector_store(chroma_dir, model_name)
    
    if vector_store is None:
        # 문서 로드 및 분할
        chunks = load_and_split_documents()
        if chunks:
            vector_store = create_vector_store(chunks, chroma_dir, model_name)
        else:
            print("문서가 없어 벡터 저장소를 생성할 수 없습니다.")
            return
    
    # 검색기 생성 (최대 유사도 점수와 함께 반환)
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3,
            # "fetch_k": 5,  # 더 많은 문서 검토 - 이 옵션은 현재 ChromaDB 버전에서 지원되지 않음
            # "score_threshold": None  # 점수 임계값 없음 - 이 옵션도 현재 ChromaDB 버전에서 지원되지 않음
        }
    )
    
    # 마지막 검색 결과와 점수를 저장하기 위한 변수
    last_retrieved_docs = []
    last_retrieved_scores = []
    
    # RAG 체인 생성
    try:
        rag_chain = create_rag_chain(retriever, args.prompt, args.custom, model_name)
    except Exception as e:
        print(f"RAG 체인 생성 실패: {str(e)}")
        return
    
    # 프롬프트 유형 정보 표시
    print(f"\n프롬프트 유형: {args.prompt}")
    if args.prompt == "custom" and args.custom:
        print(f"사용자 정의 지시사항: {args.custom}")
    
    # 대화형 인터페이스
    print("\n=== Ollama RAG 시스템에 질문하기 (종료하려면 'exit' 입력) ===")
    print("특별 명령어: 'sources' - 마지막 질문에 사용된 문서 청크와 유사도 점수 표시")
    
    while True:
        query = input("\n질문: ")
        
        # 특별 명령어 처리
        if query.lower() == 'exit':
            break
        elif query.lower() in ['sources', '소스', '출처']:
            if last_retrieved_docs:
                print("\n--- 마지막 질문에 사용된 문서 청크 ---")
                for i, (doc, score) in enumerate(zip(last_retrieved_docs, last_retrieved_scores), 1):
                    # 유사도 점수에 따른 관련성 표시
                    relevance = "높음 🟢" if score > 0.7 else "중간 🟡" if score > 0.5 else "낮음 🔴"
                    
                    print(f"\n[청크 {i}] 출처: {doc.metadata.get('source', '알 수 없음')}")
                    print(f"유사도: {score:.4f} (관련성: {relevance})")
                    content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    print(f"내용: {content_preview}")
            else:
                print("\n아직 검색된 문서가 없습니다.")
            continue
        
        # 질문에 대한 응답 생성
        try:
            # 문서 검색 및 유사도 점수 가져오기
            # 단순화된 방식으로 검색 (호환성 문제 해결)
            docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
            
            # 검색 결과와 점수 분리하여 저장
            if docs_with_scores:
                docs, scores = zip(*docs_with_scores)
                last_retrieved_docs = docs
                last_retrieved_scores = scores
            else:
                docs = []
                scores = []
                last_retrieved_docs = []
                last_retrieved_scores = []
            
            # 소스 표시 옵션이 활성화된 경우 검색된 문서와 유사도 점수 표시
            if args.show_sources:
                print("\n--- 검색된 문서 청크 ---")
                if docs_with_scores:
                    for i, (doc, score) in enumerate(docs_with_scores, 1):
                        source = doc.metadata.get('source', '알 수 없음')
                        # 유사도 점수에 따른 관련성 표시
                        relevance = "높음 🟢" if score > 0.7 else "중간 🟡" if score > 0.5 else "낮음 🔴"
                        
                        print(f"\n[청크 {i}] 출처: {source}")
                        print(f"유사도: {score:.4f} (관련성: {relevance})")
                        content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                        print(f"내용: {content_preview}")
                else:
                    print("검색된 문서가 없습니다.")
                print("\n--- 응답 생성 중... ---")
            
            # 생성에 필요한 문서 목록
            if docs:
                # 응답 생성 (기존 체인 대신 직접 컨텍스트와 질문을 전달)
                response = rag_chain.invoke(query)
                
                # 유사도 점수 정보 추가 (모든 점수가 낮을 경우)
                if scores and max(scores) < 0.5:
                    print("\n답변:", response)
                    print("\n⚠️ 참고: 모든 검색된 문서의 유사도 점수가 낮습니다 (최대 {:.2f}). 답변이 부정확할 수 있습니다.".format(max(scores)))
                else:
                    print("\n답변:", response)
            else:
                print("\n컨텍스트에 관련 정보가 없습니다. 이 질문에 답변할 수 없습니다.")
            
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {str(e)}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 