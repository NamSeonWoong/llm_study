"""
Ollama와 ChromaDB를 활용한 간단한 RAG 시스템
"""

import os
import argparse
import wandb
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.documents import Document

# 환경 변수 설정
load_dotenv()

# Ollama 서버 설정
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# wandb 초기화 함수
def init_wandb(project_name="ollama-rag", prompt_type="default", model_name="llama3:8b"):
    """
    Weights & Biases 초기화
    
    Args:
        project_name (str): wandb 프로젝트 이름
        prompt_type (str): 프롬프트 유형
        model_name (str): 모델 이름
    """
    try:
        # wandb 버전 확인
        print(f"🔍 wandb 버전: {wandb.__version__}")
        
        # wandb API 키가 환경 변수에 있는지 확인
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            print("⚠️ WANDB_API_KEY가 설정되지 않았습니다. wandb 로깅이 비활성화됩니다.")
            
            # .env 파일 존재 여부 확인
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
            if os.path.exists(env_path):
                print(f"✓ .env 파일이 존재합니다: {env_path}")
                with open(env_path, 'r') as f:
                    env_content = f.read()
                    if "WANDB_API_KEY" in env_content:
                        print("✓ .env 파일에 WANDB_API_KEY가 포함되어 있습니다.")
                    else:
                        print("✗ .env 파일에 WANDB_API_KEY가 없습니다.")
            else:
                print(f"✗ .env 파일이 존재하지 않습니다: {env_path}")
            
            return None
        else:
            # API 키 첫 4자리만 출력 (보안상 이유로)
            api_key_preview = api_key[:4] + "..." if len(api_key) > 4 else "유효하지 않은 형식"
            print(f"✓ WANDB_API_KEY가 설정되어 있습니다 (시작: {api_key_preview})")
        
        # 실험 이름 생성 (모델 + 프롬프트 타입)
        experiment_name = f"{model_name}_{prompt_type}"
        print(f"🔄 wandb 실험 이름: {experiment_name}")
        
        # wandb 초기화
        print("🔄 wandb 초기화 중...")
        run = wandb.init(
            project=project_name,
            name=experiment_name,
            config={
                "prompt_type": prompt_type,
                "model_name": model_name,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "retriever_k": 1,
                "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
            },
            # 같은 이름의 실험이 있으면 재개(resume)
            resume="allow"
        )
        
        print(f"✓ wandb 초기화 성공: {run.id}")
        return run
    except Exception as e:
        print(f"✗ wandb 초기화 오류: {str(e)}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return None

# 문서 로드 및 분할 함수
def load_documents(directory: str) -> List:
    """문서 로드"""
    documents = []
    
    # 디버깅 정보
    print(f"📂 문서를 로드할 디렉토리: {os.path.abspath(directory)}")
    
    try:
        files = os.listdir(directory)
        print(f"📄 발견된 파일: {files}")
        
        for filename in files:
            file_path = os.path.join(directory, filename)
            print(f"   확인 중: {filename} (is_file: {os.path.isfile(file_path)}, endswith .txt: {filename.endswith('.txt')})")
            
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        print(f"   ✓ 파일 읽기 성공: {filename} (길이: {len(text)} 자)")
                        # Document 객체로 변환하여 추가
                        documents.append(Document(page_content=text, metadata={"source": filename}))
                except Exception as e:
                    print(f"   ✗ 파일 읽기 실패: {filename}, 오류: {str(e)}")
    except Exception as e:
        print(f"디렉토리 탐색 중 오류 발생: {str(e)}")
    
    print(f"🔢 로드된 문서 수: {len(documents)}")
    return documents

def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """문서 분할"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

# 벡터 저장소 생성 함수
def create_vector_store(documents: List, persist_directory: str = "./chroma_db") -> Chroma:
    """벡터 저장소 생성"""
    embeddings = OllamaEmbeddings(
        model="llama3:8b",
        base_url=OLLAMA_HOST
    )
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    try:
        vector_store.persist()
    except AttributeError:
        print("Warning: persist() 메서드가 지원되지 않습니다. 벡터 저장소는 메모리에만 저장됩니다.")
    
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
def create_rag_chain(retriever, prompt_type="default", custom_instruction=None, model_name='exaone:latest'):
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
    parser = argparse.ArgumentParser(description="Ollama RAG 시스템")
    parser.add_argument("--model", default="exaone3.5:latest", help="사용할 Ollama 모델")
    parser.add_argument("--prompt", default="default", choices=["default", "expert", "concise", "custom"],
                      help="프롬프트 유형 선택")
    parser.add_argument("--custom", help="사용자 정의 프롬프트 지시사항")
    args = parser.parse_args()

    # 현재 스크립트의 절대 경로 가져오기
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 데이터 디렉토리 경로 설정 (절대 경로)
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
    
    # wandb 초기화
    wandb_run = init_wandb(prompt_type=args.prompt, model_name=args.model)

    # 데이터 디렉토리 확인
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"'{DATA_DIR}' 디렉토리를 생성했습니다. 텍스트 파일을 추가해주세요.")
        return

    # 문서 로드
    try:
        documents = load_documents(DATA_DIR)
        if not documents:
            print(f"'{DATA_DIR}' 디렉토리에 텍스트 파일이 없습니다.")
            return
    except Exception as e:
        print(f"문서 로드 실패: {str(e)}")
        return

    # 문서 분할
    try:
        chunks = split_documents(documents)
        print(f"문서를 {len(chunks)}개의 청크로 분할했습니다.")
    except Exception as e:
        print(f"문서 분할 실패: {str(e)}")
        return

    # 벡터 저장소 생성 또는 로드
    try:
        if os.path.exists(CHROMA_DIR):
            print(f"기존 벡터 저장소를 로드합니다({CHROMA_DIR})...")
            vector_store = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=OllamaEmbeddings(
                    model=args.model,
                    base_url=OLLAMA_HOST
                )
            )
        else:
            print(f"새로운 벡터 저장소를 생성합니다({CHROMA_DIR})...")
            vector_store = create_vector_store(chunks, persist_directory=CHROMA_DIR)
    except Exception as e:
        print(f"벡터 저장소 생성/로드 실패: {str(e)}")
        return
    
    # 검색기 생성 (최대 유사도 점수와 함께 반환)
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 1,
        }
    )
    
    # 마지막 검색 결과와 점수를 저장하기 위한 변수
    last_retrieved_docs = []
    last_retrieved_scores = []
    
    # RAG 체인 생성
    try:
        rag_chain = create_rag_chain(retriever, args.prompt, args.custom, args.model)
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
    
    # 질문 카운터 및 기록 변수 초기화
    question_count = 0
    response_times = []
    last_query = "없음"
    last_response = "없음"
    
    while True:
        query = input("\n질문: ")
        
        # 특별 명령어 처리
        if query.lower() == 'exit':
            break

        # 질문 카운터 증가
        question_count += 1
        
        # 질문에 대한 응답 생성
        try:
            # 문서 검색 및 유사도 점수 가져오기
            docs_with_scores = vector_store.similarity_search_with_score(query, k=1)
            
            # 검색 결과와 점수 분리하여 저장
            docs = [doc for doc, _ in docs_with_scores]
            scores = [score for _, score in docs_with_scores]
            
            # 마지막 검색 결과와 점수 업데이트
            last_retrieved_docs = docs
            last_retrieved_scores = scores
            
            if docs:
                # 응답 생성 시간 측정
                start_time = datetime.now()
                response = rag_chain.invoke(query)
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                # 마지막 질문과 응답 업데이트
                last_query = query
                last_response = response
                response_times.append(response_time)
                
                # wandb 로깅
                if wandb_run:
                    try:
                        print(f"🔄 wandb에 로깅 시도 중... (질문 #{question_count})")
                        # 질문 그룹 설정 (모델_프롬프트 조합으로 그룹화)
                        group_id = f"{args.model}_{args.prompt}"
                        
                        wandb.log({
                            "step": question_count,
                            "group": group_id,
                            "query": query,
                            "response": response,
                            "response_time": response_time,
                            "avg_similarity_score": sum(scores) / len(scores),
                            "max_similarity_score": max(scores),
                            "min_similarity_score": min(scores),
                            "num_retrieved_docs": len(docs)
                        })
                        
                        # 개별 질문-응답 쌍을 테이블로 저장
                        qa_table = wandb.Table(columns=["질문", "응답", "응답시간(초)", "평균유사도"])
                        qa_table.add_data(query, response, response_time, sum(scores) / len(scores))
                        wandb.log({f"QA_pair_{question_count}": qa_table})
                        
                        print(f"✓ wandb 로깅 성공")
                    except Exception as e:
                        print(f"✗ wandb 로깅 오류: {str(e)}")
                        import traceback
                        print(f"상세 오류: {traceback.format_exc()}")
                else:
                    print("! wandb_run이 None입니다. 로깅을 건너뜁니다.")
                
                # 유사도 점수 정보 추가 (모든 점수가 낮을 경우)
                if all(score < 0.5 for score in scores):
                    print("\n⚠️ 주의: 검색된 문서의 유사도가 모두 낮습니다. 답변의 정확성을 신중하게 검토해주세요.")
                
                # 답변 출력
                print(f"\n답변: {response}")
                print(f"유사도 점수: {scores}")
                
                # 모든 질문에 대해 관련 문서와 유사도 자동 출력
                print("\n--- 참조된 문서 정보 ---")
                for i, (doc, score) in enumerate(zip(docs, scores), 1):
                    relevance = "높음 🟢" if score > 0.7 else "중간 🟡" if score > 0.5 else "낮음 🔴"
                    print(f"[문서 {i}] 출처: {doc.metadata.get('source', '알 수 없음')}")
                    print(f"유사도: {score:.4f} (관련성: {relevance})")
            else:
                print("\n컨텍스트에 관련 정보가 없습니다. 이 질문에 답변할 수 없습니다.")
            
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {str(e)}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            
            # 오류 로깅
            if wandb_run:
                wandb.log({
                    "error": str(e),
                    "error_traceback": traceback.format_exc()
                })

    # 실험 요약 정보 기록 및 종료
    if wandb_run and question_count > 0:
        try:
            print(f"🔄 wandb에 실험 요약 정보 기록 중...")
            # 모든 질문-응답 쌍을 요약한 테이블 생성
            summary_table = wandb.Table(columns=["총 질문 수", "평균 응답 시간", "마지막 질문", "마지막 응답"])
            
            # 실험 전체 통계 저장
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            summary_table.add_data(question_count, avg_response_time, last_query, last_response)
            wandb.log({"experiment_summary": summary_table})
            
            print(f"✓ wandb 실험 요약 정보 기록 성공")
            
            print(f"\n실험 요약:")
            print(f"총 질문 수: {question_count}")
            print(f"평균 응답 시간: {avg_response_time:.4f}초")
        except Exception as e:
            print(f"✗ wandb 실험 요약 기록 오류: {str(e)}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
    
    # wandb 종료
    if wandb_run:
        try:
            print("🔄 wandb 종료 중...")
            wandb.finish()
            print("✓ wandb 종료 성공")
        except Exception as e:
            print(f"✗ wandb 종료 오류: {str(e)}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 