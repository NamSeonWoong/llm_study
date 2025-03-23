# Ollama와 ChromaDB를 활용한 RAG 시스템

이 프로젝트는 로컬에서 실행되는 Ollama 모델과 ChromaDB를 활용한 간단한 RAG(Retrieval-Augmented Generation) 시스템을 구현한 예제입니다.

## 특징

- **100% 로컬 동작**: API 키 없이 모든 것이 로컬에서 처리됩니다.
- **Ollama 통합**: 로컬에서 실행되는 LLM 모델을 사용합니다.
- **ChromaDB 벡터 저장소**: 문서 임베딩과 검색에 ChromaDB를 활용합니다.
- **대체 임베딩**: Ollama가 없는 경우 HuggingFace 임베딩으로 자동 전환됩니다.
- **간단한 사용법**: 텍스트 파일을 추가하고 질문하면 됩니다.

## 필수 요구사항

1. **Python 3.8 이상**
2. **Ollama**
   - [Ollama 공식 웹사이트](https://ollama.ai/download)에서 설치
   - 또는 `brew install ollama` (macOS Homebrew)

## 설치 방법

1. **가상환경 생성 및 활성화** (선택사항이지만 권장):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **패키지 설치**:
   ```bash
   pip install -r ollama_requirements.txt
   ```

## 사용 방법

1. **Ollama 서버 실행**:
   ```bash
   ollama serve
   ```

2. **필요한 모델 다운로드** (별도 터미널에서):
   ```bash
   ollama pull llama2  # 또는 다른 모델 (mistral, gemma 등)
   ```

3. **텍스트 데이터 추가**:
   - `data` 디렉토리에 텍스트 파일(.txt)을 추가합니다.
   - 없는 경우 자동으로 생성됩니다.

4. **RAG 시스템 실행**:
   ```bash
   python ollama_rag.py
   ```

5. **질문하기**:
   - 프롬프트에 질문을 입력하면 RAG 시스템이 문서를 검색하고 답변을 생성합니다.
   - 종료하려면 'exit'를 입력하세요.

## 작동 원리

1. **문서 처리**: 텍스트 파일을 작은 청크로 분할합니다.
2. **임베딩 생성**: Ollama 또는 HuggingFace 모델을 사용하여 문서 청크의 임베딩을 생성합니다.
3. **벡터 저장소**: ChromaDB에 임베딩과 원본 텍스트를 저장합니다.
4. **검색**: 사용자 질문의 임베딩을 생성하고 유사한 문서를 검색합니다.
5. **응답 생성**: 검색된 문서와 질문을 Ollama LLM에 전달하여 컨텍스트 기반 응답을 생성합니다.

## 사용자 정의

- **다른 Ollama 모델 사용**:
  - `ollama_rag.py` 파일에서 `model="llama2"` 부분을 원하는 모델로 변경하세요.
  - 예: `model="mistral"`, `model="gemma:2b"` 등

- **임베딩 모델 변경**:
  - Ollama 임베딩이 아닌 HuggingFace 임베딩만 사용하려면 `try-except` 블록을 제거하고 `HuggingFaceEmbeddings`만 사용하도록 수정하세요.

- **OLLAMA_HOST 변경**:
  - 기본값은 `http://localhost:11434`입니다.
  - 환경 변수로 설정하거나 코드에서 직접 변경할 수 있습니다.

## 문제 해결

1. **"command not found: ollama"**:
   - Ollama가 설치되어 있지 않습니다. [Ollama 공식 웹사이트](https://ollama.ai/download)에서 설치하세요.

2. **Ollama 연결 실패**:
   - Ollama 서버가 실행 중인지 확인하세요: `ollama serve`
   - 올바른 URL을 사용하고 있는지 확인하세요 (기본: `http://localhost:11434`)

3. **모델 로드 실패**:
   - 모델이 아직 다운로드되지 않았을 수 있습니다: `ollama pull llama2`
   - 코드의 모델 이름과 다운로드한 모델 이름이 일치하는지 확인하세요.

4. **메모리 오류**:
   - 더 작은 모델을 사용해보세요 (예: llama2:7b 대신 gemma:2b)
   - chunk_size를 줄여보세요 (기본값: 1000)

## 라이선스

MIT 