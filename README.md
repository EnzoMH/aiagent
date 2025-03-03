주어진 프로젝트 구조를 바탕으로 README.md 파일을 작성해 드리겠습니다. 다음은 GitHub에 바로 사용할 수 있는 README.md 파일입니다:

```markdown
# AI 제안서 어시스턴트

AI 제안서 어시스턴트는 제안서 작성과 RFP 분석을 돕는 AI 기반 웹 애플리케이션입니다. 이 애플리케이션은 다양한 LLM(Large Language Model)을 활용하여 문서 분석, 제안서 작성 전략 수립, 평가 항목 분석 및 대응 방안을 제공합니다.

## 주요 기능

- 다양한 AI 모델 지원 (Claude, Gemini, Meta/LLaMA)
- RFP 문서 분석 및 해석
- 제안서 작성 전략 수립
- 웹소켓 기반 실시간 채팅 인터페이스
- 파일 업로드 및 분석 (PDF, HWP, HWPX, DOC, DOCX)
- 마크다운 형식 지원 (Gemini 모델)

## 기술 스택

- **백엔드**: FastAPI, Python 3.11
- **프론트엔드**: HTML, CSS, JavaScript
- **AI 모델**:
  - Claude API (Anthropic)
  - Gemini API (Google)
  - LLaMA 3.2 (Meta, 로컬 추론)
- **추가 기능**:
  - WebSocket 실시간 통신
  - 비동기 처리 (asyncio)
  - 파일 처리 (PyPDF2, HWPLoader)

## 프로젝트 구조

```
smh/                         # 프로젝트 루트 디렉토리
├── static/                   # 정적 파일 디렉토리
│   ├── css/                  # CSS 스타일시트
│   │   ├── style.css        # 기본 채팅창 스타일
│   │   └── prop.css         # 제안서 생성 화면 스타일
│   ├── js/                   # JavaScript 파일
│   │   ├── main.js          # 메인 채팅 인터페이스 스크립트
│   │   └── prop.js          # 제안서 생성 스크립트
│   ├── image/                # 이미지 리소스
│   ├── home.html            # 메인 채팅 인터페이스 HTML
│   └── prop.html            # 제안서 생성 인터페이스 HTML
│
├── utils/                    # 유틸리티 모듈
│   ├── __init__.py
│   ├── dc.py                # 문서 처리 유틸리티
│   ├── pg.py                # 제안서 생성 유틸리티
│   ├── agent.py             # AI 에이전트 컨트롤러
│   └── ps.py                # 제안서 서버 유틸리티
│
├── llama-korean/             # LLaMA 3.2 한국어 모델 (Bllossom)
├── EVE-Korean-Instruction/   # LLaMA 3.1 한국어 모델 (EVE)
├── main.py                   # FastAPI 메인 애플리케이션
├── prop.py                   # 제안서 생성 백엔드
├── .env                      # 환경 변수 설정 파일
└── requirements.txt          # 의존성 패키지 목록
```

## 설치 방법

1. 레포지토리 클론
   ```bash
   git clone https://github.com/yourusername/ai-proposal-assistant.git
   cd ai-proposal-assistant
   ```

2. 가상 환경 설정
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 의존성 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

4. 환경 변수 설정
   - `.env` 파일을 생성하고 다음 항목 설정:
     ```
     CLAUDE_API_KEY=your_claude_api_key
     GEMINI_API_KEY=your_gemini_api_key
     ```

5. 로컬 LLM 모델 설정 (선택 사항)
   - LLaMA 3.2 기반 한국어 모델 다운로드
   - 모델 파일을 `llama-korean/` 디렉토리에 위치

## 실행 방법

```bash
uvicorn main:app --host 0.0.0.0 --port 8005 --reload
```

웹 브라우저에서 `http://localhost:8005`로 접속하여 애플리케이션을 사용할 수 있습니다.

## 사용 방법

1. **채팅 모드**: 메인 페이지에서 AI 모델을 선택하고 질문이나 요청을 입력합니다.
   - Meta: 빠른 응답 (로컬 LLM)
   - Claude: 상세한 분석
   - Gemini: 균형잡힌 성능

2. **문서 분석**: RFP나 기타 문서를 업로드하여 AI로 분석할 수 있습니다.
   - 지원 파일 형식: PDF, HWP, HWPX, DOC, DOCX
   - 파일 드래그 앤 드롭 또는 파일 선택 버튼 이용

3. **제안서 작성**: '새 제안서 작성' 버튼을 클릭하여 제안서 작성 인터페이스로 이동합니다.

## 구현된 기능

- 실시간 대화형 인터페이스 (WebSocket)
- 다중 AI 모델 통합
- 파일 업로드 및 분석
- 마크다운 렌더링 (Gemini 모델)
- 대화 컨텍스트 유지

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 기여 방법

프로젝트에 기여하고 싶으시다면:
1. 이슈를 생성하거나 기존 이슈를 확인합니다.
2. 변경사항에 대한 Pull Request를 생성합니다.
3. 코드 리뷰 후 병합됩니다.

## 연락처

문의사항이 있으시면 [이메일 주소]로 연락 주시거나 GitHub 이슈를 통해 문의해 주세요.
```

이 README.md는 프로젝트의 구조와 기능을 잘 설명하고 있습니다. 필요에 따라 다음 정보를 추가로 제공해 주시면 더 상세하게 업데이트할 수 있습니다:

1. 프로젝트의 GitHub 레포지토리 URL
2. 연락처 정보 (이메일 등)
3. 라이센스 정보가 다르다면 수정
4. 특별한 설치 요구사항이 있다면 추가
