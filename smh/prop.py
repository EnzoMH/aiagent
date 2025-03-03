from fastapi import APIRouter, File, UploadFile, HTTPException, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel,ValidationError, Field, validator
from typing import Optional, Dict, List, Any, AsyncGenerator, Optional, TypedDict, Tuple 
from typing import Literal, ForwardRef, Union, TYPE_CHECKING, Annotated, Union, MutableMapping
from uuid import uuid4
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field, replace
from dotenv import load_dotenv
import os
import google.generativeai as genai
import logging
import re
import io
import olefile
from PyPDF2 import PdfReader
from langchain_teddynote.document_loaders import HWPLoader
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
from google.generativeai.types import GenerationConfig as GeminiConfig
from functools import wraps
import tempfile
import types
from inspect import iscoroutine
from contextlib import asynccontextmanager
import time
# Router 설정
# Router 설정
router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# Gemini 공통 설정 클래스 추가
class GeminiConfig:
    """Gemini 모델 공통 설정"""
    MODEL_NAME = "gemini-1.5-flash"
    DEFAULT_CONFIG = {
        "temperature": 0.1,
        "top_p": 1,
                "top_k": 1,
        "max_output_tokens": 3000,
    }

    @classmethod
    def create_model(cls, config: Optional[Dict[str, Any]] = None):
        """모델 인스턴스 생성"""
        generation_config = config or cls.DEFAULT_CONFIG
        return genai.GenerativeModel(
            model_name=cls.MODEL_NAME,
            generation_config=generation_config
        )

    @classmethod
    async def generate_content(cls, prompt: str, config: Optional[Dict[str, Any]] = None) -> str:
        """비동기 컨텐츠 생성"""
        try:
            model = cls.create_model(config)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: model.generate_content(prompt).text
            )
            return response
        except Exception as e:
            logger.error(f"Gemini 생성 오류: {str(e)}")
            raise

class HWPContentHandler:
    """HWPLoader를 위한 핸들러 클래스입니다.
    
    이 클래스는 바이너리 HWP 컨텐츠를 안전하게 처리하고 텍스트를 추출합니다.
    임시 파일 생성 및 삭제를 자동으로 관리하며, Windows 환경의 파일 잠금 문제도
    적절히 처리합니다.
    """
    
    def __init__(self, content: bytes, logger=None):
        self.content = content
        self.logger = logger or logging.getLogger(__name__)
        
    def process(self) -> str:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.hwp', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(self.content)
            
            hwp_loader = HWPLoader(file_path=temp_path)
            text = hwp_loader.load()[0].page_content
            return text
            
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    del hwp_loader  # 명시적 메모리 해제
                    os.remove(temp_path)
                except Exception as e:
                    self.logger.warning(f"임시 파일 삭제 실패: {str(e)}")

class CustomException(HTTPException):
    """사용자 정의 예외 클래스"""
    def __init__(self, status_code: int, detail: str, error_code: str = None):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code

class APIResponse(BaseModel):
    """표준화된 API 응답 모델"""
    status: str = "success"
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class TaskManager:
    """비동기 작업 관리자"""
    _tasks: Dict[str, asyncio.Task] = {}

    @classmethod
    async def create_task(cls, doc_id: str, coro):
        """새로운 비동기 작업 생성"""
        if doc_id in cls._tasks:
            raise CustomException(
                status_code=400,
                detail="Task already exists",
                error_code="TASK_EXISTS"
            )
        
        cls._tasks[doc_id] = asyncio.create_task(coro)
        try:
            await cls._tasks[doc_id]
        finally:
            del cls._tasks[doc_id]

class Settings(BaseModel):
    """애플리케이션 설정"""
    GEMINI_API_KEY: str
    UPLOAD_DIR: str
    MAX_RETRIES: int = 3
    CHUNK_SIZE: int = 1000
    
    @classmethod
    def load_from_env(cls):
        """환경 변수에서 설정 로드"""
        load_dotenv()
        return cls(
            GEMINI_API_KEY=os.getenv("GEMINI_API_KEY"),
            UPLOAD_DIR=os.getenv("UPLOAD_DIR", "uploads")
        )
settings = Settings.load_from_env()

def error_handler(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationError as e:
            raise CustomException(
                status_code=422,
                detail=str(e),
                error_code="VALIDATION_ERROR"
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise CustomException(
                status_code=500,
                detail="Internal server error",
                error_code="INTERNAL_ERROR"
            )
    return wrapper

@asynccontextmanager
async def lifespan(app: FastAPI):
   """애플리케이션 시작/종료 시 실행될 코드"""
   logger = logging.getLogger(__name__)
   
   try:
       # 시작 시
       os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
       logger.info(f"업로드 디렉토리 생성 완료: {settings.UPLOAD_DIR}")
       logger.info("Gemini API 초기화 시작")
       genai.configure(api_key=settings.GEMINI_API_KEY)
       logger.info("서버 시작 완료")
       
       yield
       
   finally:
       # 종료 시
       logger.info("서버 종료 시작")
       for file in os.listdir(settings.UPLOAD_DIR):
           try:
               os.remove(os.path.join(settings.UPLOAD_DIR, file))
               logger.debug(f"임시 파일 삭제: {file}")
           except Exception as e:
               logger.error(f"파일 삭제 실패: {file}, 오류: {str(e)}")
       logger.info("서버 종료 완료")

# exception handler를 함수로 정의
async def custom_exception_handler(request, exc: CustomException):
    return JSONResponse(
        status_code=exc.status_code,
        content=APIResponse(
            status="error",
            error={
                "code": exc.error_code,
                "detail": exc.detail
            }
        ).dict()
    )

class RFPDocument(BaseModel):
    # 핵심 섹션 정보
    business_overview: Dict[str, str] = Field(
        default_factory=lambda: {
            "start_date": "계약체결일",
            "purpose": "",
            "background": "",
            "period": "",
            "budget": ""
        }
    )
    
    detailed_tasks: Dict[str, str] = Field(
        default_factory=lambda: {
            "content": ""
        }
    )
    
    proposal_requirements: Dict[str, str] = Field(
        default_factory=lambda: {
            "required_documents": "",
            "specifications": ""
        }
    )
    
    evaluation_scores: Dict[str, float] = Field(
        default_factory=lambda: {
            "descriptive": 0.0,  # 기본값 설정
            "price": 0.0        # 기본값 설정
        }
    )
    
    submission_guidelines: Dict[str, str] = Field(
        default_factory=lambda: {
            "deadline": "",
            "method": "",
            "page_limit": ""
        }
    )
    
    # 상태 관리
    status: Dict[str, bool] = Field(
        default_factory=lambda: {
            "file_uploaded": False,
            "sections_extracted": False,
            "table_extracted": False,
            "outline_generated": False,
            "content_generated": False
        }
    )
    
    @validator('evaluation_scores')
    def validate_scores(cls, v):
        """평가 점수 검증 로직"""
        # 필수 키 확인
        required_keys = {'descriptive', 'price'}
        if not all(key in v for key in required_keys):
            raise ValueError("평가 점수에는 'descriptive'와 'price' 항목이 모두 필요합니다")
            
        # 값 범위 확인
        if not (0 <= v['descriptive'] <= 100 and 0 <= v['price'] <= 100):
            raise ValueError("각 평가 점수는 0에서 100 사이의 값이어야 합니다")
            
        # 합계가 0이거나 100인 경우만 허용
        total = sum(v.values())
        if total != 0 and abs(total - 100) > 0.01:  # 부동소수점 오차 고려
            raise ValueError("평가 점수의 합은 0이거나 100이어야 합니다")
            
        return v
    
    def update_section(self, section_name: str, content: Dict[str, Any]):
        if hasattr(self, section_name):
            setattr(self, section_name, content)
            self.status["sections_extracted"] = True
    
    def is_ready_for_outline(self) -> bool:
        return self.status["sections_extracted"] and self.status["table_extracted"]
    
    def get_progress(self) -> Dict[str, Any]:
        completed = sum(1 for v in self.status.values() if v)
        total = len(self.status)
        return {
            "percentage": (completed / total) * 100,
            "completed": [k for k, v in self.status.items() if v],
            "pending": [k for k, v in self.status.items() if not v]
        }

class FileHandler:
    """파일 처리 및 보안 검증을 담당하는 클래스"""
    
    ALLOWED_EXTENSIONS = {'.pdf', '.hwp', '.hwpx'}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 10MB
    
    @classmethod
    async def validate_file(cls, file: UploadFile) -> bytes:
        """파일 검증 및 바이너리 데이터 반환"""
        logger = logging.getLogger(__name__)
        
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            raise CustomException(400, "지원하지 않는 파일 형식")

        content = await file.read()
        if len(content) > cls.MAX_FILE_SIZE:
            raise CustomException(400, "파일 크기 초과")
            
        if not content:
            raise CustomException(400, "빈 파일")

        return content

class DocumentStore:
    _text_store: Dict[str, str] = {}  # 전처리된 텍스트 저장소
    _document_store: Dict[str, RFPDocument] = {}  # 문서 저장소
    _store_lock = asyncio.Lock()

    @classmethod
    async def save_processed_text(cls, doc_id: str, text: str):
        async with cls._store_lock:
            cls._text_store[doc_id] = text

    @classmethod
    async def get_processed_text(cls, doc_id: str) -> str:
        if doc_id not in cls._text_store:
            raise CustomException(404, "전처리된 텍스트를 찾을 수 없습니다")
        return cls._text_store[doc_id]

    @classmethod
    async def save_document(cls, doc_id: str, doc: RFPDocument):
        async with cls._store_lock:
            cls._document_store[doc_id] = doc

    @classmethod
    async def get_document(cls, doc_id: str) -> RFPDocument:
        if doc_id not in cls._document_store:
            raise CustomException(404, "문서를 찾을 수 없습니다")
        return cls._document_store[doc_id]


class GeminiResponseFormatter:
   @staticmethod
   def format_extraction_results(extraction_results: list) -> dict:
       # 결과 정리
       page_limit, duration, scores, tasks = extraction_results
       
       # 결과 포맷팅
       formatted_tasks = GeminiResponseFormatter.format_detailed_tasks(tasks)
       scores_dict = {
           "descriptive": float(scores[0]) if isinstance(scores, tuple) else 0.0,  
           "price": float(scores[1]) if isinstance(scores, tuple) else 0.0
       }
       
       return {
           "page_limit": str(page_limit) if not isinstance(page_limit, Exception) else "",
           "duration": duration if not isinstance(duration, Exception) else "",
           "scores": scores_dict,
           "tasks": formatted_tasks
       }

   @staticmethod 
   def format_detailed_tasks(text: str) -> str:
       if isinstance(text, Exception) or not text:
           return ""
           
       formatted = []
       for line in text.split("\n"):
           line = line.strip()
           if re.match(r'^\d+\.', line):
               formatted.append(f"\n{line}")
           elif line.startswith('***'):
               title = line.replace('*', '').split(':', 1)[0]
               formatted.append(f"\n  • {title}:")
               if ':' in line:
                   content = line.split(':', 1)[1].strip()
                   formatted.append(f"    {content}")
           elif line:
               formatted.append(f"    {line}")
               
       return "\n".join(formatted)
    
# 문서 저장소 
class DocumentProcessor:
    """
    문서 처리 클래스
    - 파일 처리는 동기 방식으로 처리 (CPU/IO 작업)
    - Gemini API 호출만 비동기로 처리
    """
    def __init__(self):
        self.logger = self._setup_logger()
        self.chunk_size = 5000
        # PyMuPDF 사용 가능 여부 확인
        try:
            import fitz
            self.fitz_lib = fitz
        except ImportError:
            self.fitz_lib = None
            self.logger.warning("PyMuPDF를 사용할 수 없습니다. PyPDF2를 사용합니다.")

    def _setup_logger(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger

    def process_file(self, content: bytes, file_type: str) -> Dict[str, Any]:
        """
        파일을 처리하고 텍스트를 추출합니다.
        반환값을 단순화하여 추출된 텍스트만 반환합니다.
        """
        try:
            self.logger.info(f"파일 처리 시작 - 타입: {file_type}")
            
            if file_type not in ["pdf", "hwp", "hwpx"]:
                raise ValueError(f"지원하지 않는 파일 형식: {file_type}")

            # 1. 텍스트 추출
            text = self._process_pdf(content) if file_type == "pdf" else self._process_hwp(content)
            
            # 2. 텍스트 전처리만 수행
            processed_text = self._clean_text(text)
            self.logger.info(f"전처리된 텍스트 길이: {len(processed_text)}")
            
            # 3. 단순화된 결과 반환
            return {
                'processed_text': processed_text
            }

        except Exception as e:
            self.logger.error(f"파일 처리 중 오류: {str(e)}")
            raise

    def _process_pdf(self, content: bytes) -> str:
        """PDF 파일 처리"""
        self.logger.info("PDF 처리 시작")
        pdf_stream = io.BytesIO(content)
        text = ""

        try:
            # 1. fitz(PyMuPDF) 시도
            if self.fitz_lib:
                self.logger.debug("fitz(PyMuPDF) 사용 시도")
                try:
                    doc = self.fitz_lib.open(stream=pdf_stream, filetype="pdf")
                    self.logger.info(f"PDF 페이지 수: {len(doc)}")
                    
                    pages_text = []
                    for i, page in enumerate(doc):
                        self.logger.debug(f"페이지 {i+1} 처리 중...")
                        text_content = page.get_text().strip()
                        if text_content:
                            pages_text.append(text_content)
                        else:
                            self.logger.warning(f"페이지 {i+1} 기본 추출 실패, 블록 추출 시도")
                            blocks = page.get_text("blocks")
                            block_text = " ".join(block[4] for block in blocks if isinstance(block[4], str))
                            pages_text.append(block_text)
                    
                    text = "\n".join(filter(None, pages_text))
                    doc.close()
                    
                except Exception as e:
                    self.logger.warning(f"fitz 처리 실패, PyPDF2 시도: {str(e)}")
                    text = ""

            # 2. PyPDF2 fallback
            if not text.strip():
                self.logger.info("PyPDF2 사용 시도")
                reader = PdfReader(pdf_stream)
                extracted_texts = []
                
                for i, page in enumerate(reader.pages):
                    self.logger.debug(f"PyPDF2 페이지 {i+1} 처리 중...")
                    page_text = page.extract_text().strip()
                    if page_text:
                        extracted_texts.append(page_text)
                        
                text = "\n".join(extracted_texts)

            return text

        except Exception as e:
            self.logger.error(f"PDF 처리 실패: {str(e)}")
            raise

    def _process_hwp(self, content: bytes) -> str:
        """HWP 파일을 처리하여 텍스트를 추출합니다."""
        self.logger.info("HWP 파일 처리 시작")
        
        # HWPContentHandler를 사용하여 텍스트 추출
        handler = HWPContentHandler(content, logger=self.logger)
        text = handler.process()
        return text

    def _clean_text(self, text: str) -> str:
        """텍스트 전처리"""
        # 공백 정리

        text = re.sub(r'\s+', ' ', text)
        
        # 빈 줄 정리

        text = re.sub(r'\n\s*\n', '\n\n', text)

        
        # 특수문자 처리

        text = re.sub(r'[^\w\s\n가-힣]+', ' ', text)

        return text.strip()

    def _create_chunks(self, text: str) -> List[str]:
        """문서 청킹"""
        # 섹션 단위로 분리       
        sections = re.split(r'\n(?=[\d.]+\s+[가-힣])', text)
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            if len(current_chunk) + len(section) < self.chunk_size:
                current_chunk += section
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = section
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _clean_text(self, text: str) -> str:
        """
        텍스트 전처리를 수행합니다. HWPLoader에서 이미 기본적인 전처리를 수행했지만,
        RFP 문서 분석에 특화된 추가 전처리를 수행합니다.
        
        Args:
            text (str): HWPLoader에서 추출된 원본 텍스트
            
        Returns:
            str: 전처리된 텍스트
        """
        try:
            # 불필요한 공백 제거
            text = re.sub(r'\s+', ' ', text.strip())
            
            # 특수문자 정규화 (HWP 특유의 문자 처리)
            text = text.replace('ȃ', '')  # HWP 특수 문자 제거
            text = text.replace('䊖', '')  # HWP 특수 문자 제거
            text = text.replace('䠶', '')  # HWP 특수 문자 제거
            
            # 목차 번호 형식 통일
            text = re.sub(r'Ⅰ', '1.', text)
            text = re.sub(r'Ⅱ', '2.', text)
            text = re.sub(r'Ⅲ', '3.', text)
            text = re.sub(r'Ⅳ', '4.', text)
            text = re.sub(r'Ⅴ', '5.', text)
            
            # 문단 구분자 정규화
            text = re.sub(r'\n\s*\n', '\n', text)
            
            return text
            
        except Exception as e:
            self.logger.error(f"텍스트 전처리 중 오류: {str(e)}")
            return text  # 오류 발생시 원본 반환

    async def extract_sections(self, chunks: List[str]) -> str:
        """
        Gemini API를 사용하여 RFP 문서의 섹션들을 추출합니다.
        
        이 메서드는 다음과 같은 과정으로 작동합니다:
        1. 텍스트 전처리: HWP 특수문자 제거 및 포맷 정규화 
        2. Gemini API 호출: 구조화된 프롬프트로 섹션 정보 추출
        3. 응답 반환: 라우터에서 JSON 변환 및 검증 수행
        """
        try:
            self.logger.info("=== 섹션 추출 시작 ===")
            self.logger.info(f"청크 개수: {len(chunks)}")
            self.logger.info(f"전체 청크 크기: {sum(len(chunk) for chunk in chunks)} 바이트")

            for i, chunk in enumerate(chunks, 1):
                self.logger.info(f"=== 청크 {i} 처리 시작 ===")
                
                # 텍스트 전처리 수행
                cleaned_chunk = self._clean_text(chunk)
                self.logger.info(f"청크 {i} 전처리 후 크기: {len(cleaned_chunk)} 바이트")

                prompt = f"""
                아래 제안요청서(RFP) 텍스트를 분석하여 정확히 다음의 섹션들에 대한 정보를 추출해주세요.
                모든 섹션에 대해 반드시 응답해주시기 바랍니다.

                1. business_overview (사업 개요):
                - 사업의 목적 (purpose)
                - 사업의 배경 (background)
                - 사업 기간 (period)
                - 예산 정보 (budget)

                2. detailed_tasks (제안 요청 사항항):
                - 제안 요청 사항들을 포함하여 상세히 설명

                3. proposal_requirements (제안 요청 내용):
                - 제안서 작성 시 필수 포함 사항
                - 주요 요구사항 명세

                4. evaluation_scores (평가 점수):
                - descriptive (기술평가 점수)
                - price (가격평가 점수)
                * 반드시 합계가 100이 되어야 함

                5. submission_guidelines (제출 지침):
                - deadline (제출 기한)
                - method (제출 방법)
                - page_limit (페이지 제한)

                분석할 텍스트:
                {cleaned_chunk}
                """
                
                self.logger.info(f"Gemini API 호출 시작 (청크 {i})")
                start_time = time.time()
                
                try:
                    async with asyncio.timeout(600):
                        response = await GeminiConfig.generate_content(prompt)
                        duration = time.time() - start_time
                        self.logger.info(f"Gemini API 응답 시간: {duration:.2f}초")

                        if response:
                            self.logger.info(f"청크 {i} 유효한 응답 수신 (길이: {len(response)}바이트)")
                            return response
                        else:
                            self.logger.warning(f"청크 {i} 빈 응답 수신")

                except asyncio.TimeoutError:
                    self.logger.error(f"청크 {i} 처리 시간 초과 (30초)")
                    continue
                except Exception as e:
                    self.logger.error(f"청크 {i} 처리 중 오류 발생: {str(e)}")
                    continue

                self.logger.warning(f"청크 {i}에서 유효한 응답을 받지 못함")

            self.logger.warning("=== 모든 청크 처리 실패 ===")
            return ""

        except Exception as e:
            self.logger.error(f"섹션 추출 중 치명적 오류 발생: {str(e)}", exc_info=True)
            raise

    def _merge_sections(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """섹션들을 하나의 일관된 구조로 병합합니다."""
        try:
            self.logger.info("merge_sections 메서드 시작")
            
            # 기본 구조 생성
            normalized = {
                "business_overview": self._normalize_business_overview(
                    sections.get("business_overview", {})
                ),
                "detailed_tasks": {
                    "content": sections.get("detailed_tasks", {}).get("content", "")
                },
                "proposal_requirements": {
                    "required_documents": sections.get("proposal_requirements", {}).get("required_documents", ""),
                    "specifications": sections.get("proposal_requirements", {}).get("specifications", "")
                },
                "evaluation_scores": sections.get("evaluation_scores", {
                    "descriptive": 0.0,
                    "price": 0.0
                }),
                "submission_guidelines": {
                    "deadline": sections.get("submission_guidelines", {}).get("deadline", ""),
                    "method": sections.get("submission_guidelines", {}).get("method", ""),
                    "page_limit": sections.get("submission_guidelines", {}).get("page_limit", "")
                }
            }
            
            self.logger.info(f"정규화된 섹션 구조: {normalized}")
            return normalized
            
        except Exception as e:
            self.logger.error(f"섹션 병합 중 오류: {str(e)}", exc_info=True)
            return self._create_default_analysis()

    def _normalize_business_overview(self, data: Dict[str, str]) -> Dict[str, str]:
        """비즈니스 개요 섹션을 정규화합니다."""
        normalized = {
            "purpose": data.get("purpose", ""),
            "background": data.get("background", ""),
            "period": data.get("period", ""),
            "budget": data.get("budget", ""),
            "start_date": "계약체결일"  # 기본값 설정
        }
        return normalized
        
    def gemini_response_to_json(self, response: str) -> Dict[str, Any]:
        try:
            self.logger.info("=== Gemini 응답 파싱 시작 ===")
            self.logger.info(f"원본 응답 길이: {len(response)}")
            
            sections = {
                "business_overview": {},
                "detailed_tasks": "",
                "proposal_requirements": "",
                "evaluation_scores": {},
                "submission_guidelines": {}
            }
            
            current_section = None
            section_content = []
            
            # 응답 구조 파악을 위한 로깅
            for line in response.split('\n'):
                line = line.strip()
                if not line: continue
                
                # 섹션 전환 감지와 로깅
                if any(f"{i}. {key}" in line for i, key in enumerate(sections.keys(), 1)):
                    self.logger.info(f"섹션 전환 감지: {line}")
                    
                    if current_section and section_content:
                        self.logger.info(f"이전 섹션 '{current_section}' 처리 시작")
                        sections[current_section] = self._parse_section_content('\n'.join(section_content))
                        self.logger.info(f"이전 섹션 처리 결과: {type(sections[current_section])}")
                    
                    current_section = next(key for key in sections.keys() if key in line.lower())
                    self.logger.info(f"현재 섹션 설정: {current_section}")
                    section_content = []
                elif current_section:
                    section_content.append(line)
                    
            # 마지막 섹션 처리와 로깅
            if current_section and section_content:
                self.logger.info(f"마지막 섹션 '{current_section}' 처리")
                sections[current_section] = self._parse_section_content('\n'.join(section_content))
                
            self.logger.info("=== 섹션 처리 완료 ===")
            return sections
                
        except Exception as e:
            self.logger.error(f"응답 변환 실패: {str(e)}", exc_info=True)  # 스택 트레이스 추가
            return self._create_default_analysis()
        
    def _parse_markdown_field(self, line: str) -> Tuple[str, str]:
        """마크다운 형식의 필드를 키와 값으로 분리합니다.
        
        예시: "** purpose:** 사업 목적" -> ("purpose", "사업 목적")
        """
        try:
            # "**키:**값" 형식 파싱
            if '**' in line and ':' in line:
                parts = line.split(':**', 1)  # 최대 1번만 분할
                if len(parts) == 2:
                    key = parts[0].strip('* ').strip('**')
                    # 영문 키 추출 (있는 경우)
                    if '(' in key and ')' in key:
                        key = key.split('(')[-1].split(')')[0].strip()
                    value = parts[1].strip()
                    return key, value
        except Exception as e:
            self.logger.warning(f"필드 파싱 실패: {str(e)}, 라인: {line}")
        return None, None

    def _parse_section_content(self, content: str) -> Dict[str, Any]:
        """섹션 내용을 파싱하여 구조화된 데이터로 변환합니다."""
        try:
            if not content.strip():
                return {}

            # 비즈니스 개요 섹션 파싱
            if any(marker in content for marker in ['purpose', 'background', 'period', 'budget']):
                # 기본 구조 정의
                result = {
                    "start_date": "계약체결일",  # 항상 기본값 설정
                    "purpose": "",
                    "background": "",
                    "period": "",
                    "budget": ""
                }
                
                current_field = None
                current_content = []
                
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    key, value = self._parse_markdown_field(line)
                    if key and key in result:  # 키가 result에 있는 경우만 처리
                        if current_field:  # 이전 필드 저장
                            result[current_field] = ' '.join(current_content).strip()
                        current_field = key
                        current_content = [value] if value else []
                    elif current_field:  # 현재 필드에 내용 추가
                        current_content.append(line)
                
                # 마지막 필드 처리
                if current_field and current_content:
                    result[current_field] = ' '.join(current_content).strip()
                
                return result

            # 세부 과업 내용 섹션
            elif 'detailed_tasks' in content.lower():
                return {"content": content.strip()}

            # 제안 요청 내용 섹션
            elif 'proposal_requirements' in content.lower():
                return self._parse_requirements_section(content)

            # 평가 점수 섹션
            elif any(marker in content.lower() for marker in ['descriptive', 'price', '평가 점수']):
                return self._parse_evaluation_scores(content)

            # 제출 지침 섹션
            elif 'submission_guidelines' in content.lower():
                return self._parse_submission_guidelines(content)

            return {}
                
        except Exception as e:
            self.logger.error(f"섹션 파싱 중 오류: {str(e)}", exc_info=True)
            return {}
        
    def _parse_requirements_section(self, content: str) -> Dict[str, str]:
        """제안 요청 내용 섹션을 파싱합니다."""
        required_docs = []
        specifications = []
        current_section = required_docs
        
        for line in content.split('\n'):
            line = line.strip()
            if '필수 포함 사항' in line:
                current_section = required_docs
            elif '요구사항 명세' in line:
                current_section = specifications
            elif line:
                current_section.append(line)
                
        return {
            "required_documents": ' '.join(required_docs).strip(),
            "specifications": ' '.join(specifications).strip()
        }

    def _parse_evaluation_scores(self, content: str) -> Dict[str, float]:
        """평가 점수 섹션을 파싱합니다."""
        try:
            scores = {"descriptive": 0.0, "price": 0.0}
            
            for line in content.split('\n'):
                if 'descriptive' in line.lower() or '기술평가' in line:
                    _, value = self._parse_markdown_field(line)
                    if value:
                        scores['descriptive'] = float(''.join(filter(str.isdigit, value)))
                elif 'price' in line.lower() or '가격평가' in line:
                    _, value = self._parse_markdown_field(line)
                    if value:
                        scores['price'] = float(''.join(filter(str.isdigit, value)))
                        
            return scores
        except Exception as e:
            self.logger.warning(f"평가 점수 파싱 실패: {str(e)}")
            return {"descriptive": 0.0, "price": 0.0}
        
    def _create_default_analysis(self) -> Dict[str, Any]:
        """기본 분석 결과 생성"""
        return {
        "business_overview": {
            "start_date": "",
            "purpose": "",
            "background": "",
            "period": "",
            "budget": ""
        },
        "detailed_tasks": {
            "content": ""  # 주요 과업 내용을 담는 필드
        },
        "proposal_requirements": {
            "required_documents": "",  # 필수 제출 서류
            "specifications": ""       # 요구사항 명세
        },
        "evaluation_scores": {
            "descriptive": 90,
            "price": 10
        },
        "submission_guidelines": {
            "deadline": "",
            "method": "",
            "page_limit": ""
            }
        }

class ContentGenerator:
    def __init__(self):
        self.template_cache = {}
        self.logger = logging.info(__name__)
        
    async def generate_outline(self, doc: RFPDocument) -> Dict[str, Dict]:
        """목차 구조 생성"""
        if not doc.is_ready_for_outline():
            raise ValueError("Required sections not extracted yet")
            
        page_limit = int(doc.submission_guidelines.get("page_limit", 0))
        prompt = self._create_outline_prompt(doc, page_limit)
        return await GeminiConfig.generate_content(prompt)

    async def generate_content(self, section: str, context: Dict[str, Any]) -> str:
        """섹션별 내용 생성"""
        template = await self._get_template(section)
        prompt = template.format(**context)
        return await GeminiConfig.generate_content(prompt)

    async def _get_template(self, section: str) -> str:
        """섹션별 템플릿 로드 또는 생성"""
        if section not in self.template_cache:
            prompt = f"Create a detailed template for generating {section} content"
            self.template_cache[section] = await GeminiConfig.generate_content(prompt)
        return self.template_cache[section]
    
    async def _create_outline_prompt(self, doc: RFPDocument, page_limit: int) -> str:
        return f"""
        제안서의 목차를 구조에 맞춰 추출해주세요
        제약사항:
        - 총 페이지 수: {page_limit}쪽
        - 사업내용: {doc.business_overview}
        - 세부과업: {doc.detailed_tasks}
        - 평가기준: {doc.evaluation_scores}

        JSON 구조:
        {{
            "chapters": [{{
                "title": "대목차명",
                "page": "할당 페이지 수",
                "sub_chapters": [{{
                    "title": "중목차명",
                    "page": "할당 페이지 수",
                    "sections": [{{
                        "title": "소목차명", 
                        "page": "할당 페이지 수"
                    }}]
                }}]
            }}],
            "total_pages": {page_limit}
        }}
        """

class ExtractionService:
    """텍스트에서 주요 정보를 추출하는 서비스"""
    
    logger = logging.getLogger(__name__)
    
    @classmethod
    def _log_extraction_attempt(cls, method: str, text_length: int):
        """추출 시도 로깅"""
        cls.logger.info(f"=== {method} 추출 시작 ===")
        cls.logger.info(f"입력 텍스트 길이: {text_length}")
        
    @classmethod
    def _log_extraction_result(cls, method: str, result: Any):
        """추출 결과 로깅"""
        cls.logger.info(f"=== {method} 추출 결과 ===")
        cls.logger.info(f"추출된 값: {result}")
    
    @staticmethod
    async def extract_page_limit(text: str) -> int:
        """페이지 제한 추출"""
        try:
            ExtractionService._log_extraction_attempt("페이지 제한", len(text))
            
            prompt = """
            다음 텍스트에서 정성제안서(또는 기술제안서)의 전체 페이지 제한을 찾아 숫자만 반환해주세요.
            ** 반드시 제안서 작성방법 ** 이라는 상위항목에서 이를 찾아주세요. 종종 제안서 페이지 제한은 없는 경우도 있습니다
            그런경우에는 '0'으로 출력해주세요요
            페이지 수는 '쪽', '페이지', '매' 등으로 표현될 수 있습니다.

            텍스트:
            {text}

            숫자만 반환해주세요.
            """
            
            start_time = time.time()
            response = await GeminiConfig.generate_content(
                prompt.format(text=text),
                {"temperature": 0.1, "max_output_tokens": 100}
            )
            duration = time.time() - start_time
            
            ExtractionService.logger.info(f"Gemini API 응답 시간: {duration:.2f}초")
            ExtractionService.logger.info(f"원본 응답: {response}")
            
            result = int(''.join(filter(str.isdigit, response.strip())))
            ExtractionService._log_extraction_result("페이지 제한", result)
            
            return result
            
        except Exception as e:
            ExtractionService.logger.error(f"페이지 제한 추출 실패: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def extract_duration(text: str) -> str:
        """사업기간 추출"""
        prompt = """
        다음 텍스트에서 전체 사업기간을 찾아 기간만 반환해주세요.
        예시 형식: "계약체결일로부터 12개월" 또는 "2025.1.1 ~ 2025.12.31"

        텍스트:
        {text}
        """
        
        response = await GeminiConfig.generate_content(
            prompt.format(text=text),
            {
                "temperature": 0.1,
                "max_output_tokens": 200
            }
        )
        return response.strip()

    @staticmethod
    async def extract_score(text: str) -> Tuple[float, float]:
        """평가점수 추출 (기술평가, 가격평가)"""
        try:
            ExtractionService._log_extraction_attempt("평가점수", len(text))
            
            prompt = """
            제안요청서에서 평가배점/기준 관련 정보를 추출해주세요:

            1. '평가방법', '평가기준', '평가배점' 등의 섹션을 찾아주세요
            2. 기술/제안평가(정성)와 가격평가(정량) 배점을 숫자로 추출
            3. 총점은 반드시 100점이어야 합니다
            4. 만약 세부 평가항목이 있다면 함께 추출해주세요

            텍스트: {text}

            points 형식으로 "기술: 숫자, 가격: 숫자" 반환
            """
            
            start_time = time.time()
            response = await GeminiConfig.generate_content(
                prompt.format(text=text),
                {"temperature": 0.1, "max_output_tokens": 150}
            )
            duration = time.time() - start_time
            
            ExtractionService.logger.info(f"Gemini API 응답 시간: {duration:.2f}초")
            ExtractionService.logger.info(f"원본 응답: {response}")
            
            scores = response.strip().split(',')
            result = (float(scores[0]), float(scores[1]))
            
            # 점수 검증
            if not (0 <= result[0] <= 100 and 0 <= result[1] <= 100):
                raise ValueError("평가 점수가 유효한 범위(0-100)를 벗어났습니다")
                
            if abs(result[0] + result[1] - 100) > 0.01:  # 부동소수점 오차 고려
                raise ValueError("평가 점수의 합이 100이 되어야 합니다")
                
            ExtractionService._log_extraction_result("평가점수", result)
            return result
            
        except Exception as e:
            ExtractionService.logger.error(f"평가점수 추출 실패: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def extract_detail_task(text: str) -> str:
        """세부과업지침 텍스트 추출"""
        try:
            ExtractionService._log_extraction_attempt("세부과업지침", len(text))
            
            prompt = """
            아래 제안요청서의 '제안 요청사항', '세부 과업내용', '과업 수행 지침' 등의 섹션을 찾아 상세히 추출해주세요:

            1. 먼저 해당 섹션의 시작 부분을 찾아주세요 (보통 로마자 숫자나 1,2,3 등으로 시작)
            2. 해당 섹션의 전체 구조와 계층을 파악해주세요
            3. 상세 내용을 계층구조로 정리해주세요
            4. 향후 제안서 작성에 필요한 핵심 요구사항을 추출해주세요

            텍스트: {cleaned_chunk}
            """
            
            start_time = time.time()
            ExtractionService.logger.info("Gemini API 호출 시작")
            
            response = await GeminiConfig.generate_content(
                prompt.format(text=text),
                {
                    "temperature": 0.2,
                    "max_output_tokens": 1000
                }
            )
            
            duration = time.time() - start_time
            ExtractionService.logger.info(f"Gemini API 응답 시간: {duration:.2f}초")
            
            # 응답 처리 및 검증
            result = response.strip()
            ExtractionService.logger.info(f"추출된 텍스트 길이: {len(result)}")
            
            if not result:
                ExtractionService.logger.warning("추출된 세부과업 내용이 비어있습니다")
                return ""
            
            ExtractionService._log_extraction_result("세부과업지침", f"길이: {len(result)} 문자")
            return result
            
        except Exception as e:
            ExtractionService.logger.error(f"세부과업 추출 실패: {str(e)}", exc_info=True)
            raise
    
class ContentGenerationService:
    """제안서 콘텐츠 생성 서비스"""
    
    @staticmethod
    async def proposal_planning(context: Dict[str, Any]) -> Dict[str, Any]:
        """제안서 기획"""
        prompt = """
        다음 정보를 바탕으로 제안서 기획 방향을 제시해주세요:
        - 사업개요: {business_overview}
        - 세부과업: {detailed_tasks}
        - 평가기준: {evaluation_criteria}

        다음 구조로 JSON 응답을 생성해주세요:
        {
            "main_concept": "핵심 컨셉",
            "key_messages": ["핵심 메시지1", "핵심 메시지2", ...],
            "differentiation_points": ["차별화 포인트1", "차별화 포인트2", ...]
        }
        """
        
        return await GeminiConfig.generate_content(
            prompt.format(**context),
            {
                "temperature": 0.7,
                "max_output_tokens": 1000
            }
        )

    @staticmethod
    async def toc_generator(context: Dict[str, Any]) -> Dict[str, Any]:
        """목차 생성"""
        prompt = """
        다음 정보를 바탕으로 제안서 목차를 생성해주세요:
        - 페이지 제한: {page_limit}쪽
        - 주요 과업: {tasks}
        - 평가 기준: {evaluation_criteria}

        다음 규칙을 준수해주세요:
        1. 대목차-중목차-소목차 구조로 생성
        2. 소목차별로 페이지 수 배분
        3. 전체 페이지 수가 {page_limit}쪽과 일치해야 함
        """
        
        return await GeminiConfig.generate_content(
            prompt.format(**context),
            {
                "temperature": 0.3,
                "max_output_tokens": 2000
            }
        )

    @staticmethod
    async def generate_headcopy(context: Dict[str, Any]) -> List[str]:
        """헤드카피 생성"""
        prompt = """
        다음 정보를 바탕으로 각 소목차에 맞는 헤드카피를 생성해주세요:
        - 핵심 컨셉: {main_concept}
        - 소목차: {sub_chapters}

        각 헤드카피는:
        1. 10단어 이내의 임팩트 있는 문구
        2. 소목차의 핵심 내용을 함축
        3. 차별화된 강점 부각
        """
        
        return await GeminiConfig.generate_content(
            prompt.format(**context),
            {
                "temperature": 0.8,
                "max_output_tokens": 1000
            }
        )

    @staticmethod
    async def generate_subcopy(context: Dict[str, Any]) -> List[str]:
        """서브카피 생성"""
        prompt = """
        다음 정보를 바탕으로 각 헤드카피를 보완하는 서브카피를 생성해주세요:
        - 헤드카피: {headcopy}
        - 소목차 내용: {content}

        각 서브카피는:
        1. 20단어 이내의 설명적 문구
        2. 헤드카피의 의미를 구체화
        3. 실현 가능한 가치 제시
        """
        
        return await GeminiConfig.generate_content(
            prompt.format(**context),
            {
                "temperature": 0.7,
                "max_output_tokens": 1500
            }
        )

    @staticmethod
    async def generate_content(context: Dict[str, Any]) -> str:
        """본문 내용 생성"""
        prompt = """
        다음 정보를 바탕으로 제안서 본문을 생성해주세요:
        - 소목차: {chapter_title}
        - 할당 페이지: {page_count}쪽
        - 헤드카피: {headcopy}
        - 서브카피: {subcopy}
        - 핵심 메시지: {key_messages}

        다음 사항을 준수해주세요:
        1. 페이지당 약 400단어 기준
        2. 구체적인 실행 방안과 기대효과 포함
        3. 차별화 포인트 강조
        4. 평가기준에 부합하는 내용 구성
        """
        
        return await GeminiConfig.generate_content(
            prompt.format(**context),
            {
                "temperature": 0.4,
                "max_output_tokens": 3000
            }
        )

# 라우터 구현
@router.post("/upload")
@error_handler
async def upload_file(file: UploadFile) -> APIResponse:
    logger.info(f"파일 업로드 시작: {file.filename}")

    try:
        # 파일 검증 및 읽기
        content = await FileHandler.validate_file(file)
        logger.info(f"파일 검증 완료: {file.filename}")
        logger.info(f"content 길이: {len(content)}")

        
    
        file_type = file.filename.lower().split('.')[-1]
        logger.info(f"파일 확장자: {file_type}")
        logger.info(f"파일 확장자 길이: {len(file_type)}")

        # 문서 처리 - 텍스트 추출만
        processor = DocumentProcessor()
        processing_result = processor.process_file(content, file_type)
        
        logger.info(f"processing_result 길이: {len(processing_result)}")

        
        # 기본 문서 객체 생성 (간소화된 RFPDocument)
        doc_id = str(uuid4())
        doc = RFPDocument(
        business_overview={
            "start_date": "계약체결일",
            "purpose": "",
            "background": "",
            "period": "",
            "budget": ""
        },
        detailed_tasks={
            "content": ""
        },
        proposal_requirements={
            "required_documents": "",
            "specifications": ""
        },
        evaluation_scores={
            "descriptive": 0.0,
            "price": 0.0
        },
        submission_guidelines={
            "deadline": "",
            "method": "",
            "page_limit": ""
            }
        )
        doc.status["file_uploaded"] = True  # 업로드 상태 업데이트
        doc_store =  DocumentStore()
        await doc_store.save_document(doc_id, doc)
        
        # 전처리된 텍스트 별도 저장
        await doc_store.save_processed_text(doc_id, processing_result['processed_text'])

        return APIResponse(
            status="success",
            message="파일 업로드가 완료되었습니다. 섹션 분석은 별도로 요청해주세요.",
            data={"document_id": doc_id}
        )

    except Exception as e:
        logger.error(f"파일 처리 오류: {str(e)}")
        raise
   
@router.post("/extract-sections/{doc_id}")
@error_handler
async def extract_sections(doc_id: str) -> APIResponse:
    """
    문서에서 섹션 정보를 추출하고 구조화하는 엔드포인트입니다.
    
    이 함수는 다음과 같은 단계로 작동합니다:
    1. 문서 검증 및 전처리된 텍스트 로드
    2. Gemini를 통한 섹션 정보 추출 및 파싱
    3. 추가 정보(페이지 제한, 기간, 점수, 과업) 추출
    4. 모든 정보를 정규화된 형식으로 통합
    5. 문서 객체 생성 및 저장
    """
    try:
        # 1. 문서 및 텍스트 검증
        doc_store = DocumentStore()
        doc = await doc_store.get_document(doc_id)
        if not doc.status["file_uploaded"]:
            raise CustomException(400, "파일 업로드가 필요합니다")
            
        processed_text = await doc_store.get_processed_text(doc_id)
        logger.info(f"문서 텍스트 로드 완료: {len(processed_text)} 바이트")

        # 2. 섹션 정보 추출 및 파싱
        processor = DocumentProcessor()
        raw_sections = await processor.extract_sections([processed_text])
        sections = processor.gemini_response_to_json(raw_sections)
        logger.info("섹션 정보 파싱 완료")

        # 3. 추가 정보 추출 - 병렬 처리로 성능 최적화
        extraction_results = await asyncio.gather(
            ExtractionService.extract_page_limit(processed_text),
            ExtractionService.extract_duration(processed_text),
            ExtractionService.extract_score(processed_text),
            ExtractionService.extract_detail_task(processed_text),
            return_exceptions=True
        )
        
        # 추출 결과 처리 로깅 추가
        logger.info("=== 추출 결과 처리 시작 ===")

        # 페이지 제한 처리 로깅
        logger.info(f"페이지 제한 추출 결과 타입: {type(extraction_results[0])}")
        if isinstance(extraction_results[0], Exception):
            logger.warning(f"페이지 제한 추출 실패: {str(extraction_results[0])}")
        else:
            logger.info(f"페이지 제한 추출 성공: {extraction_results[0]}")
        page_limit = (
            extraction_results[0] if not isinstance(extraction_results[0], Exception) else ""
        )
        
        # 기간 처리 로깅
        logger.info(f"사업 기간 추출 결과 타입: {type(extraction_results[1])}")
        if isinstance(extraction_results[1], Exception):
            logger.warning(f"사업 기간 추출 실패: {str(extraction_results[1])}")
        else:
            logger.info(f"사업 기간 추출 성공: {extraction_results[1]}")
        duration = (
            extraction_results[1] if not isinstance(extraction_results[1], Exception) else ""
        )
        
        # 평가 점수 처리 로깅
        logger.info(f"평가 점수 추출 결과 타입: {type(extraction_results[2])}")
        if isinstance(extraction_results[2], Exception):
            logger.warning(f"평가 점수 추출 실패: {str(extraction_results[2])}")
        else:
            logger.info(f"평가 점수 추출 성공: {extraction_results[2]}")
        scores = (
            extraction_results[2] if not isinstance(extraction_results[2], Exception) 
            else (0.0, 0.0)
        )
        
        # 세부 과업 처리 로깅
        logger.info(f"세부 과업 추출 결과 타입: {type(extraction_results[3])}")
        if isinstance(extraction_results[3], Exception):
            logger.warning(f"세부 과업 추출 실패: {str(extraction_results[3])}")
        else:
            logger.info(f"세부 과업 추출 성공: {len(extraction_results[3])} 바이트")
        tasks = (
            extraction_results[3] if not isinstance(extraction_results[3], Exception) else ""
        )

        # 평가 점수 구성 로깅
        logger.info("=== 평가 점수 구성 시작 ===")
        evaluation_scores = {
            "descriptive": float(scores[0]) if isinstance(scores, tuple) else 0.0,
            "price": float(scores[1]) if isinstance(scores, tuple) else 0.0
        }
        logger.info(f"최종 평가 점수 구성: {evaluation_scores}")
        
        # detailed_tasks 병합 로직 개선
        detailed_content = tasks if tasks else sections.get("detailed_tasks", {}).get("content", "")
        if not detailed_content:
            logger.warning("detailed_tasks 내용이 비어있습니다")

        # 4. 데이터 정규화
        normalized_sections = {
            "business_overview": {
                "start_date": "계약체결일",
                "purpose": sections.get("business_overview", {}).get("purpose", ""),
                "background": sections.get("business_overview", {}).get("background", ""),
                "period": duration or sections.get("business_overview", {}).get("period", ""),
                "budget": sections.get("business_overview", {}).get("budget", "")
            },
            "detailed_tasks": {
               "content": detailed_content  # 수정된 부분
            },
            "proposal_requirements": {
                "required_documents": sections.get("proposal_requirements", {}).get("required_documents", ""),
                "specifications": sections.get("proposal_requirements", {}).get("specifications", "")
            },
            "evaluation_scores": evaluation_scores,
            "submission_guidelines": {
                "deadline": sections.get("submission_guidelines", {}).get("deadline", ""),
                "method": sections.get("submission_guidelines", {}).get("method", ""),
                "page_limit": str(page_limit)
            }
        }
        logger.info("=== 정규화된 섹션 데이터 검증 시작 ===")
        logger.info(f"정규화된 섹션 구조: {json.dumps(normalized_sections, indent=2)}")    
        logger.info("정규화된 데이터 구조 생성 완료")
        
        # 필수 필드 존재 여부 확인
        required_fields = {
            "business_overview", "detailed_tasks", "proposal_requirements",
            "evaluation_scores", "submission_guidelines"
        }
        missing_fields = required_fields - set(normalized_sections.keys())
        if missing_fields:
            logger.error(f"누락된 필수 필드: {missing_fields}")

        # business_overview 필드 검증
        business_fields = {"start_date", "purpose", "background", "period", "budget"}
        missing_business = business_fields - set(normalized_sections["business_overview"].keys())
        if missing_business:
            logger.warning(f"business_overview 누락 필드: {missing_business}")

        logger.info("=== RFPDocument 생성 전 데이터 타입 검증 ===")

        # evaluation_scores 타입 검증
        scores = normalized_sections["evaluation_scores"]
        logger.info(f"평가 점수 데이터: {scores}")
        if not all(isinstance(v, float) for v in scores.values()):
            logger.error("평가 점수가 float 타입이 아닙니다")

        # submission_guidelines page_limit 타입 검증
        page_limit = normalized_sections["submission_guidelines"]["page_limit"]
        logger.info(f"페이지 제한 값: {page_limit} (타입: {type(page_limit)})")
        
        
        # RFPDocument 생성 및 상태 검증
        doc = RFPDocument(**normalized_sections)
        # 상태 업데이트 전 현재 상태 로깅
        logger.info(f"상태 업데이트 전: {doc.status}")
        
        doc.status.update({
            "file_uploaded": True,
            "sections_extracted": True,
            "table_extracted": True
        })
        logger.info("=== RFPDocument 생성 및 상태 검증 ===")
        logger.info(f"문서 상태: {doc.status}")
        logger.info(f"문서 진행률: {doc.get_progress()}")
        
        logger.info(f"상태 업데이트 후: {doc.status}")
        
        
        
        # 저장 전 최종 검증
        logger.info("=== 최종 데이터 검증 ===")
        logger.info(f"평가 점수: {doc.evaluation_scores}")
        logger.info(f"business_overview: {doc.business_overview}")
        logger.info(f"submission_guidelines: {doc.submission_guidelines}")
        
        # 문서 저장
        await doc_store.save_document(doc_id, doc)
        logger.info("문서 저장 완료")

        return APIResponse(
            status="success",
            message="섹션 추출이 완료되었습니다",
            data=normalized_sections
        )

    except ValidationError as e:
        logger.error(f"데이터 검증 오류: {str(e)}")
        raise CustomException(
            status_code=422,
            detail="섹션 데이터 구조가 올바르지 않습니다",
            error_code="VALIDATION_ERROR"
        )
    except Exception as e:
        logger.error(f"섹션 추출 프로세스 실패: {str(e)}")
        raise CustomException(
            status_code=500,
            detail="섹션 추출 중 오류가 발생했습니다",
            error_code="EXTRACTION_ERROR"
        )

@router.post("/plan/{doc_id}")
@error_handler
async def plan_proposal(doc_id: str) -> APIResponse:
   doc_store = DocumentStore()
   doc = await doc_store.get_document(doc_id)
   if not doc.is_ready_for_outline():
       raise CustomException(400, "섹션 분석이 필요합니다")
   
   planning_result = await ContentGenerationService.proposal_planning({
       "business_overview": doc.business_overview,
       "detailed_tasks": doc.detailed_tasks,
       "evaluation_criteria": doc.evaluation_scores
   })
   
   doc.status["outline_generated"] = True
   await doc_store.save_document(doc_id, doc)
   
   return APIResponse(status="success", data=planning_result)



@router.post("/generate-toc/{doc_id}")
@error_handler
async def generate_toc(doc_id: str) -> APIResponse:
   doc_store = DocumentStore()  
   doc = await doc_store.get_document(doc_id)
   
   if not doc.submission_guidelines.get("page_limit"):
       raise CustomException(400, "페이지 제한이 필요합니다")
   
   toc = await ContentGenerationService.toc_generator({
       "page_limit": doc.submission_guidelines["page_limit"],
       "tasks": doc.detailed_tasks,
       "evaluation_criteria": doc.evaluation_scores
   })
   
   doc.outline = {"table_of_contents": toc}
   doc.status["outline_generated"] = True
   await doc_store.save_document(doc_id, doc)
   
   return APIResponse(status="success", data=toc)


@router.post("/generate-content/{doc_id}")
@error_handler
async def generate_section_content(doc_id: str, section_info: Dict[str, Any]) -> APIResponse:
   doc_store = DocumentStore()  
   doc = await doc_store.get_document(doc_id)
   if not doc.status["outline_generated"]:
       raise CustomException(400, "목차 생성이 필요합니다")
   
   result = await generate_section_with_retries(doc, section_info)
   
   doc.status["content_generated"] = True
   await doc_store.save_document(doc_id, doc)
   
   return APIResponse(status="success", data=result)

async def generate_section_with_retries(doc: RFPDocument, section_info: Dict[str, Any]):
    """재시도 로직이 포함된 섹션 생성"""
    retries = 3
    for attempt in range(retries):
        try:
            headcopy = await ContentGenerationService.generate_headcopy({
                "main_concept": doc.outline["main_concept"],
                "sub_chapters": section_info
            })
            
            subcopy = await ContentGenerationService.generate_subcopy({
                "headcopy": headcopy,
                "content": section_info
            })
            
            content = await ContentGenerationService.generate_content({
                "chapter_title": section_info["title"],
                "page_count": section_info["page_count"],
                "headcopy": headcopy,
                "subcopy": subcopy,
                "key_messages": doc.outline.get("key_messages", [])
            })
            
            return {
                "headcopy": headcopy,
                "subcopy": subcopy,
                "content": content
            }
            
        except Exception as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1 * (attempt + 1))


__all__ = ['router', 'lifespan', 'custom_exception_handler']