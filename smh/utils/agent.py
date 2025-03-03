#agent.py

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import os
import json
import asyncio
from abc import ABC, abstractmethod
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# 공통 예외 클래스
# 공통 예외 클래스들 위치에 추가
class GenerationError(Exception):
    """AI 생성 관련 기본 예외"""
    pass

class InvalidResponseError(GenerationError):
    """응답 형식 오류"""
    pass

class ModelError(GenerationError):
    """모델 실행 오류"""
    pass

@dataclass
class GenerationConfig:
    """Gemini 모델 생성 설정"""
    temperature: float = 0.9
    top_p: float = 1
    top_k: int = 1
    max_output_tokens: int = 3000
    
    @classmethod
    def for_head(cls) -> 'GenerationConfig':
        return cls(max_output_tokens=150)
        
    @classmethod
    def for_sub(cls) -> 'GenerationConfig':
        return cls(max_output_tokens=300)
        
    @classmethod
    def for_contents(cls) -> 'GenerationConfig':
        return cls(max_output_tokens=4000)

class PromptTemplate(ABC):
    """프롬프트 템플릿 기본 클래스"""
    @abstractmethod
    def create(self, **kwargs) -> str:
        pass

class HeadCopyPrompt(PromptTemplate):
    def create(self, **kwargs) -> str:
        return f"""
        당신은 NewKL의 전문 교육 콘텐츠 카피라이터이자 사업제안서의 기획가입니다.
        다음 섹션에 대한 강력하고 임팩트 있는 헤드카피를 작성해주세요.

        섹션 정보:
        제목: {kwargs.get('title')}
        페이지 수: {kwargs.get('pages')}
        핵심 키워드: {kwargs.get('keywords')}

        RFP 관련 내용:
        {kwargs.get('rfp_data')}

        작성 요구사항:
        1. 교육의 핵심 가치를 함축적으로 전달
        2. 교육 효과를 강조
        3. 차별화 포인트를 부각
        4. 15단어 이내로 간결하게 작성
        5. 학습자의 관심을 끌 수 있는 표현 사용
        """
class SubCopyPrompt(PromptTemplate):
   def create(self, **kwargs) -> str:
       return f"""
       당신은 NewKL의 교육 솔루션 전문가이자 사업제안서의 기획가입니다.
       앞서 작성된 헤드카피를 보완하는 서브카피를 작성해주세요.

       섹션 정보:
       제목: {kwargs.get('title')}
       페이지 수: {kwargs.get('pages')}
       핵심 키워드: {kwargs.get('keywords')}

       작성 요구사항:
       1. 구체적인 교육 솔루션 제시
       2. 실현 가능한 성과 명시
       3. 차별화된 교육 방법론 강조
       4. 2-3문장으로 구성
       5. 정량적 수치나 구체적 사례 포함
       """

class ContentPrompt(PromptTemplate):
   def create(self, **kwargs) -> str:
       return f"""
       당신은 NewKL의 수석 교육 과정 설계자이자 사업제안서의 기획가입니다.
       다음 섹션의 상세 제안 내용을 작성해주세요.

       섹션 정보:
       제목: {kwargs.get('title')}
       페이지 수: {kwargs.get('pages')}
       컨텐츠 유형: {kwargs.get('content_type')}

       RFP 요구사항:
       {kwargs.get('rfp_data')}
       """

class ChatPrompt(PromptTemplate):
   def create(self, **kwargs) -> str:
       base = f"""
       당신은 NewKL의 도우미, NewklBot입니다.
       사용자의 질문에 대해 전문적이고 명확한 답변을 제공해주세요.

       사용자 질문:
       {kwargs.get('question')}
       """
       
       if kwargs.get('context'):
           base += f"\n\n참고 문서:\n{kwargs.get('context')}"
           
       return base

class SectionExtractionPrompt(PromptTemplate):
    def create(self, **kwargs) -> str:
        return f"""
        다음 제안요청서 내용을 분석하여 정확히 다음 JSON 형식으로만 응답해주세요.
        다른 설명이나 마크다운 형식은 포함하지 말아주세요.

        {{
            "사업개요": {{
                "사업명": "",
                "배경": "",
                "목적": "",
                "예산": "",
                "기간": ""
            }},
            "평가정보": {{
                "기술평가": {{
                    "총점": 0,
                    "항목": [],
                    "배점": {{}}
                }},
                "가격평가": {{
                    "총점": 0
                }}
            }},
            "제안범위": [],
            "수행일정": {{
                "착수": "",
                "중간": "",
                "최종": "",
                "세부일정": []
            }}
        }}

        분석할 내용:
        {kwargs.get('content')}
        """


class AgentController:
    """AI 모델 제어 및 컨텐츠 생성 관리"""
    
    def __init__(self):
        self._init_logger()
        self._init_api()
        self._init_model()
        self._init_prompt_templates()
        
    def _init_logger(self) -> None:
        """로거 초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _init_api(self) -> None:
        """API 초기화"""
        try:
            load_dotenv()
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY 없음")
            
            genai.configure(api_key=api_key)
            
        except Exception as e:
            self.logger.error(f"API 초기화 실패: {str(e)}")
            raise ModelError(f"API 초기화 실패: {str(e)}")

    def _init_model(self) -> None:
        """모델 초기화"""
        try:
            self.model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=GenerationConfig().__dict__,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            )
        except Exception as e:
            self.logger.error(f"모델 초기화 실패: {str(e)}")
            raise ModelError(f"모델 초기화 실패: {str(e)}")

    def _init_prompt_templates(self) -> None:
        """프롬프트 템플릿 초기화"""
        self.prompts = {
        'head_copy': HeadCopyPrompt(),
        'section_extraction': SectionExtractionPrompt(),  # 추가
        'sub_copy': SubCopyPrompt(),
        'content': ContentPrompt(),
        'chat': ChatPrompt()
        }

    async def generate_with_recovery(self, 
                               prompt: str, 
                               retry_count: int = 3,
                               fallback_handler: Optional[callable] = None) -> str:
        """재시도 로직이 포함된 생성 메서드"""
        last_error = None
        
        for attempt in range(retry_count):
            try:
                response = await self.model.generate_content_async(prompt)
                if response and response.text:
                    return response.text
                    
            except Exception as e:
                last_error = e
                self.logger.warning(f"생성 시도 {attempt + 1}/{retry_count} 실패: {str(e)}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)

        # 모든 시도 실패시 fallback 처리
        if fallback_handler:
            try:
                return await fallback_handler(prompt)
            except Exception as fb_error:
                self.logger.error(f"폴백 처리 실패: {str(fb_error)}")

        raise GenerationError(f"컨텐츠 생성 실패: {str(last_error)}")

    def parse_json_response(self, text: str) -> Dict[str, Any]:
        try:
            if not text or not text.strip():
                self.logger.error("Empty response received")
                return self._get_default_structure()
                
            cleaned_text = text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]

            # JSON 패턴 찾기
            json_pattern = r'({[\s\S]*})'
            matches = list(re.finditer(json_pattern, cleaned_text))
            
            for match in matches:
                try:
                    json_str = match.group(1)
                    
                    # 1. 특수 문자 처리
                    json_str = json_str.replace('‧', '·')
                    
                    # 2. 줄바꿈 정규화
                    json_str = re.sub(r'\n\s*', ' ', json_str)
                    
                    # 3. 따옴표 처리
                    def normalize_quotes(m):
                        content = m.group(1)
                        # 작은따옴표를 큰따옴표로 변환
                        content = content.replace('"', '\\"').replace("'", '"')
                        return f'"{content}"'
                    
                    # 따옴표로 둘러싸인 내용 처리
                    json_str = re.sub(r'"([^"]*)"', normalize_quotes, json_str)
                    json_str = re.sub(r"'([^']*)'", normalize_quotes, json_str)
                    
                    # 4. 키 문자열화 (한글 포함)
                    json_str = re.sub(r'([{,])\s*([a-zA-Z_가-힣][a-zA-Z0-9_가-힣]*)\s*:', r'\1"\2":', json_str)
                    
                    self.logger.debug(f"Attempting to parse JSON:\n{json_str}")
                    data = json.loads(json_str)
                    return data
                    
                except json.JSONDecodeError as je:
                    self.logger.warning(f"JSON decode error: {str(je)}")
                    continue

            return self._get_default_structure()

        except Exception as e:
            self.logger.error(f"JSON parsing failed: {str(e)}\nInput text: {text[:200]}...")
            return self._get_default_structure()
        
    async def _generate_gemini_response(self, prompt: str, websocket: WebSocket) -> str:
        response = await self.gemini_model.generate_content_async(prompt)
        text = response.text
        
        # 청크 단위로 전송하여 스트리밍 효과 구현
        chunks = self._split_text_into_chunks(text)
        for chunk in chunks:
            await self._send_chunk(websocket, chunk, "gemini")
            await asyncio.sleep(0.01)  # 약간의 지연으로 스트리밍 효과 구현
        
        return text

    def _split_text_into_chunks(self, text: str, chunk_size: int = 10) -> List[str]:
        """텍스트를 청크로 분할"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        
        return chunks

    def _get_default_structure(self) -> Dict[str, Any]:
        """기본 응답 구조 반환"""
        return {
            "사업개요": {
                "사업명": "",
                "배경": "",
                "목적": "",
                "예산": "",
                "기간": ""
            },
            "평가정보": {
                "기술평가": {
                    "총점": 0,
                    "항목": [],
                    "배점": {}
                },
                "가격평가": {
                    "총점": 0
                }
            },
            "제안범위": [],
            "수행일정": {
                "착수": "",
                "중간": "",
                "최종": "",
                "세부일정": []
            }
        }
        
    def _parse_text_fallback(self, text: str) -> Dict[str, Any]:
        """JSON 파싱 실패시 텍스트 기반 파싱 시도"""
        try:
            # 기본 구조 생성
            parsed_data = {
                "사업개요": {},
                "평가정보": {
                    "기술평가": {"총점": 0, "항목": [], "배점": {}},
                    "가격평가": {"총점": 0}
                },
                "제안범위": [],
                "수행일정": {"착수": "", "중간": "", "최종": "", "세부일정": []}
            }
            
            # 텍스트 기반 파싱 로직
            lines = text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 섹션 헤더 감지
                if ':' in line and not line.startswith(' '):
                    current_section = line.split(':')[0].strip()
                    
            return parsed_data
            
        except Exception as e:
            self.logger.error(f"agent.py, _parse_text_fallback함수 ,Fallback 파싱 실패: {str(e)}")
            return {}
        
    def _preprocess_json_string(self, json_str: str) -> str:
        """JSON 문자열 전처리 강화"""
        # 1. 작은따옴표 내의 내용 보존
        preserved_quotes = []
        def preserve_quoted(match):
            preserved_quotes.append(match.group(1))
            return f"__QUOTE_{len(preserved_quotes)-1}__"
        
        # 작은따옴표 내용 임시 저장
        processed = re.sub(r"'([^']*)'", preserve_quoted, json_str)
        
        # 2. 키 문자열화
        processed = re.sub(r'([{,])\s*([a-zA-Z_가-힣][a-zA-Z0-9_가-힣]*)\s*:', r'\1"\2":', processed)
        
        # 3. 저장된 따옴표 내용 복원 (큰따옴표로)
        for i, content in enumerate(preserved_quotes):
            processed = processed.replace(f"__QUOTE_{i}__", f'"{content}"')
        
        return processed

    # 컨텐츠 생성 메서드들
    async def generate_head_copy(self, section_data: Dict[str, Any], rfp_data: Dict[str, Any]) -> Dict[str, Any]:
        """헤드카피 생성"""
        prompt = self.prompts['head_copy'].create(
            title=section_data["title"],
            pages=section_data.get("pages", "미지정"),
            keywords=", ".join(section_data.get("key_points", [])),
            rfp_data=json.dumps(rfp_data, ensure_ascii=False, indent=2)
        )
        
        response = await self.generate_with_recovery(prompt)
        return self.parse_json_response(response)

    async def generate_sub_copy(self, section_data: Dict[str, Any], rfp_data: Dict[str, Any]) -> Dict[str, Any]:
       """서브카피 생성"""
       prompt = self.prompts['sub_copy'].create(
           title=section_data["title"],
           pages=section_data.get("pages", "미지정"), 
           keywords=", ".join(section_data.get("key_points", [])),
           rfp_data=json.dumps(rfp_data, ensure_ascii=False, indent=2)
       )
       
       response = await self.generate_with_recovery(prompt)
       return self.parse_json_response(response)

    async def generate_content_section(self, section_data: Dict[str, Any], rfp_data: Dict[str, Any]) -> Dict[str, Any]:
        """섹션 컨텐츠 생성"""
        prompt = self.prompts['content'].create(
            title=section_data["title"],
            pages=section_data.get("pages", "미지정"),
            content_type=section_data.get("content_type", "text"),
            rfp_data=json.dumps(rfp_data, ensure_ascii=False, indent=2)
        )
        
        response = await self.generate_with_recovery(prompt)
        return self.parse_json_response(response)

    # 챗봇 응답 생성 메서드들
    async def generate_chat_response(self, 
                                    question: str, 
                                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """챗봇 응답 생성"""
        prompt = self.prompts['chat'].create(
            question=question,
            context=json.dumps(context, ensure_ascii=False, indent=2) if context else None
        )
        
        response = await self.generate_with_recovery(prompt)
        return self.parse_json_response(response)

    async def extract_sections(self, text_content: str) -> Dict[str, Any]:
        try:
            chunks = self._split_content(text_content)
            sections = {}
            
            for chunk in chunks:
                # 로깅 추가
                self.logger.info(f"Processing chunk (length: {len(chunk)})")
                
                prompt = self._create_section_extraction_prompt(chunk)
                response = await self.generate_with_recovery(prompt)
                
                # Gemini 응답 로깅
                self.logger.info(f"Gemini Raw Response:\n{response}")
                
                try:
                    chunk_sections = self.parse_json_response(response)
                    sections = self._merge_sections(sections, chunk_sections)
                except Exception as e:
                    self.logger.error(f"JSON parsing error for chunk: {str(e)}")
                    self.logger.error(f"Failed response:\n{response}")
                    continue
                
            if not sections:
                self.logger.warning("agent.py, 섹션이 없습니다, fallback 실행합니다다")
                return await self._fallback_section_extraction("")
                
            return sections
            
        except Exception as e:
            self.logger.error(f"Section extraction failed: {str(e)}")
            raise
        
    def _create_section_extraction_prompt(self, chunk: str) -> str:
        """섹션 추출을 위한 프롬프트 생성"""
        prompt_template = SectionExtractionPrompt()
        return prompt_template.create(content=chunk)

    # 유틸리티 메서드들
    def _split_content(self, text: str, max_tokens: int = 8000) -> List[str]:
        """컨텐츠를 적절한 크기로 분할"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            # 토큰 수 추정
            estimated_tokens = len(para.split()) * 1.5
            
            if current_length + estimated_tokens > max_tokens:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = estimated_tokens
            else:
                current_chunk.append(para)
                current_length += estimated_tokens
                
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks

    async def _fallback_section_extraction(self, prompt: str) -> str:
        """섹션 추출 실패시 기본값 반환"""
        basic_structure = {
            "sections": {
                "overview": "",
                "requirements": [],
                "evaluation": {
                    "criteria": [],
                    "scores": {}
                }
            }
        }
        return json.dumps(basic_structure, ensure_ascii=False)

    # 응답 검증 메서드들
    def _validate_response_structure(self, response: Dict[str, Any], expected_keys: List[str]) -> bool:
        """응답 구조 검증"""
        return all(key in response for key in expected_keys)

    def _validate_content_response(self, response: Dict[str, Any]) -> bool:
        """컨텐츠 응답 검증"""
        expected = ["title", "content", "key_points"]
        return self._validate_response_structure(response, expected)

    def _validate_chat_response(self, response: Dict[str, Any]) -> bool:
        """챗봇 응답 검증"""
        expected = ["response", "key_points", "recommendations"]
        return self._validate_response_structure(response, expected)
    
    def _merge_sections(self, existing_sections: Dict[str, Any], new_sections: Dict[str, Any]) -> Dict[str, Any]:
        """섹션 병합"""
        if not existing_sections:
            return new_sections
            
        merged = existing_sections.copy()
        
        # 각 최상위 섹션에 대해 병합
        for key in new_sections:
            if key not in merged:
                merged[key] = new_sections[key]
            else:
                # 이미 존재하는 섹션이면 타입에 따라 병합
                if isinstance(new_sections[key], dict):
                    if isinstance(merged[key], dict):
                        merged[key].update(new_sections[key])
                elif isinstance(new_sections[key], list):
                    if isinstance(merged[key], list):
                        merged[key].extend(x for x in new_sections[key] if x not in merged[key])
                else:
                    # 기본값은 새로운 값으로 덮어쓰기
                    merged[key] = new_sections[key]
                    
        return merged

