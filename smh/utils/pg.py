#pg.py

from typing import Dict, Any, List, Optional
from fastapi import HTTPException
import logging
import json
import traceback
from functools import wraps
import re

from .agent import AgentController
from .ps import ProposalServer, WorkflowStatus, ProposalServerError

def error_handler(func):
    """에러 처리 데코레이터"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"{func.__name__} 실행 중 오류 발생: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            if isinstance(e, ProposalServerError):
                raise HTTPException(status_code=e.error_code, detail=e.message)
            raise HTTPException(status_code=500, detail=f"{func.__name__} 실행 실패: {str(e)}")
    return wrapper

class ProposalGenerator:
    """제안서 생성 클래스"""
    
    def __init__(self):
        """초기화"""
        self._setup_logger()
        self.agent = AgentController()
        self.server = ProposalServer(self.agent)
        self.current_proposal_id = None
    
    def _setup_logger(self) -> None:
        """로거 설정"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        file_handler = logging.FileHandler('logs/proposal_generator.log', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    @error_handler
    async def rfp_extract_sections(self, full_text: str) -> Dict[str, Any]:
        try:
            self.current_proposal_id = f"proposal_{str(hash(full_text))[:8]}"
            
            # 1. Server 초기화
            await self.initialize_proposal(self.current_proposal_id, full_text)
            
            # 2. Agent로 섹션 추출
            extracted_sections = await self.agent.extract_sections(full_text)
            
            # 3. Server로 섹션 처리
            processed_sections = await self.server.process_sections({
                "sections": extracted_sections,
                "type": "rfp"
            })
            
            return processed_sections
        except Exception as e:
            self.logger.error(f"RFP 섹션 추출 실패: {str(e)}")
            raise
        
    @error_handler
    async def generate_toc(self, sections: Dict[str, Any], total_pages: int, presentation_time: int):
        try:
            # 1. 섹션 데이터 정규화 및 메타데이터 추출
            normalized_sections = self._normalize_sections(sections)
            metadata = {
                'page_limit': total_pages,
                'presentation_time': presentation_time,
                'project_metadata': self._extract_project_metadata(normalized_sections)
            }
            
            # 2. 서버 상태 및 ID 확인
            if not self.current_proposal_id:
                self.current_proposal_id = f"proposal_{str(hash(str(sections)))[:8]}"
                await self.initialize_proposal(self.current_proposal_id, str(sections))
            
            # 3. 서버에서 TOC 생성
            toc_result = await self.server.generate_toc(
                total_pages=total_pages,
                presentation_time=presentation_time
            )
            
            return toc_result
            
        except Exception as e:
            self.logger.error(f"목차 생성 실패: {str(e)}")
            raise

    @error_handler
    async def generate_page_content(self, 
                                  section_data: Dict[str, Any],
                                  rfp_data: Dict[str, Any]) -> Dict[str, Any]:
        """페이지 내용 생성"""
        try:
            if self.server.workflow_status != WorkflowStatus.TOC_GENERATED:
                raise ProposalServerError("목차가 생성되지 않았습니다")
                
            # 헤드카피 생성
            head_copy = await self.agent.generate_head_copy(section_data, rfp_data)
            
            # 서브카피 생성
            sub_copy = await self.agent.generate_sub_copy(section_data, rfp_data)
            
            # 컨텐츠 생성
            content = await self.agent.generate_content_section(section_data, rfp_data)
            
            # 결과 병합
            return {
                "title": section_data["title"],
                "head_copy": head_copy.get("head_copy", ""),
                "sub_copy": sub_copy.get("sub_copy", ""),
                "content": content.get("section_content", {}),
                "key_points": (
                    head_copy.get("key_message", []) + 
                    sub_copy.get("benefits", [])
                )
            }
            
        except Exception as e:
            self.logger.error(f"페이지 내용 생성 실패: {str(e)}")
            raise

    @error_handler
    async def get_generation_status(self) -> Dict[str, Any]:
        """생성 상태 조회"""
        try:
            if not self.current_proposal_id:
                return {
                    "status": "not_started",
                    "progress": 0
                }
                
            progress = await self.server.get_generation_progress()
            current_state = await self.server.get_current_state()
            
            return {
                "status": progress["status"],
                "progress": progress["progress"],
                "current_section": current_state.get("current_section"),
                "total_sections": progress["total_sections"],
                "completed_sections": progress["completed_sections"]
            }
            
        except Exception as e:
            self.logger.error(f"생성 상태 조회 실패: {str(e)}")
            raise
        
    @error_handler
    async def process_document(self, content: bytes, file_ext: str) -> Dict[str, Any]:
        """문서 처리 및 초기 분석"""
        try:
            # 새 제안서 초기화
            proposal_id = f"proposal_{str(hash(content))[:8]}"
            await self.server.initialize_proposal(
                proposal_id,
                {
                    'content': content,
                    'file_type': file_ext
                }
            )
            
            # 서버에 문서 처리 요청
            doc_result = await self.server.process_document({
                "content": content,
                "file_type": file_ext
            })
            
            if not doc_result:
                raise ProposalServerError("문서 처리에 실패했습니다")
            
            return {
                "text": doc_result.get("text", ""),
                "table_sections": doc_result.get("table_sections", []),
                "metadata": doc_result.get("metadata", {})
            }
            
        except Exception as e:
            self.logger.error(f"문서 처리 실패: {str(e)}")
            raise

    @error_handler
    async def extract_sections(self, text: str) -> Dict[str, Any]:
        try:
            self.current_proposal_id = f"proposal_{str(hash(text))[:8]}"
            
            # 서버 초기화
            await self.server.initialize_proposal(
                self.current_proposal_id,
                {"raw_text": text}
            )
            
            # 섹션 데이터 구조 수정
            sections_data = {
                "sections": await self.agent.extract_sections(text),
                "type": "rfp"
            }
            
            return await self.server.process_sections(sections_data)
        except Exception as e:
            self.logger.error(f"섹션 추출 실패: {str(e)}")
            raise

    @error_handler
    async def extract_default_sections(self, text: str) -> Dict[str, Any]:
        """기본 섹션 추출"""
        try:
            import re
            
            # 기본 섹션 패턴 정의
            section_patterns = {
                '사업개요': r'사업\s*개요|개요|사업\s*목적',
                '사업범위': r'사업\s*범위|과업\s*범위|제안\s*범위',
                '제안요청내용': r'제안\s*요청|과업\s*내용|수행\s*내용',
                '평가항목': r'평가\s*항목|평가\s*기준|심사\s*기준'
            }
            
            sections = {}
            current_position = 0
            
            # 각 섹션 패턴에 대해 매칭 시도
            for section_name, pattern in section_patterns.items():
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    start = matches[0].start()
                    
                    # 이전 섹션의 끝부터 현재 섹션의 시작까지가 이전 섹션의 내용
                    if current_position < start:
                        previous_section = list(sections.keys())[-1] if sections else None
                        if previous_section:
                            sections[previous_section] = text[current_position:start].strip()
                    
                    # 다음 섹션의 시작 위치 찾기
                    next_section_start = text.find('\n', start)
                    if next_section_start == -1:
                        next_section_start = len(text)
                    
                    sections[section_name] = text[start:next_section_start].strip()
                    current_position = next_section_start
                else:
                    sections[section_name] = ""
            
            return sections
                
        except Exception as e:
            self.logger.error(f"기본 섹션 추출 실패: {str(e)}")
            raise

    @error_handler
    async def analyze_sections(self, sections: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        """섹션 분석 수행"""
        try:
            if not self.current_proposal_id:
                raise ProposalServerError("pg.py, analyze_sectinos, 활성화된 제안서가 없습니다")
                
            analysis_result = await self.server.analyze_sections({
                "sections": sections,
                "raw_text": raw_text
            })
            
            return {
                "sections": analysis_result.get("processed_sections", {}),
                "scoring_table": analysis_result.get("scoring_table", {})
            }
            
        except Exception as e:
            self.logger.error(f"pg.py, analyze_sections; 섹션 분석 실패: {str(e)}")
            raise
    
    def _normalize_sections(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """섹션 데이터 정규화"""
        normalized = {}
        for key, value in sections.items():
            # 섹션 키 정규화
            normalized_key = key.strip().replace("*", "").replace("#", "")
            
            # 섹션 값 정규화
            if isinstance(value, str):
                normalized[normalized_key] = value.strip()
            elif isinstance(value, dict):
                normalized[normalized_key] = self._normalize_sections(value)
            elif isinstance(value, list):
                normalized[normalized_key] = [
                    item.strip() if isinstance(item, str) else item 
                    for item in value
                ]
            else:
                normalized[normalized_key] = value
                
        return normalized

    def _normalize_section_content(self, content: Any) -> Dict[str, Any]:
        """섹션 컨텐츠 정규화"""
        if isinstance(content, list):
            return {
                "content": content,
                "metadata": {
                    "word_count": sum(len(str(item).split()) for item in content),
                    "has_tables": False,
                    "has_figures": False,
                    "references": []
                },
                "type": "list"
            }
        elif isinstance(content, str):
            return {
                "content": content,
                "metadata": {
                    "word_count": len(content.split()),
                    "has_tables": '표' in content,
                    "has_figures": '그림' in content,
                    "references": re.findall(r'\[(.*?)\]', content)
                },
                "type": "text"
            }
        elif isinstance(content, dict):
            return content  # 이미 정규화된 형태라고 가정
        else:
            return {
                "content": str(content),
                "metadata": {
                    "word_count": 0,
                    "has_tables": False,
                    "has_figures": False,
                    "references": []
                },
                "type": "unknown"
            }
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        import re
        
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 섹션 구분자 정규화
        text = re.sub(r'[IVX]+\.|[0-9]+\.', lambda x: f"\n{x.group()}", text)
        
        return text.strip()

    def _postprocess_sections(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """섹션 후처리 및 검증"""
        required_sections = {'사업개요', '사업범위', '제안요청내용', '평가항목'}
        processed = {}
        
        for key, value in sections.items():
            # 필수 섹션 확인
            if any(req in key for req in required_sections):
                processed[key] = value
                
        # 누락된 필수 섹션 처리
        for req in required_sections:
            if not any(req in key for key in processed.keys()):
                processed[req] = ""
                
        return processed
    
    def _calculate_page_allocation(self, evaluation_data: Optional[Dict[str, Any]], total_pages: int) -> Dict[str, int]:
        """페이지 할당 계산"""
        try:
            # 예약된 페이지 (표지, 목차 등)
            reserved_pages = 4  # 표지(1) + 목차(1) + 요약(2)
            available_pages = max(total_pages - reserved_pages, 1)

            # 기본 가중치 정의
            weights = {
                '사업개요': 0.2,
                '세부과업': 0.4,
                '관리방안': 0.2,
                '지원방안': 0.2
            }

            # 평가 데이터가 있는 경우 가중치 조정
            if evaluation_data and evaluation_data.get('정성평가', {}).get('배점'):
                total_score = sum(evaluation_data['정성평가']['배점'].values())
                if total_score > 0:
                    weights = {
                        section: score / total_score 
                        for section, score in evaluation_data['정성평가']['배점'].items()
                    }

            # 페이지 할당
            allocations = {}
            for section, weight in weights.items():
                allocations[section] = max(1, int(available_pages * weight))

            return allocations

        except Exception as e:
            self.logger.error(f"페이지 할당 계산 실패: {str(e)}")
            # 기본 할당 반환
            return {
                '사업개요': max(1, int(total_pages * 0.2)),
                '세부과업': max(1, int(total_pages * 0.4)),
                '관리방안': max(1, int(total_pages * 0.2)),
                '지원방안': max(1, int(total_pages * 0.2))
            }

    async def cleanup(self) -> None:
        """리소스 정리"""
        try:
            if self.current_proposal_id:
                await self.server.cleanup()
                self.current_proposal_id = None
                
        except Exception as e:
            self.logger.error(f"리소스 정리 실패: {str(e)}")