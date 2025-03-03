# ps.py

from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
import json
import traceback
import re
import asyncio  # asyncio 추가
from enum import Enum


# AgentController import 수정
if TYPE_CHECKING:
    from .agent import AgentController
else:
    # 이 부분을 수정
    from .agent import AgentController  # 직접 임포트로 변경

# 섹션 관련 상수
SECTION_TYPES = {
    "OVERVIEW": "overview",
    "MAIN_CONTENTS_OVERVIEW": "main_contents_overview",
    "MAIN_CONTENTS_DETAIL": "main_contents_detail",
    "MAINTAINANCE_PLAN": "maintainance_plan"
}


# 목차 템플릿
# 목차 템플릿 수정
TOC_TEMPLATES = {
    '사업개요': {
        'title': "Ⅰ. 제안 개요",
        'order': 1,
        'min_pages': 3,
        'max_pages': 5
    },
    '세부과업': {
        'sections': [
            {
                'title': "Ⅱ. 사업 추진계획",
                'order': 2,
                'min_pages': 5,
                'max_pages': 10
            },
            {
                'title': "Ⅲ. 사업 추진체계",
                'order': 3,
                'min_pages': 3,
                'max_pages': 7
            }
        ]
    },
    '관리방안': {
        'title': "Ⅳ. 사업 관리방안",
        'order': 4,
        'min_pages': 3,
        'max_pages': 6
    },
    '지원방안': {
        'title': "Ⅴ. 기술 지원방안",
        'order': 5,
        'min_pages': 3,
        'max_pages': 6
    }
}

# 섹션 매핑 상수 추가
SECTION_MAPPINGS = {
'사업개요': ['제안개요', '사업개요', '과업개요', '제안배경', '사업목적', '개요', '일반사항'],
'세부과업': ['추진계획', '수행계획', '추진전략', '세부내용', '추진체계', '운영계획', '실행방안', '과업내용'],
'제안서작성방법': ['제안서 작성요령', '제안서 작성안내', '작성지침', '제안서 작성지침', '작성방법'],
'평가항목': ['평가항목', '평가기준', '평가방법', '심사기준', '선정기준']
}

@dataclass
class ProposalConfig:
    """제안서 기본 설정"""
    max_pages: int = 100
    min_pages: int = 20
    max_section_depth: int = 3
    default_presentation_time: int = 20  # minutes
    
    @classmethod
    def default(cls) -> 'ProposalConfig':
        return cls()

@dataclass
class ProposalState:
    """제안서 상태 관리"""
    id: str
    status: str = "initialized"
    created_at: datetime = field(default_factory=datetime.now)
    sections: Dict[str, Any] = field(default_factory=dict)
    page_allocations: Dict[str, int] = field(default_factory=dict)
    evaluation_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs) -> None:
        """상태 업데이트 메서드"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

class WorkflowStatus(Enum):
    """워크플로우 상태"""
    INITIALIZED = "initialized"
    FILE_PROCESSED = "file_processed"
    SECTIONS_ANALYZED = "sections_analyzed"
    TOC_GENERATED = "toc_generated"
    CONTENT_GENERATING = "content_generating"
    COMPLETED = "completed"
    ERROR = "error"
    METADATA_EXTRACTED = "metadata_extracted"  # 메타데이터 추출 완료
    EVALUATION_ANALYZED = "evaluation_analyzed"  # 평가정보 분석 완료
    
class ProposalServerError(Exception):
    """기본 예외 클래스"""
    def __init__(self, message: str, error_code: int = 500):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
        
class WorkflowError(ProposalServerError):
    """워크플로우 관련 예외"""
    pass

class PageAllocationError(ProposalServerError):
    """페이지 할당 관련 예외"""
    pass

class ValidationError(ProposalServerError):
    """데이터 검증 예외"""
    pass

class ProposalServer:
    """제안서 생성 서버"""
    # 1. 초기화 및 설정 메서드
    ## 1-1. 서버 초기화
    def __init__(self, agent_controller: Optional[AgentController] = None): 
        """초기화"""
        self._ensure_logs_directory()
        self.logger = self._setup_logger()
        self.config = ProposalConfig.default()
        self.agent = agent_controller or AgentController()
        
        # 상태 관리
        self.current_proposal: Optional[ProposalState] = None
        self.workflow_status = WorkflowStatus.INITIALIZED
        
        # 캐시 및 임시 저장소
        self._section_cache: Dict[str, Any] = {}
        self._page_allocation_cache: Dict[str, int] = {}
        
        # 전역 상수들을 인스턴스 속성으로 추가
        self.SECTION_MAPPINGS = SECTION_MAPPINGS
        self.TOC_TEMPLATES = TOC_TEMPLATES  # 추가된 부분
            
        self.logger.info("ProposalServer 초기화 완료")
        
    ## 1-2. 로그 디렉토리 생성   
    def _ensure_logs_directory(self) -> None:
        """로그 디렉토리 생성"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        if not os.path.exists('logs/proposals'):
            os.makedirs('logs/proposals')

    ## 1-3. 로거 설정
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        # 파일 핸들러 설정
        file_handler = logging.FileHandler(
            'logs/proposals/server.log', 
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # 기존 핸들러 제거 후 새로 추가
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    # 2. 워크플로우 관리 메서드
    ## 2-1. 제안서 초기화
    async def initialize_proposal(self, proposal_id: str, file_data: Dict[str, Any]) -> None:
        """제안서 초기화 및 상태 설정

        Args:
            proposal_id (str): 제안서 고유 ID
            file_data (Dict[str, Any]): {
                'content': str | bytes - 파일/텍스트 내용,
                'file_type': str - 파일 타입 (예: 'hwp', 'pdf', 'text', 'json')
            }

        Raises:
            ProposalServerError: 초기화 실패시
        """
        try:
            # ProposalState 객체 생성
            self.current_proposal = ProposalState(
                id=proposal_id,
                status=WorkflowStatus.INITIALIZED.value,
                sections={},
                evaluation_data={},
                metadata={
                    'content': file_data.get('content'),
                    'file_type': file_data.get('file_type'),
                    'created_at': datetime.now().isoformat()
                }
            )
            
            # 워크플로우 상태 업데이트
            self.workflow_status = WorkflowStatus.INITIALIZED
            
            # 캐시 초기화
            self._section_cache = {}
            self._page_allocation_cache = {}
            
            self.logger.info(f"ps.py, initialize_proposal함수, 제안서 초기화 완료: {proposal_id}")
            
        except Exception as e:
            self.logger.error(f"ps.py, initialize_proposal함수, 제안서 초기화 실패: {str(e)}")
            raise ProposalServerError(f"ps.py, initialize_proposal함수, 제안서 초기화 실패: {str(e)}")
    
    ## 2-2. 워크플로우 상태 검증  
    def _validate_workflow_state(self, expected_status: List[WorkflowStatus]) -> bool:
        """워크플로우 상태 검증 개선"""
        if not self.current_proposal:
            raise WorkflowError("ps.py, validate_workflow_stae, 활성화된 제안서가 없습니다")
            
        current = self.workflow_status
        if current not in expected_status:
            status_names = [status.value for status in expected_status]
            self.logger.error(
                f"잘못된 워크플로우 상태: {current.value} "
                f"(예상 상태: {', '.join(status_names)})"
            )
            raise WorkflowError(
                f"현재 상태({current.value})에서는 이 작업을 수행할 수 없습니다. "
                f"필요한 상태: {', '.join(status_names)}"
            )
        return True
    
    # 3. 섹션 처리 메서드
    ## 3-1. 섹션 데이터 처리  
    async def process_sections(self, sections_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._validate_workflow_state([WorkflowStatus.INITIALIZED, WorkflowStatus.FILE_PROCESSED])
            
            # sections_data 구조 검증 추가
            if not isinstance(sections_data, dict):
                raise ValidationError("유효하지 않은 sections_data 형식")
                
            normalized_sections = await self._normalize_sections(sections_data.get("sections", {}))
            evaluation_data = await self._extract_evaluation_info(normalized_sections)
            metadata = self._extract_detailed_metadata(normalized_sections)
            
            processed_sections = {
                'general': await self._process_general_sections(normalized_sections),
                'evaluation': evaluation_data,
                'metadata': metadata
            }
            
            self.current_proposal.sections = processed_sections
            self.workflow_status = WorkflowStatus.SECTIONS_ANALYZED
            
            return processed_sections
        except Exception as e:
            self.logger.error(f"섹션 처리 실패: {str(e)}")
            raise ProposalServerError(f"섹션 처리 실패: {str(e)}")
        
    async def _extract_evaluation_info(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """평가 정보 추출 강화"""
        evaluation_data = {
            'scoring': {'정성평가': {}, '정량평가': {}},
            'criteria': [],
            'weights': {},
            'total_score': 0
        }

        for key, content in sections.items():
            if any(term in key for term in ['평가', '배점', '심사']):
                if isinstance(content, str):
                    # 정규식을 사용한 점수 추출
                    scores = re.findall(r'(\d+)점', content)
                    evaluation_data['total_score'] += sum(map(int, scores))
                    
                    # 평가 기준 추출
                    criteria = re.findall(r'[가-힣\s]+(평가|심사)[가-힣\s]*[:：]([^.]*)', content)
                    evaluation_data['criteria'].extend([c[1].strip() for c in criteria])

        return evaluation_data
    
    def _extract_detailed_metadata(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """상세 메타데이터 추출"""
        metadata = {
            'project': {
                'name': None,
                'duration': None,
                'budget': None
            },
            'requirements': {
                'page_limit': None,
                'presentation_time': None
            },
            'deadlines': {}
        }

        # 메타데이터 추출 로직
        for key, content in sections.items():
            content_str = str(content).lower()
            
            # 프로젝트 정보 추출
            if '사업명' in content_str:
                metadata['project']['name'] = re.search(r'사업명[:\s]*([^\n]+)', content_str)
                
            # 예산 정보 추출
            if '예산' in content_str or '사업비' in content_str:
                budget_match = re.search(r'(?:예산|사업비)[:\s]*([0-9,.]+)(?:원|백만원|억원)?', content_str)
                if budget_match:
                    metadata['project']['budget'] = budget_match.group(1)

        return metadata
    
    ## 3-2. 일반 섹션 처리
    async def _process_general_sections(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """일반 섹션 처리"""
        processed = {}
        for section_name, content in sections.items():
            # 섹션 이름 표준화
            std_name = self._standardize_section_name(section_name)
            processed[std_name] = {
                'original_name': section_name,
                'content': content,
                'type': self._determine_section_type(std_name),
                'metadata': self._extract_section_metadata(content)
            }
        return processed

    ## 3-3. 평가 섹션 처리
    async def _process_evaluation_sections(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """평가 관련 섹션 처리"""
        scoring_data = sections.get('배점표', {})
        evaluation = {
            'total_score': self._calculate_total_score(scoring_data),
            'categories': self._process_scoring_categories(scoring_data),
            'criteria': await self._extract_evaluation_criteria(sections)
        }
        return evaluation
    
    ## 3-4. 섹션 이름 표준화화
    def _standardize_section_name(self, name: str) -> str:
        """섹션 이름 표준화"""
        for std_name, variants in self.SECTION_MAPPINGS.items():
            if any(variant.lower() in name.lower() for variant in variants):
                return std_name
        return name
    
    async def _normalize_sections(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """섹션 데이터 정규화"""
        try:
            normalized = {
                '사업개요': {},
                '세부과업': {},
                '제안서작성방법': {},
                '평가항목': {}
            }
            
            # 텍스트 내용을 섹션별로 분류
            for key, content in sections.items():
                matched_section = None
                
                # SECTION_MAPPINGS를 사용하여 섹션 매칭
                for section_type, variants in self.SECTION_MAPPINGS.items():
                    if any(variant.lower() in key.lower() for variant in variants):
                        matched_section = section_type
                        break
                
                if matched_section:
                    if isinstance(content, str):
                        if matched_section not in normalized:
                            normalized[matched_section] = {}
                        normalized[matched_section][key] = content.strip()
                    elif isinstance(content, dict):
                        normalized[matched_section].update(content)
                        
            self.logger.debug(f"ps.py, normalize_sections 함수,정규화된 섹션: {json.dumps(normalized, ensure_ascii=False)}")
            return normalized
            
        except Exception as e:
            self.logger.error(f"ps.py, normalize_sections 함수,섹션 정규화 실패: {str(e)}")
            raise ProposalServerError(f"ps.py, normalize_sections 함수,섹션 정규화 실패: {str(e)}")
    
    async def analyze_sections(self, sections: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        try:
            self.logger.info("ps.py, analyze_sections,섹션 분석 시작")
            self.logger.debug(f"Received sections type: {type(sections)}")
            self.logger.debug(f"Received sections keys: {sections.keys() if isinstance(sections, dict) else 'Not a dict'}")

            # sections이 비어있거나 딕셔너리가 아닌 경우 처리
            if not isinstance(sections, dict) or not sections:
                sections = {
                    "사업개요": {},
                    "평가정보": {
                        "기술평가": {"총점": 0, "항목": [], "배점": {}},
                        "가격평가": {"총점": 0}
                    },
                    "제안범위": [],
                    "수행일정": {"착수": "", "중간": "", "최종": "", "세부일정": []}
                }

            # 섹션 처리
            processed_sections = {
                "사업개요": sections.get("사업개요", {}),
                "세부과업": {},
                "제안서작성방법": {},
                "평가항목": sections.get("평가정보", {})
            }

            # 메타데이터 추출
            metadata = self._extract_detailed_metadata(sections.get("사업개요", {}))

            analysis_result = {
                "sections": processed_sections,
                "metadata": metadata
            }

            # 현재 제안서 상태 업데이트
            if self.current_proposal:
                self.current_proposal.update(
                    sections=analysis_result,
                    status=WorkflowStatus.SECTIONS_ANALYZED.value
                )

            self.workflow_status = WorkflowStatus.SECTIONS_ANALYZED
            self.logger.info("ps.py, analyze_sections, 섹션 분석 완료")

            return analysis_result

        except Exception as e:
            self.logger.error(f"섹션 분석 실패: {str(e)}")
            return {
                "sections": {},
                "metadata": {}
            }
    
    # 4. 페이지 할당 및 계산 메서드
    ## 4-1. 페이지 분배 계산산
    def _calculate_page_distribution(
    self,
    total_pages: int,
    evaluation_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
        """페이지 분배 계산"""
        try:
            # 기본 페이지 할당
            reserved_pages = 4  # 표지(1) + 목차(1) + 요약(2)
            available_pages = max(total_pages - reserved_pages, 1)
            
            # 평가 데이터 기반 가중치 계산
            weights = self._calculate_section_weights(evaluation_data)
            
            # 페이지 할당
            allocations = {}
            remaining_pages = available_pages
            
            for section, weight in weights.items():
                # 페이지 수 계산 (최소 1페이지 보장)
                pages = max(1, int(available_pages * weight))
                remaining_pages -= pages
                
                allocations[section] = {
                    'pages': pages,
                    'weight': weight,
                    'original_score': (evaluation_data or {}).get('정성평가', {})
                                    .get('부문별점수', {}).get(section, 0)
                }
            
            # 남은 페이지 분배
            if remaining_pages > 0:
                self._distribute_remaining_pages(allocations, remaining_pages)
            
            return allocations

        except Exception as e:
            self.logger.error(f"페이지 분배 계산 실패: {str(e)}")
            raise PageAllocationError(f"페이지 분배 계산 실패: {str(e)}")
    
    ## 4-2. 남은 페이지 분배    
    def _distribute_remaining_pages(
        self,
        allocations: Dict[str, Dict[str, Any]],
        remaining_pages: int
    ) -> None:
        """남은 페이지 분배"""
        # 가중치 기반으로 정렬
        sorted_sections = sorted(
            allocations.items(),
            key=lambda x: (x[1]['weight'], x[1]['original_score']),
            reverse=True
        )
        
        # 남은 페이지 분배
        for i in range(remaining_pages):
            section_name = sorted_sections[i % len(sorted_sections)][0]
            allocations[section_name]['pages'] += 1
    
    ## 4-3. 섹션별 가중치 계산        
    def _calculate_section_weights(
    self,
    evaluation_data: Optional[Dict[str, Any]]
) -> Dict[str, float]:
        """섹션별 가중치 계산"""
        try:
            # evaluation_data가 None이면 기본값 사용
            if not evaluation_data:
                self.logger.warning("평가 데이터가 없습니다. 기본 가중치를 사용합니다.")
                return {
                    '사업개요': 0.2,
                    '세부과업': 0.4,
                    '관리방안': 0.2,
                    '지원방안': 0.2
                }

            total_score = evaluation_data.get('정성평가', {}).get('총점', 100)
            section_scores = evaluation_data.get('정성평가', {}).get('부문별점수', {})
            
            # 섹션 점수가 없으면 기본 가중치 사용
            if not section_scores:
                self.logger.warning("섹션 점수가 없습니다. 기본 가중치를 사용합니다.")
                return {
                    '사업개요': 0.2,
                    '세부과업': 0.4,
                    '관리방안': 0.2,
                    '지원방안': 0.2
                }
                
            weights = {}
            for section, score in section_scores.items():
                weights[section] = score / total_score if total_score > 0 else 0.25
                
            # 가중치 정규화
            weight_sum = sum(weights.values())
            if weight_sum > 0:
                weights = {k: v/weight_sum for k, v in weights.items()}
            else:
                self.logger.warning("가중치 합이 0입니다. 균등 분배를 적용합니다.")
                section_count = len(section_scores)
                weights = {k: 1.0/section_count for k in section_scores.keys()}
                
            return weights

        except Exception as e:
            self.logger.warning(f"가중치 계산 중 오류 발생: {str(e)}, 기본 가중치를 사용합니다.")
            return {
                '사업개요': 0.2,
                '세부과업': 0.4,
                '관리방안': 0.2,
                '지원방안': 0.2
            }
    
    ## 4-4. 페이지 제약 검증
    def _validate_page_constraints(self, total_pages: int) -> int:
        """페이지 제약조건 검증"""
        if total_pages < self.config.min_pages:
            self.logger.warning(
                f"요청된 페이지 수({total_pages})가 최소 제한({self.config.min_pages})보다 작습니다."
            )
            return self.config.min_pages
            
        if total_pages > self.config.max_pages:
            self.logger.warning(
                f"요청된 페이지 수({total_pages})가 최대 제한({self.config.max_pages})를 초과합니다."
            )
            return self.config.max_pages
            
        return total_pages
    
    ## 4-5. 발표시간 기반 페이지 수 계산
    def _calculate_presentation_pages(self, presentation_time: int) -> int:
        """발표시간 기반 페이지 수 계산 (30초/페이지)"""
        return min(presentation_time * 2, self.config.max_pages)

    # 5. 목차 생성 관련 메서드
    ## 5-1. 목차 생성
    async def generate_toc(
        self,
        total_pages: int,
        presentation_time: int
    ) -> Dict[str, Any]:
        """목차 생성"""
        try:
            self._validate_workflow_state([WorkflowStatus.SECTIONS_ANALYZED])
            
            # 기본 제약조건 검증
            validated_pages = self._validate_page_constraints(total_pages)
            presentation_pages = self._calculate_presentation_pages(presentation_time)
            final_pages = min(validated_pages, presentation_pages)
            
            # 페이지 할당 계산
            page_allocations = self._calculate_page_distribution(
                final_pages,
                self.current_proposal.evaluation_data
            )
            
            # 목차 구조 생성
            toc_structure = await self._create_toc_structure(
                page_allocations,
                self.current_proposal.sections
            )
            
            # 현재 상태 업데이트
            self.current_proposal.page_allocations = page_allocations
            self._toc_data = toc_structure
            self.workflow_status = WorkflowStatus.TOC_GENERATED
            
            return {
                'structure': toc_structure,
                'allocations': page_allocations,
                'total_pages': final_pages
            }

        except Exception as e:
            self.logger.error(f"목차 생성 실패: {str(e)}")
            raise ProposalServerError(f"목차 생성 실패: {str(e)}")
    
    ## 5-2. 목차 구조 생성    
    async def _create_toc_structure(
    self,
    page_allocations: Dict[str, Dict[str, Any]],
    sections: Dict[str, Any]
) -> List[Dict[str, Any]]:
        """목차 구조 생성 개선"""
        try:
            toc_sections = []
            current_page = 1

            # 1. 섹션 데이터 검증
            if not sections:
                raise ValueError("섹션 데이터가 없습니다.")

            # 2. 템플릿 기반 목차 생성
            for template_key, template_value in self.TOC_TEMPLATES.items():
                # 단일 섹션 처리
                if 'title' in template_value:
                    section = await self._create_section_structure(
                        template_value['title'],
                        page_allocations,
                        sections,
                        current_page
                    )
                    if section:
                        current_page += section['pages']
                        toc_sections.append(section)
                
                # 복수 섹션 처리 (세부과업 등)
                elif 'sections' in template_value:
                    for sub_template in template_value['sections']:
                        section = await self._create_section_structure(
                            sub_template['title'],
                            page_allocations,
                            sections,
                            current_page
                        )
                        if section:
                            current_page += section['pages']
                            toc_sections.append(section)

            # 3. 섹션 정렬
            toc_sections.sort(key=lambda x: self._get_section_order(x['title']))
            
            # 4. 페이지 번호 재조정
            current_page = 1
            for section in toc_sections:
                section['start_page'] = current_page
                current_page += section['pages']

            return toc_sections

        except Exception as e:
            self.logger.error(f"목차 구조 생성 실패: {str(e)}")
            raise
    
#     ## 5-3. 섹션 구조 생성
    async def _create_section_structure(
    self,
    title: str,
    allocations: Dict[str, Dict[str, Any]],
    sections: Dict[str, Any],
    start_page: int
) -> Optional[Dict[str, Any]]:
        """섹션 구조 생성"""
        try:
            # 1. 섹션 기본 정보 설정
            mapped_title = self._map_section_title(title)
            section_data = allocations.get(mapped_title, {})
            pages = section_data.get('pages', self._get_default_pages(title))

            # 2. 기본 섹션 구조 생성
            section = {
                'title': title,
                'pages': pages,
                'start_page': start_page,
                'key_points': [],
                'sub_sections': [],
                'order': self._get_section_order(title)
            }

            # 3. AgentController를 통한 컨텐츠 구조화
            try:
                content_data = {
                    'title': title,
                    'pages': pages,
                    'section_type': self._determine_section_type(title)
                }
                
                content_structure = await self.agent.generate_content_section(
                    content_data,
                    sections.get('general', {})
                )

                if content_structure:
                    section.update({
                        'key_points': content_structure.get('key_points', []),
                        'sub_sections': self._process_subsections(
                            content_structure.get('details', []),
                            start_page,
                            pages
                        ),
                        'content_overview': content_structure.get('overview', '')
                    })

            except Exception as e:
                self.logger.warning(f"섹션 '{title}' 컨텐츠 구조화 실패: {str(e)}")

            return section

        except Exception as e:
            self.logger.error(f"섹션 구조 생성 실패 ({title}): {str(e)}")
            return None
    
    def _map_section_title(self, title: str) -> str:
        """섹션 제목 매핑"""
        for std_title, variants in self.SECTION_MAPPINGS.items():
            if any(variant in title for variant in variants):
                return std_title
        return title

    def _get_default_pages(self, title: str, total_pages: int = 20) -> int:
        """섹션별 기본 페이지 수 반환"""
        # 섹션별 비율 정의
        section_ratios = {
            'overview': 0.15,              # 개요 섹션 (15%)
            'main_contents_overview': 0.25, # 본론섹션_1_(본론개요) (25%)
            'main_contents_detail': 0.5,    # 본론섹션_2_(본론상세) (50%)
            'maintainance_plan': 0.1       # 관리섹션 (10%)
        }
        
        # 최소 페이지 수 정의
        min_pages = {
            'overview': 2,
            'main_contents_overview': 3,
            'main_contents_detail': 8,
            'maintainance_plan': 2
        }

        # TOC_TEMPLATES에서 기본값 찾기
        for template_key, template in self.TOC_TEMPLATES.items():
            if 'title' in template and template['title'] == title:
                ratio = section_ratios.get(self._determine_section_type(title), 0.15)
                calculated_pages = max(
                    min_pages.get(self._determine_section_type(title), 2),
                    int(total_pages * ratio)
                )
                return calculated_pages

        # 섹션 타입 확인 후 페이지 수 계산
        section_type = self._determine_section_type(title)
        ratio = section_ratios.get(section_type, 0.15)  # 기본 비율 15%
        calculated_pages = int(total_pages * ratio)
        
        # 최소 페이지 수 보장
        return max(min_pages.get(section_type, 2), calculated_pages)

    def _get_section_order(self, title: str) -> int:
        """섹션 순서 반환"""
        # TOC_TEMPLATES에서 순서 찾기
        for template_key, template in self.TOC_TEMPLATES.items():
            if 'title' in template and template['title'] == title:
                return template.get('order', len(self.TOC_TEMPLATES) + 1)
            elif 'sections' in template:
                for sub_template in template['sections']:
                    if sub_template['title'] == title:
                        return sub_template.get('order', len(self.TOC_TEMPLATES) + 1)
        
        # 기본 섹션 순서 정의
        default_orders = {
            'Ⅰ. 제안 개요': 1,
            'Ⅱ. 사업 추진계획': 2,
            'Ⅲ. 사업 추진체계': 3,
            'Ⅳ. 사업 관리방안': 4,
            'Ⅴ. 사후 관리방안': 5
        }
        
        return default_orders.get(title, len(default_orders) + 1)  # 마지막 순서 + 1
    
    ## 5-4. 하위 섹션 처리
    def _process_subsections(
        self,
        details: List[Dict[str, Any]],
        start_page: int,
        total_pages: int
    ) -> List[Dict[str, Any]]:
        """하위 섹션 처리"""
        sub_sections = []
        pages_per_subsection = max(1, total_pages // max(len(details), 1))
        
        for i, detail in enumerate(details):
            sub_section = {
                'title': detail.get('title', f'하위섹션 {i+1}'),
                'content_type': detail.get('type', 'text'),
                'pages': pages_per_subsection,
                'page_number': start_page + (i * pages_per_subsection)
            }
            sub_sections.append(sub_section)
            
        return sub_sections

    # 6. 컨텐츠 생성 메서드
    ## 6-1. 컨텐츠 생성    
    async def generate_content(self) -> AsyncGenerator[Dict[str, Any], None]:
        """제안서 컨텐츠 생성"""
        try:
            self._validate_workflow_state([WorkflowStatus.TOC_GENERATED])
            self.workflow_status = WorkflowStatus.CONTENT_GENERATING
            
            if not self._toc_data:
                raise ProposalServerError("목차 데이터가 없습니다")
                
            sections_data = self.current_proposal.sections
            
            for section in self._toc_data:
                try:
                    content = await self._generate_section_content(
                        section,
                        sections_data
                    )
                    
                    # 생성 이벤트 전달
                    yield {
                        "event_type": "section_complete",
                        "section_title": section['title'],
                        "content": content,
                        "status": "success"
                    }
                    
                except Exception as e:
                    self.logger.error(f"섹션 '{section['title']}' 생성 실패: {str(e)}")
                    yield {
                        "event_type": "section_error",
                        "section_title": section['title'],
                        "error": str(e),
                        "status": "error"
                    }
                    
            self.workflow_status = WorkflowStatus.COMPLETED
            
        except Exception as e:
            self.logger.error(f"컨텐츠 생성 실패: {str(e)}")
            self.workflow_status = WorkflowStatus.ERROR
            yield {
                "event_type": "generation_error",
                "error": str(e),
                "status": "error"
            }
            
    ## 6-2. 섹션 컨텐츠 생성
    async def _generate_section_content(
        self,
        section: Dict[str, Any],
        sections_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """섹션 컨텐츠 생성"""
        try:
            # 헤드카피 생성
            head_copy = await self.agent.generate_head_copy(
                section,
                sections_data
            )
            
            # 서브카피 생성
            sub_copy = await self.agent.generate_sub_copy(
                section,
                sections_data
            )
            
            # 본문 컨텐츠 생성
            content_structure = {
                "title": section['title'],
                "head_copy": head_copy.get('head_copy', ''),
                "sub_copy": sub_copy.get('sub_copy', ''),
                "key_points": head_copy.get('key_message', []) + 
                            sub_copy.get('benefits', []),
                "content": {
                    "overview": "",
                    "sections": []
                }
            }
            
            # 하위 섹션 컨텐츠 생성
            for sub_section in section.get('sub_sections', []):
                sub_content = await self._generate_subsection_content(
                    sub_section,
                    section,
                    sections_data
                )
                
                if sub_content:
                    content_structure['content']['sections'].append(sub_content)
            
            return content_structure
            
        except Exception as e:
            self.logger.error(f"섹션 컨텐츠 생성 실패: {str(e)}")
            raise
    
    ## 6-3. 하위 섹션 컨텐츠 생성        
    async def _generate_subsection_content(
        self,
        sub_section: Dict[str, Any],
        parent_section: Dict[str, Any],
        sections_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """하위 섹션 컨텐츠 생성"""
        try:
            content = await self.agent.generate_content_section(
                {
                    "title": sub_section['title'],
                    "pages": sub_section['pages'],
                    "content_type": sub_section.get('content_type', 'text')
                },
                sections_data
            )
            
            if not content:
                return None
                
            return {
                "title": sub_section['title'],
                "overview": content.get('section_content', {}).get('overview', ''),
                "items": content.get('section_content', {}).get('key_points', []),
                "details": content.get('section_content', {}).get('details', []),
                "outcomes": content.get('section_content', {}).get('expected_outcomes', [])
            }
            
        except Exception as e:
            self.logger.warning(f"하위 섹션 컨텐츠 생성 실패: {str(e)}")
            return None
    
    def method_requiring_agent_controller(self):
        from .agent import AgentController  # 필요한 시점에 임포트
        
    # 9. 누락된 메서드들
    def _determine_section_type(self, section_name: str) -> str:
        """섹션 타입 결정"""
        try:
            # 로마 숫자나 명시적 섹션 이름으로 판단
            if 'Ⅰ' in section_name or '개요' in section_name:
                return SECTION_TYPES["OVERVIEW"]
            elif 'Ⅱ' in section_name or '추진계획' in section_name:
                return SECTION_TYPES["MAIN_CONTENTS_OVERVIEW"]
            elif 'Ⅲ' in section_name or '추진체계' in section_name:
                return SECTION_TYPES["MAIN_CONTENTS_DETAIL"]
            elif any(x in section_name for x in ['Ⅳ', 'Ⅴ', '관리']):
                return SECTION_TYPES["MAINTAINANCE_PLAN"]

            # 기존 키워드 기반 판단 (폴백)
            if '기술' in section_name or '구현' in section_name:
                return SECTION_TYPES["MAIN_CONTENTS_DETAIL"]
            elif '운영' in section_name:
                return SECTION_TYPES["MAINTAINANCE_PLAN"]
            elif '일반' in section_name:
                return SECTION_TYPES["OVERVIEW"]
                
            return SECTION_TYPES["OVERVIEW"]  # 기본값
            
        except Exception as e:
            self.logger.warning(f"섹션 타입 결정 실패: {str(e)}")
            return SECTION_TYPES["OVERVIEW"]

    def _extract_section_metadata(self, content: Any) -> Dict[str, Any]:
        """섹션 메타데이터 추출"""
        try:
            metadata = {
                'word_count': len(str(content).split()) if isinstance(content, str) else 0,
                'has_tables': False,
                'has_figures': False,
                'references': []
            }
            
            if isinstance(content, str):
                # 표 존재 여부 체크
                metadata['has_tables'] = '표' in content or 'table' in content.lower()
                # 그림 존재 여부 체크
                metadata['has_figures'] = '그림' in content or 'figure' in content.lower()
                # 참조 추출
                references = re.findall(r'\[(.*?)\]', content)
                metadata['references'] = list(set(references))
            
            return metadata
        except Exception as e:
            self.logger.warning(f"메타데이터 추출 실패: {str(e)}")
            return {}

    async def extract_metadata(self, text: str) -> Dict[str, Any]:
        """메타데이터 추출"""
        try:
            self.logger.info("메타데이터 추출 시작")
            
            # 1차: Gemini를 통한 추출 시도
            try:
                prompt = self._create_metadata_extraction_prompt(text)
                response = await self.agent.generate_with_recovery(prompt)
                parsed_data = self.agent.parse_json_response(response)
                
                if 'metadata' in parsed_data:
                    metadata = parsed_data['metadata']
                    if self._validate_metadata(metadata):  # 검증 통과시
                        self.logger.info("Gemini를 통한 메타데이터 추출 성공")
                        return metadata
            except Exception as e:
                self.logger.warning(f"Gemini 메타데이터 추출 실패: {str(e)}")
            
            # 2차: 정규식 기반 추출 시도
            self.logger.info("정규식 기반 메타데이터 추출 시도")
            metadata = self._extract_metadata_fields(text)
            self._validate_metadata(metadata)
            
            return metadata
                
        except Exception as e:
            error_context = self._handle_extraction_error("메타데이터 추출", e)
            self.logger.error(f"메타데이터 추출 실패: {error_context}")
            raise ProposalServerError("메타데이터 추출 실패", error_context)
        
    def _handle_extraction_error(self, section_name: str, error: Exception) -> Dict[str, Any]:
        """추출 오류 처리 및 컨텍스트 수집"""
        error_context = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'section': section_name,
            'workflow_status': self.workflow_status.value,
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        }
        
        # 오류 세부정보 로깅
        self.logger.error(
            f"{section_name} 처리 오류:\n"
            f"Type: {error_context['error_type']}\n"
            f"Message: {error_context['error_message']}\n"
            f"Status: {error_context['workflow_status']}"
        )
        
        return error_context
        
    def _extract_metadata_fields(self, text: str) -> Dict[str, Any]:
        """정규식 기반 메타데이터 필드 추출"""
        patterns = {
            'organization': r'발주처[:\s]*([^\n]+)',
            'page_limit': r'(\d+)\s*(?:페이지|쪽)',
            'presentation_time': r'(\d+)\s*분(?:발표|PT)',
            'project_budget': r'(?:예산|사업비)[:\s]*(\d+(?:,\d+)*)\s*(?:원|백만원|억원)',
            'project_duration': r'(\d+)\s*(?:개월|년)',
            'project_name': r'사업명[:\s]*([^\n]+)',
            'submission_deadline': r'(?:제출|마감)[:\s]*(\d{4}[-./]\d{1,2}[-./]\d{1,2})'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            extracted[key] = match.group(1) if match else None
            
        return extracted
        
    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """메타데이터 유효성 검증"""
        required_fields = [
            'organization', 'page_limit', 'presentation_time',
            'project_budget', 'project_duration', 'project_name',
            'submission_deadline'
        ]
        
        for field in required_fields:
            if field not in metadata:
                self.logger.warning(f"누락된 필드: {field}")
                metadata[field] = None
                
            if field in ['page_limit', 'presentation_time']:
                try:
                    if metadata[field] is not None:
                        metadata[field] = int(metadata[field])
                except (ValueError, TypeError):
                    self.logger.warning(f"잘못된 숫자 형식: {field}")
                    metadata[field] = None

    def _create_metadata_extraction_prompt(self, text: str) -> str:
        """메타데이터 추출을 위한 프롬프트"""
        base_prompt = f"""당신은 RFP 문서 분석 전문가입니다. 다음 문서에서 메타데이터를 추출해주세요.

            입력 문서에서 다음 항목들을 찾아 정확히 추출해주세요:

            1. 발주기관명 (조직/기관명)
            2. 제안서 페이지 제한
            3. 발표 시간 (분 단위)
            4. 사업 예산
            5. 사업 기간
            6. 공식 사업명
            7. 제출 마감일시

            반드시 아래의 JSON 형식으로 응답해주세요:
            {{
                "metadata": {{
                    "organization": "발주기관명",
                    "page_limit": 숫자(없으면 null),
                    "presentation_time": 숫자(없으면 null),
                    "project_budget": "금액(없으면 null)",
                    "project_duration": "기간(없으면 null)",
                    "project_name": "사업명",
                    "submission_deadline": "날짜(없으면 null)"
                }}
            }}

            분석할 문서:
            {text}"""

        return f"{base_prompt}\n{text}"

    def _process_scoring_categories(self, scoring_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """평가 카테고리 처리"""
        try:
            categories = []
            for category_name, details in scoring_data.items():
                if isinstance(details, dict):
                    category = {
                        'name': category_name,
                        'score': sum(details.values()) if details else 0,
                        'subcategories': [
                            {'name': k, 'score': v} for k, v in details.items()
                        ]
                    }
                    categories.append(category)
            return categories
        except Exception as e:
            self.logger.warning(f"평가 카테고리 처리 실패: {str(e)}")
            return []

    async def _extract_evaluation_criteria(self, sections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """평가 기준 추출"""
        try:
            criteria = []
            if '평가항목' in sections:
                eval_content = sections['평가항목']
                if isinstance(eval_content, str):
                    # 텍스트에서 평가 기준 추출
                    lines = eval_content.split('\n')
                    for line in lines:
                        if re.search(r'[0-9]+\.*\s*[가-힣]', line):
                            criteria.append({
                                'description': line.strip(),
                                'category': '정성평가'
                            })
            return criteria
        except Exception as e:
            self.logger.warning(f"평가 기준 추출 실패: {str(e)}")
            return []

    def _calculate_total_score(self, scoring_data: Dict[str, Any]) -> int:
        """총점 계산"""
        try:
            total = 0
            for category in scoring_data.values():
                if isinstance(category, dict):
                    total += sum(category.values())
            return total
        except Exception as e:
            self.logger.warning(f"총점 계산 실패: {str(e)}")
            return 100  # 기본값
    
    async def process_document(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """문서 처리 및 초기 분석"""
        try:
            # 현재 제안서 상태 확인
            if not self.current_proposal:
                raise ProposalServerError("ps.py, process_document,활성화된 제안서가 없습니다. initialize_proposal을 먼저 호출하세요.")
            
            self._validate_workflow_state([WorkflowStatus.INITIALIZED])

            content = file_data.get("content")
            file_type = file_data.get("file_type")

            if not content:
                raise ValidationError("문서 내용이 없습니다")

            # 파일 타입에 따른 처리
            if file_type in ['hwp', 'hwpx']:
                from .dc import DocumentProcessor
                doc_processor = DocumentProcessor()
                doc_result = doc_processor._process_hwp(content)
                
                if not doc_result:
                    raise ProposalServerError("HWP 파일 처리에 실패했습니다")
                    
                text_content = doc_result['full_text']
                table_sections = doc_result.get('table_sections', [])
                
            else:  # PDF 처리
                from .dc import DocumentProcessor
                doc_processor = DocumentProcessor()
                doc, page_limit = doc_processor.process_file(content)
                
                if not doc:
                    raise ProposalServerError("PDF 파일 처리에 실패했습니다")
                    
                text_content = "\n".join(
                    page.extract_text() for page in (doc.pages if hasattr(doc, 'pages') else doc)
                )
                table_sections = []

            if not text_content.strip():
                raise ValidationError("텍스트 추출에 실패했습니다")
            
            #현재 제안서 상태 업데이트
            self.current_proposal.update({
                'content': text_content,
                'table_sections': table_sections,
                'metatdata' : {
                    'file_type': file_type,
                    'page_limit': getattr(doc, 'page_limit', None) if 'doc' in locals() else None,
                    'processed_at': datetime.now().isoformat()
                }
            })

            self.workflow_status = WorkflowStatus.FILE_PROCESSED
            
            return {
                'text': text_content,
                'table_sections': table_sections,
                'metadata': {
                    'page_limit': getattr(doc, 'page_limit', None) if 'doc' in locals() else None,
                    'file_type': file_type
                }
            }

        except Exception as e:
            self.logger.error(f"문서 처리 실패: {str(e)}\n{traceback.format_exc()}")
            raise ProposalServerError(f"문서 처리 실패: {str(e)}")
    
    def _map_standard_sections(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """표준 섹션 매핑"""
        mapped = {}
        for key, value in sections.items():
            # 현재 섹션 이름이 어떤 표준 섹션에 매핑되는지 확인
            for template_key in self.TOC_TEMPLATES.keys():
                if any(variant in key.lower() for variant in self.SECTION_MAPPINGS.get(template_key, [])):
                    mapped[template_key] = value
                    break
            
            # 매핑되지 않은 경우 원본 키 사용
            if key not in mapped:
                mapped[key] = value
                
        return mapped
    

    