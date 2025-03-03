import logging
import re
import io
import olefile
from typing import Optional, Tuple, Dict, Any, List
from fastapi import HTTPException
from PyPDF2 import PdfReader
from langchain_teddynote.document_loaders import HWPLoader

class DocumentProcessor:
    def __init__(self):
        try:
            self.logger = logging.getLogger(__name__)
            self.page_pattern = re.compile(r'(\d+)\s*(페이지|쪽|매)\s*(이내|이하|미만|까지)')
            
            # fitz 라이브러리 임포트 시도
            try:
                import fitz
                self.fit_lib = fitz
                self.logger.info("fitz library successfully imported")
            except ImportError:
                self.fit_lib = None
                self.logger.warning("fitz library not found, will try PyMuPDF as fallback")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize DocumentProcessor: {str(e)}")
            raise HTTPException(status_code=500, detail="DocumentProcessor initialization failed")

    @staticmethod
    def process_file(file_content: bytes) -> Tuple[Optional[Any], Optional[int]]:
        """파일을 처리하고 문서 객체와 페이지 제한을 반환합니다."""
        try:
            pdf_stream = io.BytesIO(file_content)
            doc = None

            try:
                import fitz
                doc = fitz.open(stream=pdf_stream, filetype="pdf")
                logging.info("PDF 파일을 성공적으로 열었습니다 (fitz)")
            except Exception:
                doc = PdfReader(pdf_stream)
                
            page_limit = DocumentProcessor.extract_page_limit(doc)
            return doc, page_limit
            
        except Exception as e:
            logging.error(f"파일 처리 중 오류 발생: {str(e)}")
            return None, None

    @staticmethod
    def _process_hwp(file_content: bytes) -> Dict[str, Any]:
        """HWP 파일을 처리하고 텍스트와 표 구조를 추출합니다."""
        try:
            hwp_stream = io.BytesIO(file_content)
            hwp_loader = HWPLoader("dummy_path")
            load_file = olefile.OleFileIO(hwp_stream)
            
            if not hwp_loader._is_valid_hwp(load_file.listdir()):
                raise ValueError("유효하지 않은 HWP 파일입니다.")
            
            text = hwp_loader._extract_text(load_file, load_file.listdir())
            
            # 표 구조 식별을 위한 패턴
            table_patterns = [
                r'배점표',
                r'평가항목.*배점',
                r'항목.*점수',
                r'평가기준.*점수',
                r'\d+\s*점.*\d+\s*점',
                r'정성적.*평가.*기준',
                r'정량적.*평가.*기준'
            ]
            
            table_sections = DocumentProcessor._extract_tables(text, table_patterns)
            
            return {
                'full_text': text,
                'table_sections': table_sections
            }

        except Exception as e:
            logging.error(f"HWP 파일 처리 중 오류 발생: {str(e)}")
            return None

    @staticmethod
    def _extract_tables(text: str, patterns: list) -> List[Dict[str, Any]]:
        """텍스트에서 표 구조를 추출하여 딕셔너리 리스트로 반환합니다."""
        table_sections = []
        current_lines = []
        current_title = ""
        in_table = False
        
        for line in text.split('\n'):
            # 테이블 시작 패턴 확인
            pattern_match = next((p for p in patterns if re.search(p, line, re.IGNORECASE)), None)
            
            if pattern_match:
                # 이전 테이블 처리
                if current_lines:
                    table_sections.append({
                        "title": current_title,
                        "content": '\n'.join(current_lines),
                        "type": "table"
                    })
                current_title = line.strip()
                current_lines = []
                in_table = True
                continue
                
            if in_table:
                if re.search(r'[\w\s]+([\d.]+)\s*점', line) or \
                re.search(r'^\s*[0-9.]+\s*$', line) or \
                re.search(r'소계|합계|총점', line):
                    current_lines.append(line)
                elif len(line.strip()) == 0 and current_lines:
                    table_sections.append({
                        "title": current_title,
                        "content": '\n'.join(current_lines),
                        "type": "table"
                    })
                    current_lines = []
                    in_table = False
                    
        # 마지막 테이블 처리
        if current_lines:
            table_sections.append({
                "title": current_title,
                "content": '\n'.join(current_lines),
                "type": "table"
            })
            
        return table_sections

    @staticmethod
    def extract_page_limit(doc: Any) -> Optional[int]:
        """문서에서 페이지 제한을 추출합니다."""
        pattern = re.compile(r'(\d+)\s*(페이지|쪽|매)\s*(이내|이하|미만|까지)')
        
        for page in doc.pages if isinstance(doc, PdfReader) else doc:
            text = page.get_text() if isinstance(doc, PdfReader) else page.get_text("text")
            match = pattern.search(text)
            if match:
                return int(match.group(1))
        return None