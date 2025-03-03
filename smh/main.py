# main.py 수정
import os
import logging
import json
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel

# 기본 FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import UploadFile, File, HTTPException 
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 환경변수 관련
from dotenv import load_dotenv

# AI 모델 관련
import google.generativeai as genai
from anthropic import AsyncAnthropic
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 파일 처리 관련
from langchain_teddynote.document_loaders import HWPLoader
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document
import PyPDF2
from PyPDF2 import PdfReader

# utils imports
from utils.pg import ProposalGenerator 
from utils.dc import DocumentProcessor
from utils.agent import AgentController
from utils.ps import ProposalServer

# transformers 관련 (마지막에 import)
from transformers import AutoTokenizer
from llama_cpp import Llama
from huggingface_hub import InferenceClient

from prop import router as proposal_router

import re

load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI Application Setup
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(proposal_router)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class WSMessage(BaseModel):
    """WebSocket message model"""
    type: str
    content: Any
    model: str = "meta"

class FileAnalysis(BaseModel):
    """File analysis response model"""
    file_id: str
    status: str
    filename: str
    sections: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

# AI Model Manager
class AIModelManager:
    def __init__(self):
        self.setup_models()
        try:
            # AgentController 초기화 추가
            self.agent = AgentController()
            logger.info("AgentController initialized successfully")
        except Exception as e:
            logger.error(f"AgentController initialization failed: {e}")
            raise
    
    def setup_models(self):
        """Initialize AI models"""
        try:
            # Claude setup
            self.anthropic = AsyncAnthropic(api_key=os.getenv('CLAUDE_API_KEY'))
            
            # Gemini setup
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.gemini_config = {
                'temperature': 0.9,
                'top_p': 1,
                'top_k': 1,
                'max_output_tokens': 3000,
            }
            self.gemini_model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=self.gemini_config
            )
            
            # Local LLM setup
            self.local_model = self._initialize_local_model()
            self.tokenizer = AutoTokenizer.from_pretrained('Bllossom/llama-3.2-Korean-Bllossom-3B')
            
            logger.info("AI models initialized successfully")
        except Exception as e:
            logger.error(f"AI model initialization failed: {e}")
            raise

    def _initialize_local_model(self):
        """Initialize local LLM with GPU support if available"""
        MODEL_PATH = r"C:\Users\MyoengHo Shin\pjt\newkl\smh\llama-korean\llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf"
        try:
            return Llama(
                model_path=MODEL_PATH,
                n_ctx=2048,
                n_threads=8,
                n_batch=1024,
                n_gpu_layers=32,
                f16_kv=True,
                offload_kqv=True
            )
        except Exception as e:
            logger.warning(f"GPU initialization failed, falling back to CPU: {e}")
            return Llama(
                model_path=MODEL_PATH,
                n_ctx=2048,
                n_threads=8,
                n_batch=512,
                n_gpu_layers=0
            )

    async def generate_response(self, prompt: str, model: str, websocket: WebSocket) -> str:
        """Generate response using specified model"""
        try:
            if model == "gemini":
                return await self._generate_gemini_response(prompt, websocket)
            elif model == "claude":
                return await self._generate_claude_response(prompt, websocket)
            else:  # meta
                return await self._generate_local_response(prompt, websocket)
        except Exception as e:
            logger.error(f"Error generating response with {model}: {e}")
            raise

    async def _generate_gemini_response(self, prompt: str, websocket: WebSocket) -> str:
        response = await self.gemini_model.generate_content_async(prompt)
        return self._format_gemini_response(response.text)

    async def _generate_claude_response(self, prompt: str, websocket: WebSocket) -> str:
        stream = await self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        response = ""
        async for chunk in stream:
            if chunk.type == "content_block_delta" and chunk.delta:
                response += chunk.delta.text
                await self._send_chunk(websocket, chunk.delta.text, "claude")
        
        return response

    async def _generate_local_response(self, prompt: str, websocket: WebSocket) -> str:
        formatted_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=False
        )
        
        response = ""
        for chunk in self.local_model(formatted_prompt, max_tokens=512, stream=True):
            if chunk and "choices" in chunk:
                text = chunk["choices"][0]["text"]
                if text:
                    response += text
                    await self._send_chunk(websocket, text, "meta")
        
        return response

    async def _send_chunk(self, websocket: WebSocket, text: str, model: str):
        """Send response chunk through WebSocket"""
        await websocket.send_text(json.dumps({
            "type": "assistant",
            "content": text,
            "streaming": True,
            "model": model
        }))

    def _format_gemini_response(self, text: str) -> str:
        """Format Gemini response"""
        formatted = text.replace('•', '* ')
        return formatted

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.conversation_histories: Dict[WebSocket, List[Dict]] = {}
        self.file_contexts: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.conversation_histories[websocket] = []
        self.file_contexts[websocket] = None

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.conversation_histories.pop(websocket, None)
        self.file_contexts.pop(websocket, None)

    def add_to_history(self, websocket: WebSocket, message: Dict):
        if websocket not in self.conversation_histories:
            self.conversation_histories[websocket] = []
        self.conversation_histories[websocket].append(message)

    def get_history(self, websocket: WebSocket) -> List[Dict]:
        return self.conversation_histories.get(websocket, [])

    def store_file_context(self, websocket: WebSocket, context: Dict):
        self.file_contexts[websocket] = context

    def get_file_context(self, websocket: WebSocket) -> Optional[Dict]:
        return self.file_contexts.get(websocket)



# Initialize managers and processors
model_manager = AIModelManager()
connection_manager = ConnectionManager()
doc_processor = DocumentProcessor()
proposal_generator = ProposalGenerator()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve main chat interface"""
    with open("static/home.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/prop", response_class=FileResponse)
async def read_prop():
    """Serve proposal generator interface"""
    return FileResponse("static/prop.html")

# ... existing code ...

@app.websocket("/chat")
async def chat_endpoint(websocket: WebSocket):
    """Chat WebSocket endpoint for handling chat interactions"""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # 메시지 수신 및 파싱
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # 메시지 유효성 검증
            if not all(key in message_data for key in ["type", "content"]):
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": "Invalid message format"
                }))
                continue
                
            # 사용자 메시지 처리
            user_message = message_data.get("content", "")
            model_type = message_data.get("model", "meta")
            
            # 채팅 이력에 사용자 메시지 추가
            connection_manager.add_to_history(websocket, {
                "type": "user",
                "content": user_message,
                "model": model_type
            })

            try:
                # 모델별 응답 생성 및 로깅 추가
                logger.info(f"Processing message: '{user_message}' with model: {model_type}")
                
                if model_type == "gemini":
                    # 직접 Gemini API 호출 (AgentController 우회)
                    try:
                        # 프롬프트 생성
                        prompt = create_context_prompt(
                            user_message, 
                            connection_manager.get_file_context(websocket)
                        )
                        
                        logger.info(f"Sending prompt to Gemini: {prompt[:100]}...")
                        
                        # 직접 Gemini API 호출
                        response = await model_manager.gemini_model.generate_content_async(prompt)
                        
                        # 응답 로깅
                        logger.info(f"Raw Gemini response type: {type(response)}")
                        logger.info(f"Raw Gemini response: {response}")
                        
                        # 텍스트 추출
                        if hasattr(response, 'text'):
                            response_text = response.text
                        elif hasattr(response, 'parts'):
                            response_text = ' '.join([part.text for part in response.parts])
                        else:
                            response_text = str(response)
                        
                        logger.info(f"Extracted Gemini response text: {response_text[:100]}...")
                        
                        # 마크다운 구조 보존을 위해 단락 단위로 분할하여 전송
                        # 빈 줄로 구분된 단락으로 나누기
                        paragraphs = response_text.split('\n\n')
                        
                        for paragraph in paragraphs:
                            if not paragraph.strip():  # 빈 단락 건너뛰기
                                continue
                                
                            # 단락 내에 줄바꿈이 있는 경우 (마크다운 리스트 등) 보존
                            lines = paragraph.split('\n')
                            
                            for line in lines:
                                if not line.strip():  # 빈 줄 건너뛰기
                                    continue
                                
                                # 각 줄 전송
                                await websocket.send_text(json.dumps({
                                    "type": "assistant",
                                    "content": line + "\n",  # 줄바꿈 추가
                                    "streaming": True,
                                    "model": "gemini"
                                }))
                                await asyncio.sleep(0.05)  # 스트리밍 효과를 위한 지연
                            
                            # 단락 사이에 빈 줄 추가
                            await websocket.send_text(json.dumps({
                                "type": "assistant",
                                "content": "\n",
                                "streaming": True,
                                "model": "gemini"
                            }))
                            
                        # 응답 완료 신호 전송
                        await websocket.send_text(json.dumps({
                            "type": "assistant",
                            "content": "",
                            "streaming": False,
                            "model": "gemini",
                            "isFullResponse": True
                        }))
                        
                        response = response_text
                        
                    except Exception as gemini_error:
                        logger.error(f"Gemini API error: {gemini_error}")
                        raise
                        
                else:
                    # Claude/Meta 모델 응답 생성
                    prompt = create_context_prompt(
                        user_message, 
                        connection_manager.get_file_context(websocket)
                    )
                    
                    response = await model_manager.generate_response(
                        prompt=prompt,
                        model=model_type,
                        websocket=websocket
                    )
                    
                    # 응답 완료 신호 전송
                    await websocket.send_text(json.dumps({
                        "type": "assistant",
                        "content": "",
                        "streaming": False,
                        "model": model_type,
                        "isFullResponse": True
                    }))
                
                # 채팅 이력에 어시스턴트 응답 추가
                connection_manager.add_to_history(websocket, {
                    "type": "assistant",
                    "content": response,
                    "model": model_type
                })
                
                logger.info(f"Response successfully sent for model: {model_type}")
                
            except Exception as e:
                logger.error(f"Response generation error: {str(e)}")
                logger.error(f"Error details: {type(e).__name__}, {e.args}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": f"Error generating response: {str(e)}",
                    "streaming": False
                }))
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        connection_manager.disconnect(websocket)
        
    finally:
        if websocket in connection_manager.active_connections:
            connection_manager.disconnect(websocket)

def create_context_prompt(user_message: str, file_context: Optional[Dict]) -> str:
    """컨텍스트 기반 프롬프트 생성"""
    if file_context and file_context.get('sections'):
        sections_info = []
        for key, value in file_context['sections'].items():
            if isinstance(value, dict):
                sections_info.append(f"{key}:")
                for k, v in value.items():
                    sections_info.append(f"  - {k}: {v}")
            else:
                sections_info.append(f"{key}: {value}")
        
        context_text = "\n".join(sections_info)
        return (
            f"문서 컨텍스트:\n"
            f"{context_text}\n\n"
            f"사용자 질문: {user_message}\n\n"
            f"위 문서 컨텍스트를 참고하여 사용자의 질문에 답변해주세요."
        )
    
    return f"질문: {user_message}\n\n가능한 한 상세하고 도움이 되는 답변을 제공해주세요."


@app.post("/mainupload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and initial analysis"""
    try:
        # Process file
        content = await file.read()
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in ['.pdf', '.hwp', '.hwpx', '.doc', '.docx']:
            raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

        # Process document
        doc_result = await doc_processor.process_document(content, file_ext)
        if not doc_result:
            raise HTTPException(status_code=400, detail="문서 처리에 실패했습니다.")

        # Extract sections
        sections = await proposal_generator.extract_sections(doc_result['text'])
        
        return {
            "status": "success",
            "filename": file.filename,
            "orientation": doc_result.get('orientation'),
            "page_limit": doc_result.get('page_limit'),
            "sections": sections
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)