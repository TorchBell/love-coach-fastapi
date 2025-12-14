from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# OpenAI Client Setup (Ensure OPENAI_API_KEY is set in .env or environment)
# For GMS compatibility, you might need to set base_url and api_key specifically.
# Assuming standard OpenAI or compatible API.
client = OpenAI(
    api_key=os.getenv("GMS_API_KEY", "your-api-key"),
    base_url=os.getenv("GMS_URL", "https://api.openai.com/v1")
)
MODEL_NAME = os.getenv("GMS_MODEL", "gpt-4o-mini")

class ChatLogItem(BaseModel):
    userId: int
    npcId: int
    messageUser: Optional[str] = None
    messageAi: Optional[str] = None
    context: Optional[str] = None

class ChatRequest(BaseModel):
    system_prompt: str
    context: str
    chat_log: List[ChatLogItem]
    new_chat: str

class ContextRequest(BaseModel):
    context: str
    chat_log: List[ChatLogItem]
    user_message: str
    ai_message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        messages = [{"role": "system", "content": request.system_prompt + "\n[Context Info]: " + request.context}]
        
        # Reverse chat log if needed (assuming incoming list is latest first)
        # Java side sends the list as is. If it's latest first, we reverse it here.
        # Based on Java code: "chatLogList는 최신순(DESC)"
        reversed_logs = request.chat_log[::-1]
        
        for log in reversed_logs:
            if log.messageUser and log.messageAi:
                messages.append({"role": "user", "content": log.messageUser})
                messages.append({"role": "assistant", "content": log.messageAi})
        
        messages.append({"role": "user", "content": request.new_chat})

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context")
async def generate_context(request: ContextRequest):
    try:
        summary_instruction = (
            "당신은 대화의 핵심 맥락을 요약하고 관리하는 AI입니다. \n"
            "주어진 [기존 문맥]과 [최근 대화 내역]을 종합하여, 다음 대화에서 캐릭터가 참고할 수 있는 '새로운 문맥'을 작성해주세요.\n"
            "사용자의 취향, 중요한 사건, 대화의 흐름 등을 중심으로 간결하게 요약하세요."
        )
        
        recent_chat = ""
        # Use last 5 logs
        recent_logs = request.chat_log[:5]
        reversed_logs = recent_logs[::-1]
        
        for log in reversed_logs:
             if log.messageUser and log.messageAi:
                recent_chat += f"User: {log.messageUser}\nAI: {log.messageAi}\n"
        
        recent_chat += f"User: {request.user_message}\nAI: {request.ai_message}\n"
        
        user_prompt = f"[기존 문맥]: {request.context if request.context else '없음'}\n\n[최근 대화 내역]:\n{recent_chat}"
        
        messages = [
            {"role": "system", "content": summary_instruction},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        
        return {"new_context": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
