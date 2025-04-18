import time
import uuid
from pprint import pprint

from transformers import pipeline
import torch
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from unsloth import FastModel


app = FastAPI(title="Gemma Chat API")

# 定义请求模型 - 使用OpenAI Chat接口数据结构
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 128

# 定义响应模型 - 使用OpenAI Chat接口数据结构
class Choice(BaseModel):
    index: int
    message: Message

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    choices: List[Choice]

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat(payload: ChatRequest = Body(..., example={
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant. help me translate the text from English to Chinese."
        },
        {
            "role": "user",
            "content": "what can i do with python, and how can i learn python?"
        }
    ],
    "max_tokens": 100
})):
    """
    聊天API端点 - 兼容OpenAI Chat接口
    
    接收消息列表并返回模型生成的回复
    """
    # 将OpenAI格式转换为Gemma模型所需格式
    # 确保消息列表符合user/assistant交替的格式
    messages = payload.messages.copy()
    # 构建Gemma格式的消息列表
    messages = []
    for msg in payload.messages:
        gemma_message = {
            "role": msg.role,
            "content": [{"type": "text", "text": msg.content}]
        }
        messages.append(gemma_message)
    # 生成回复
    pprint(messages)
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens = 128, # Use requested max_tokens or default
        # Recommended Gemma-3 settings!
        temperature = 1.0, top_p = 0.95, top_k = 64,
    )
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    # Extract the assistant's response
    # Remove the prompt part to get only the generated response
    response_text = decoded_outputs.split("<start_of_turn>model\n")[-1].split("</s>")[0].strip().replace("<end_of_turn>", "").strip()
    print('response_text', response_text)
    # 构建OpenAI格式的响应
    
    return ChatResponse(
        id=f"chatcmpl-{str(uuid.uuid4())}",
        created=int(time.time()),
        choices=[
            Choice(
                index=0,
                message=Message(
                    role="assistant",
                    content=response_text
                )
            )
        ],
    )

# 示例用法
example_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "who are you?"}
]

"""
API响应示例:
{
  "id": "chatcmpl-123abc456def",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "gemma-3-1b-it",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello there! I'm Gemma, a large language model created by the Gemma team at Google DeepMind. I'm an open-weights model, which means I'm publicly available for anyone to use! I'm designed to take text as input and generate text as output. It's nice to meet you! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 75,
    "total_tokens": 100
  }
}
"""
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="API server for language model")
    parser.add_argument("-mode", "--mode", type=str, default="default", help="Mode to run the API in (e.g. 'ft' for fine-tuned model)")
    args = parser.parse_args()
    
    if args.mode == "ft":
        print("使用微调模型")
        # You can add specific configuration for fine-tuned model here

        # ft 模型
        model_path = "../models/gemma-3-4b-it-finetune-lora"
        model, tokenizer = FastModel.from_pretrained(
            model_name = model_path, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = 1024,
            load_in_4bit = True,
        )
    else:
        print("使用默认模型")
        model_path = "../models/gemma-3-4b-it"
        model, tokenizer = FastModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 1024, # Choose any for long context!
        )


    uvicorn.run(app, host="0.0.0.0", port=6006)
