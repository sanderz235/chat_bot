# 模型调用层（纯 HTTP 请求，不依赖任何 LLM 框架）
# 单 VLM 架构：Qwen-VL 多模态模型，同时处理文本推理和图片理解

import json
import base64
import os
from typing import Optional

import requests

from config import (
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL,
    QWEN_MAX_TOKENS,
    QWEN_TEMPERATURE,
    SUMMARY_MAX_TOKENS,
)


def _get_mime_type(path: str) -> str:
    # 根据文件扩展名推断 MIME 类型
    ext = os.path.splitext(path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/png")


class QwenVLClient:

    def __init__(self):
        self.api_key = QWEN_API_KEY
        self.base_url = QWEN_BASE_URL.rstrip("/")
        self.model = QWEN_MODEL
        self.max_tokens = QWEN_MAX_TOKENS
        self.temperature = QWEN_TEMPERATURE

    def _convert_image_urls(self, messages: list[dict]) -> list[dict]:
        # 将 messages 中的本地文件路径 image_url 转换为 base64 data URL
        result = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                result.append(msg)
                continue

            new_content = []
            for part in content:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    # 已经是 data: 或 http(s): 格式，不转换
                    if not url.startswith("data:") and not url.startswith("http"):
                        if os.path.exists(url):
                            with open(url, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode("utf-8")
                            mime = _get_mime_type(url)
                            url = f"data:{mime};base64,{b64}"
                    new_content.append({
                        "type": "image_url",
                        "image_url": {"url": url},
                    })
                else:
                    new_content.append(part)
            result.append({**msg, "content": new_content})
        return result

    def chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        # 通用对话接口，支持纯文本和多模态（含图片）消息
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self._convert_image_urls(messages),
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }

        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"[Qwen API 调用失败] {e}"
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return f"[Qwen 响应解析失败] {e}\n原始响应: {resp.text[:500]}"

    def describe_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        # 对图片进行场景描述
        if not prompt:
            prompt = "请详细描述这张图片的内容，包括：场景、物体、人物、动作、氛围、颜色、文字（如果有的话）。用中文回答。"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_path}},
                ],
            }
        ]
        return self.chat(messages)