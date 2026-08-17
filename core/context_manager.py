# 基于 VLM 多模态语义压缩的上下文管理器
# 上下文使用率超过 80% 时自动压缩，将早期对话（含图片）生成语义摘要

from typing import Optional

from config import (
    QWEN_CONTEXT_WINDOW,
    RETENTION_TARGET_RATIO,
    RETENTION_MAX_RATIO,
    COMPRESS_TRIGGER_RATIO,
    SUMMARY_MAX_TOKENS,
)
from core.model_client import QwenVLClient


class ContextManager:
    # 对话上下文管理器，使用 VLM 多模态语义压缩策略维护长期记忆

    def __init__(self):
        self.messages: list[dict] = []  # 当前对话消息（含图片）
        self.summary: str = ""  # 累积语义摘要
        self.context_window = QWEN_CONTEXT_WINDOW
        self._vlm = QwenVLClient()

    # 消息管理
    def add_user_message(self, content):
        # 添加用户消息
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        # 添加助手回复
        self.messages.append({"role": "assistant", "content": content})

    def get_context_messages(self) -> list[dict]:
        # 获取当前上下文 messages，供 VLM 调用使用
        # 如果有摘要，在最前面插入为 system 消息
        result = []
        if self.summary:
            result.append({
                "role": "system",
                "content": f"[历史对话摘要]\n{self.summary}"
            })
        result.extend(self.messages)
        return result

    def clear(self):
        # 清空所有上下文。
        self.messages.clear()
        self.summary = ""

    # Token 估算
    @staticmethod
    def _estimate_image_tokens(path: str) -> int:
        # 根据图片分辨率估算 Qwen-VL 视觉 token 数
        try:
            from PIL import Image
            img = Image.open(path)
            w, h = img.size
            return max(256, min(4096, (h // 14) * (w // 14)))
        except Exception:
            return 500  # 读取失败时用默认值

    @classmethod
    def _estimate_message_tokens(cls, msg: dict) -> int:
        # 估算单条消息的 token 数
        content = msg.get("content", "")
        if isinstance(content, str):
            return int(len(content) * 1.5)
        if isinstance(content, list):
            total = 0
            for part in content:
                if part.get("type") == "text":
                    total += len(part.get("text", "")) * 1.5
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    # 跳过已经是 base64 或 http 的 URL
                    if not url.startswith("data:") and not url.startswith("http"):
                        total += cls._estimate_image_tokens(url)
                    else:
                        total += 500  # 已编码的图片用默认值
            return int(total)
        return 0

    def _estimate_total_tokens(self) -> int:
        # 估算当前 messages 的总 token 数（不含摘要）
        return sum(self._estimate_message_tokens(m) for m in self.messages)

    def _count_images(self) -> int:
        # 统计当前 messages 中的图片数量
        count = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        count += 1
        return count

    # 状态查询
    def usage_pct(self) -> float:
        # 返回当前上下文 token 使用率（百分比）
        total = self._estimate_total_tokens()
        return (total / self.context_window) * 100

    def get_stats(self) -> tuple[float, int, int]:
        # 返回统计信息: (使用率%, 消息条数, 图片张数)
        return (
            round(self.usage_pct(), 1),
            len(self.messages),
            self._count_images(),
        )

    def stats_str(self) -> str:
        # 格式化统计信息为字符串
        pct, msgs, imgs = self.get_stats()
        return f"[上下文: {pct:.0f}% | 消息: {msgs} 条 | 图片: {imgs} 张]"

    # 压缩逻辑
    
    def should_compress(self) -> bool:
        # 检查是否达到硬触发阈值（80%）  
        return self.usage_pct() >= COMPRESS_TRIGGER_RATIO * 100

    def compress(self, force: bool = False) -> tuple[bool, str]:
        # 执行上下文压缩
        if not self.messages:
            return False, "上下文为空，无需压缩"

        before_pct = self.usage_pct()

        if not force and before_pct < COMPRESS_TRIGGER_RATIO * 100:
            return False, (
                f"上下文使用率 {before_pct:.0f}%，"
                f"未达压缩阈值 {COMPRESS_TRIGGER_RATIO*100:.0f}%，不执行压缩"
            )

        # 1. 确定保留窗口
        target_tokens = self.context_window * RETENTION_TARGET_RATIO
        max_msg_tokens = self.context_window * RETENTION_MAX_RATIO
        retained = []
        retained_tokens = 0

        for msg in reversed(self.messages):
            msg_tokens = self._estimate_message_tokens(msg)
            if retained_tokens + msg_tokens <= max_msg_tokens:
                retained.insert(0, msg)
                retained_tokens += msg_tokens
            else:
                break

        to_compress = self.messages[:len(self.messages) - len(retained)]
        if not to_compress:
            return False, "没有可压缩的历史消息（保留窗口已覆盖全部上下文）"

        # 2. 构建压缩请求
        print(f"  [压缩] 正在压缩 {len(to_compress)} 条历史消息...")
        new_summary = self._do_compress(to_compress)

        # 3. 替换
        self.messages = retained
        if new_summary:
            self.summary = new_summary
        else:
            print("  [压缩] 摘要生成失败，回退消息")

        after_pct = self.usage_pct()
        result_msg = (
            f"压缩前: {before_pct:.0f}% | 压缩后: {after_pct:.0f}% | "
            f"已将 {len(to_compress)} 条消息压缩为摘要"
        )
        print(f"  [压缩] {result_msg}")
        return True, result_msg

    def _do_compress(self, to_compress: list[dict]) -> Optional[str]:
        # 将待压缩消息（含图片）发送给 VLM，生成语义摘要。
        # VLM 在压缩时能看到图片，但摘要输出为纯文本，
        # 图片在压缩后从上下文移除，语义被保留在摘要中
        old_summary = self.summary

        # 构建压缩 messages
        compression_messages = []
        if old_summary:
            compression_messages.append({
                "role": "system",
                "content": f"之前的对话摘要：\n{old_summary}",
            })
        compression_messages.extend(to_compress)
        compression_messages.append({
            "role": "user",
            "content": (
                "请将以上对话历史总结为一段连贯的语义摘要，200 字以内。\n\n"
                "要求：\n"
                "1. 总结「讨论了什么话题」，而不是「每张图里有什么」\n"
                " 如：用户分享了股市行情图，讨论了上证指数市场表现...\n"
                " 而不是：用户发了一张截图，上面有：开盘价：1,145.14...\n"
                "2. 说明图片在讨论中扮演的角色（如：报错截图、架构图、设计稿），"
                "不要复述图片内容\n"
                "3. 保留关键结论、决策、待办事项\n"
                "4. 如果前面有之前的摘要，请和新内容合并为一段，去掉重复和矛盾的信息"
            ),
        })

        try:
            summary = self._vlm.chat(
                messages=compression_messages,
                max_tokens=SUMMARY_MAX_TOKENS,
                temperature=0.3,
            )
            return summary.strip()
        except Exception as e:
            print(f"  [压缩] VLM 调用失败: {e}")
            return None