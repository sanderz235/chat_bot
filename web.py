# Gradio Web UI

import os

import gradio as gr

from core.model_client import QwenVLClient
from core.context_manager import ContextManager
from core.prompt_builder import PromptBuilder


# 全局单例
vlm_client = QwenVLClient()
context_manager = ContextManager()
prompt_builder = PromptBuilder()


# 辅助函数
def build_user_display(text: str, files: list) -> str | list:
    # 构建发送给 Chatbot 显示的用户消息内容
    if not files:
        return text
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for path in files:
        content.append({"path": path})
    return content


def build_user_content(text: str, files: list):
    # 构建存储到 ContextManager 的用户消息 content（OpenAI 多模态格式）
    if not files:
        return text
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for path in files:
        content.append({
            "type": "image_url",
            "image_url": {"url": path},
        })
    return content


# 主响应函数
def respond(message: dict, chat_history: list):
    # 处理 MultimodalTextbox 的输入，调用 VLM 并更新 Chatbot

    text = (message or {}).get("text", "").strip()
    files = (message or {}).get("files", []) or []

    # 特殊命令处理
    if text == "/clear":
        context_manager.clear()
        yield [], gr.update(value=None), "✅ 上下文已清空"
        return
    if text == "/history":
        summary = context_manager.summary or "暂无上下文摘要"
        yield chat_history, gr.update(value=None), f"📜 上下文摘要:\n{summary}"
        return
    if text == "/compress":
        compressed, msg = context_manager.compress(force=True)
        yield chat_history, gr.update(value=None), f"📦 {msg}"
        return
    if text == "/exit":
        yield chat_history, gr.update(value=None), "⚠️ Web 界面无需退出，关闭浏览器标签页即可"
        return

    if not text and not files:
        yield chat_history, gr.update(value=None), "⚠️ 请输入文本或上传图片"
        return

    # 用户消息入历史（含图片预览）
    user_display = build_user_display(text, files)
    chat_history = chat_history + [{"role": "user", "content": user_display}]
    yield chat_history, gr.update(value=None), "🔄 正在处理输入..."

    # 用户消息入上下文（OpenAI 多模态格式）
    user_content = build_user_content(text, files)
    context_manager.add_user_message(user_content)

    # 构建提示词
    context_messages = context_manager.get_context_messages()
    final_messages = prompt_builder.build(context_messages)

    # 占位助手消息
    chat_history = chat_history + [{"role": "assistant", "content": "思考中..."}]
    yield chat_history, gr.update(), f"🔄 Qwen-VL 正在生成回复... ({context_manager.stats_str()})"

    # 调用 VLM
    response = vlm_client.chat(final_messages)
    context_manager.add_assistant_message(response)
    chat_history[-1] = {"role": "assistant", "content": response}

    # 显示使用率 + 自动压缩检查
    status = f"✅ {context_manager.stats_str()}"
    if context_manager.should_compress():
        compressed, msg = context_manager.compress()
        if compressed:
            status = f"📦 {msg} | {context_manager.stats_str()}"
    yield chat_history, gr.update(), status


def clear_context_handler():
    context_manager.clear()
    return [], "✅ 上下文已清空"


def view_history_handler():
    summary = context_manager.summary or "暂无上下文摘要"
    return f"📜 上下文摘要:\n{summary}"


def compress_handler():
    compressed, msg = context_manager.compress(force=True)
    return f"📦 {msg} | {context_manager.stats_str()}"


# Gradio 界面构建
with gr.Blocks(title="多模态聊天机器人") as demo:
    gr.Markdown(
        "# 🤖 多模态聊天机器人\n"
        "模型: Qwen-VL (单 VLM 架构) | "
        "提示词工程: 角色提示 + 思维链 (CoT) | "
        "上下文记忆: VLM 多模态语义压缩"
    )

    with gr.Row():
        clear_btn = gr.Button("🧹 清空上下文", variant="stop", size="sm")
        history_btn = gr.Button("📜 查看摘要", variant="secondary", size="sm")
        compress_btn = gr.Button("📦 压缩上下文", variant="secondary", size="sm")

    status_box = gr.Textbox(
        label="系统状态",
        interactive=False,
        max_lines=3,
        elem_id="status-box",
        placeholder="状态信息将显示在这里...",
        value="✅ 就绪",
    )

    chatbot = gr.Chatbot(
        label="对话历史",
        height=520,
        layout="bubble",
        buttons=["copy", "copy_all"],
        placeholder="💡 在下方输入文本或上传图片，按 Enter 发送",
    )

    input_box = gr.MultimodalTextbox(
        interactive=True,
        file_types=["image"],
        file_count="multiple",
        placeholder="输入文本，或上传图片后按 Enter 发送...",
        submit_btn="发送",
    )

    gr.Markdown(
        "**命令**：`/clear` 清空上下文 | `/history` 查看摘要 | `/compress` 手动压缩  \n"
        "图片直接进入 VLM 上下文，模型同时看图+推理，无需 OCR 或中间描述。"
    )

    # 事件绑定
    input_box.submit(
        respond,
        inputs=[input_box, chatbot],
        outputs=[chatbot, input_box, status_box],
    )

    clear_btn.click(clear_context_handler, outputs=[chatbot, status_box])
    history_btn.click(view_history_handler, outputs=[status_box])
    compress_btn.click(compress_handler, outputs=[status_box])


if __name__ == "__main__":
    if not os.getenv("QWEN_API_KEY"):
        raise RuntimeError(
            "QWEN_API_KEY 未设置。请在 .env 文件或环境变量中配置后重试。"
        )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 920px !important; margin: auto; }
            #status-box textarea { font-size: 13px !important; }
        """,
    )