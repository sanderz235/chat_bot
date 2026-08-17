# 聊天机器人 CLI 界面

import os
import sys

from core.model_client import QwenVLClient
from core.context_manager import ContextManager
from core.prompt_builder import PromptBuilder


# 图片输入标记语法：[image:路径]
IMAGE_PREFIX = "[image:"
IMAGE_SUFFIX = "]"


def parse_input(user_input: str) -> tuple[str, list[str]]:
    # 解析用户输入，分离文本和图片路径
    text_parts = []
    image_paths = []

    i = 0
    while i < len(user_input):
        idx = user_input.find(IMAGE_PREFIX, i)
        if idx == -1:
            text_parts.append(user_input[i:])
            break

        text_parts.append(user_input[i:idx])
        end_idx = user_input.find(IMAGE_SUFFIX, idx)
        if end_idx == -1:
            text_parts.append(user_input[idx:])
            break

        image_path = user_input[idx + len(IMAGE_PREFIX):end_idx].strip()
        image_paths.append(image_path)
        i = end_idx + 1

    text = "".join(text_parts).strip()
    return text, image_paths


def build_user_content(text: str, image_paths: list[str]):
    # 构建用户消息的 content 字段
    # 纯文本时返回 str，含图片时返回 list（OpenAI 多模态格式）
    if not image_paths:
        return text
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": path},
        })
    return content


def print_banner():
    # 打印欢迎横幅
    print("=" * 60)
    print("   🤖 多模态聊天机器人")
    print("   - 模型: Qwen-VL (多模态，单模型架构)")
    print("   - 提示词工程: 角色提示 + 思维链 (CoT)")
    print("   - 上下文记忆: VLM 多模态语义压缩")
    print("=" * 60)
    print()
    print("使用说明：")
    print("  直接输入文本进行对话")
    print("  输入 [image:图片路径] 附加图片，如：")
    print('    看看这张图 [image:./photo.jpg] 里有什么？')
    print()
    print("特殊命令：")
    print("  /clear    - 清空对话上下文")
    print("  /history  - 查看上下文摘要")
    print("  /compress - 手动压缩上下文")
    print("  /exit     - 退出程序")
    print()


def check_config():
    # 检查必要的配置项
    if not os.getenv("QWEN_API_KEY"):
        print("[错误] QWEN_API_KEY 未设置，无法继续。")
        print()
        print("请设置环境变量后重新运行，例如：")
        print('  $env:QWEN_API_KEY="sk-xxx"')
        print()
        return False
    return True


def main():
    # 主循环
    print_banner()

    if not check_config():
        sys.exit(1)

    # 初始化各模块（单 VLM 架构）
    vlm_client = QwenVLClient()
    context_manager = ContextManager()
    prompt_builder = PromptBuilder()

    print("[系统] 初始化完成，开始对话！\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[系统] 再见！")
            break

        if not user_input:
            continue

        # 处理特殊命令
        if user_input == "/exit":
            print("[系统] 再见！")
            break

        elif user_input == "/clear":
            context_manager.clear()
            print("[系统] 上下文已清空")
            continue

        elif user_input == "/history":
            if context_manager.summary:
                print(f"[上下文摘要]\n{context_manager.summary}")
            else:
                print("[系统] 暂无上下文摘要")
            continue

        elif user_input == "/compress":
            compressed, msg = context_manager.compress(force=True)
            print(f"[系统] {msg}")
            continue

        # 解析输入，提取图片路径
        text, image_paths = parse_input(user_input)

        # 验证图片路径
        valid_paths = []
        for path in image_paths:
            if not os.path.exists(path):
                print(f"[警告] 图片文件不存在: {path}")
            else:
                valid_paths.append(path)

        # 构建用户消息（图片直接作为 image_url，不做 OCR/描述）
        user_content = build_user_content(text, valid_paths)

        # 添加到上下文
        context_manager.add_user_message(user_content)

        # 构建提示词
        context_messages = context_manager.get_context_messages()
        final_messages = prompt_builder.build(context_messages)

        # 调用 VLM
        print("\n助手: ", end="", flush=True)
        response = vlm_client.chat(final_messages)
        print(response)

        # 添加助手回复到上下文
        context_manager.add_assistant_message(response)

        # 显示上下文使用率
        print(f"\n{context_manager.stats_str()}")

        # 自动压缩检查（80% 硬触发）
        if context_manager.should_compress():
            compressed, msg = context_manager.compress()
            if compressed:
                print(f"[系统] {msg}")
                print(context_manager.stats_str())

        print()


if __name__ == "__main__":
    main()