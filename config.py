# 配置文件（所有配置项从 .env 文件读取）

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


def require_env(key: str) -> str:
    # 从环境变量获取必填配置
    value = os.getenv(key)
    if not value:
        raise ValueError(f"缺少必要的环境变量配置: {key}，请在 .env 文件中设置")
    return value


QWEN_API_KEY = require_env("QWEN_API_KEY")
QWEN_BASE_URL = require_env("QWEN_BASE_URL")
QWEN_MODEL = require_env("QWEN_MODEL")
QWEN_MAX_TOKENS = int(require_env("QWEN_MAX_TOKENS"))
QWEN_TEMPERATURE = float(require_env("QWEN_TEMPERATURE"))
QWEN_CONTEXT_WINDOW = int(require_env("QWEN_CONTEXT_WINDOW"))

# 压缩时摘要输出的 token 限制
SUMMARY_MAX_TOKENS = int(require_env("SUMMARY_MAX_TOKENS"))

# 上下文记忆配置
# 保留窗口目标占比（40%），单条消息容忍上限（60%）
RETENTION_TARGET_RATIO = 0.4
RETENTION_MAX_RATIO = 0.6
# 硬触发阈值：上下文使用率超过此值自动压缩
COMPRESS_TRIGGER_RATIO = 0.8


SYSTEM_ROLE_PROMPT = """你是一个专业、严谨、细心的AI助手。你的回答应该：
1. 准确、有条理，基于事实和逻辑
2. 当用户提供图片时，仔细观察图片内容并结合文字进行分析
3. 如果信息不足，坦诚说明，不编造内容
4. 使用清晰的中文回答，必要时使用结构化格式（列表、分点等）"""

# 思维链后缀指令
COT_SUFFIX = "\n\n请逐步思考：先分析问题的关键点，理清逻辑，再给出最终答案。"