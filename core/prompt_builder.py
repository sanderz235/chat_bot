# 提示词：注入角色提示和思维链

from config import SYSTEM_ROLE_PROMPT, COT_SUFFIX


class PromptBuilder:
    # 提示词构建

    def __init__(self):
        self.system_role = SYSTEM_ROLE_PROMPT
        self.cot_suffix = COT_SUFFIX

    def build(self, context_messages: list[dict]) -> list[dict]:
        # 构建最终 messages 列表
        result = []

        # 角色提示：在最前面注入系统角色
        result.append({"role": "system", "content": self.system_role})

        # 添加上下文消息（可能包含摘要 system 消息 + 最近对话）
        result.extend(context_messages)

        # 思维链：在最后一条用户消息中追加 CoT 指令
        for i in range(len(result) - 1, -1, -1):
            if result[i]["role"] == "user":
                content = result[i]["content"]
                if isinstance(content, str):
                    result[i]["content"] = content + self.cot_suffix
                elif isinstance(content, list):
                    # 找到最后一个 text 元素并追加 CoT 指令
                    for j in range(len(content) - 1, -1, -1):
                        if content[j].get("type") == "text":
                            content[j]["text"] = content[j]["text"] + self.cot_suffix
                            break
                break

        return result

    def set_role(self, role_prompt: str):
        # 动态修改角色提示
        self.system_role = role_prompt

    def set_cot(self, cot_instruction: str):
        # 动态修改思维链指令
        self.cot_suffix = cot_instruction