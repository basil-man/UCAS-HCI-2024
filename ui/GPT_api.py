import os
import openai
from openai import OpenAI
from tqdm import tqdm
import json

# 设置 OPENAI_API_KEY 和 BASE_URL 环境变量
os.environ["OPENAI_API_KEY"] = "sk-h2GW3tgdL6sGEjQENcXCjclgVPH1ErCaoNSxcouC4JPdZiZP"  # 替换为你的 OpenAI API 密钥
os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"  # 替换为自定义的 API URL

# 使用 OpenAI 客户端类
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)


def generate_prompt(action, oral_advice, max_retries=3):
    """
    使用 GPT API 根据 domain 和 class 生成相关图片的提示词，最多尝试 max_retries 次。
    """
    prompt_description = f"下面是我对于{action}的建议：{oral_advice}，我觉得这个建议还不够完善，你可以再补充一下吗？"

    retry_count = 0

    while retry_count < max_retries:
        try:
            # 调用 OpenAI API 生成提示词，使用 chat.completions.create 接口，添加 temperature 和 top_p 参数
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Only return content directly as requested. 所有的回复全都用中文",
                    },
                    {"role": "user", "content": prompt_description},
                ],
                temperature=0.9,  # 设置较高的 temperature 以增加随机性
                top_p=0.9,  # 使用较高的 top_p 来增强多样性
                frequency_penalty=0.5,  # 减少重复性
                presence_penalty=0.5,  # 鼓励生成新的内容
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating advice for {action}: {e}")
            retry_count += 1
            print(f"Retrying... ({retry_count}/{max_retries})")

    print(f"GPTapi: Failed to generate advice for {action} after {max_retries} retries.")
    return None
