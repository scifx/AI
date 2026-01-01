import os
import json
import base64
import requests
from pathlib import Path

local_api_addr= "http://localhost:11434/v1"
online_api_addr= os.getenv("OPENAI_API_BASE")

local_model= "qwen3:4b"
local_img_model= "qwen3-vl:4b"

online_model="fireworks::accounts/fireworks/models/llama4-maverick-instruct-basic"
online_img_model="fireworks::accounts/fireworks/models/llama4-maverick-instruct-basic"

local_key= "ollama"
online_key= os.getenv("OPENAI_API_KEY")


#编码图片为base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

#解析返回的工具，并执行
def fn(input_list):
    return [fn_tools[x['name']](x['arguments']) for x in input_list]

#主函数
def ai(msg,
       online=False,
       sys_prompt="",
       img=None,
       api_addr=local_api_addr,
       api_key=local_key,
       model=local_model,
       img_model=local_img_model,
       history_file=None,
       tools=[],
       format_opt=None,
    ):

    #如果在线模式
    if online:
        api_addr = online_api_addr
        api_key = online_key
        model = online_model
        img_model = online_img_model

    #构造用户消息
    if img:
        model = img_model
        img_b64 = encode_image(img)
        content = {
            "role": "user",
            "content": [
                {"type": "text", "text": msg},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
        }
    else:
        content = {"role": "user", "content": msg}

    #合并用户和系统消息
    messages = [{"role": "system", "content": sys_prompt}, content]

    #读取历史记录，并附加到消息，消息是列表格式追加
    history=[]
    if history_file:
        path = Path(history_file)
        if path.exists():
            history = json.loads(path.read_text())
            messages = history + messages

    #切换工具或格式化输出模式
    addon = {"response_format": format_opt} if format_opt else {"tools": tools} if tools else {}

    #执行API请求
    response = requests.post(
        f"{api_addr}/chat/completions",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            **addon
        }
    )

    #获取正文回复
    result_data = response.json()
    if 'choices' not in result_data:
        raise ValueError(f"API response error: {result_data}")

    choice = result_data['choices'][0]['message']
    result = choice.get('content')

    #调用工具
    if tools:
        tool_calls = choice.get('tool_calls', [])
        result = fn([f['function'] for f in tool_calls])
    elif isinstance(result, str) and result.startswith("<think>"):
        result = result.split("</think>", 1)[-1]

    if history_file:
        # 确保result是字符串格式存储到历史记录中
        result=str(result).strip()
        
        history.extend([{"role": "user", "content": msg}, {"role": "assistant", "content": result}])
        Path(history_file).write_text(json.dumps(history, ensure_ascii=False, indent=2))

    return result

#工具函数构造
def get_weather(arguments):
    args = json.loads(arguments)
    return f"{args['location']} is sunny"

#工具列表
fn_tools = {
    "get_weather": get_weather
}

if __name__ == "__main__":
    tool_schema = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }]

    # examples = [
    #     ("示例 1:", {"msg": "北京的天气如何？"}),
    #     ("示例 2:", {"msg": "请告诉我北京和上海的天气。"}),
    #     ("示例 3:", {"msg": "广州的天气呢？", "sys_prompt": "你是一个天气查询助手。"}),
    #     ("示例 4:", {"msg": "深圳的天气呢？", "history_file": "history.json"}),
    #     ("示例 5:", {"msg": "杭州的天气如何？", "online": False}),
    #     ("示例 6:", {"msg": "杭州的天气如何？", "online": False, "model": "qwen3"}),
    #     ("示例 7:", {"msg": "你看到了什么？", "online": False, "img": "D:/Desktop/test.png"})
    # ]

    # for title, params in examples:
    #     print(f"\n{title}")
    #     print(ai(**params, tools=tool_schema if 'img' not in params else None))

    # while True:
    #     ask=input("A:")
    #     if ask=="exit":
    #         exit()
    #     data=ai(ask,sys_prompt="你是一个动漫宅女，是人类哦，不要表现得跟AI一样",online=False,history_file="history.json")
    #     print(f"B:{data}")
    # print(ai("世界上最大的侦察机是？"))
    print(ai("写python代码获取当前目录最大的5个文件",sys_prompt="你是python代码生成器，除了用markdown格式输出python源码外，不能说其他话。").strip())
