

import json
import os
from openai import OpenAI

# ==========================================
# 1. Furiosa NPU LLM Client
# ==========================================

base_url = os.getenv(
    "OPENAI_BASE_URL",
    "http://localhost:8000/v1"
)

api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    timeout=120.0
)

# ==========================================
# 2. External Functions
# ==========================================

def get_weather(location: str, unit: str):
    return {
        "location": location,
        "unit": unit,
        "status": "demo",
        "message": f"Getting weather for {location} in {unit}"
    }

def get_time(timezone: str):
    return {
        "timezone": timezone,
        "status": "demo",
        "message": f"Getting current time in {timezone}"
    }

tool_functions = {
    "get_weather": get_weather,
    "get_time": get_time
}

# ==========================================
# 3. Tool Schemas
# ==========================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location", "unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time in a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string"
                    }
                },
                "required": ["timezone"]
            }
        }
    }
]

# ==========================================
# 4. Server Connection Test
# ==========================================

try:
    models = client.models.list()

    print("SERVER CONNECTION: PASS")

    for model in models.data:
        print("Loaded model:", model.id)

except Exception as e:
    print("SERVER CONNECTION: FAIL")
    print(type(e).__name__, str(e))
    raise

# ==========================================
# 5. Function Calling Test
# ==========================================

try:
    response = client.chat.completions.create(
        model=models.data[0].id,
        messages=[
            {
                "role": "user",
                "content": "오늘 판교의 날씨는 어때?"
            }
        ],
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {
                "name": "get_weather"
            }
        },
        temperature=0.0,
        max_completion_tokens=256
    )

    message = response.choices[0].message

    print("\nFUNCTION CALLING RESULT")

    if not message.tool_calls:
        raise RuntimeError(
            "Model did not return a tool call."
        )

    for call in message.tool_calls:

        function_name = call.function.name
        arguments = json.loads(
            call.function.arguments
        )

        print("\nFunction:", function_name)
        print("Arguments:", arguments)

        function = tool_functions[function_name]

        result = function(**arguments)

        print("Execution Result:")
        print(json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        ))

    print("\nFUNCTION CALLING: PASS")

except Exception as e:
    print("\nFUNCTION CALLING: FAIL")
    print(type(e).__name__, str(e))
    raise

# """
# furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:5 --host 0.0.0.0 --post 8000

# """

# import json
# import os
# from openai import OpenAI

# base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
# api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
# client = OpenAI(base_url=base_url, api_key=api_key)

# def get_weather(location:str, unit:str):
#     return f"Getting the weather for {location} in {unit}"

# def get_time(timezone:str):
#     return f"Getting the current time in {timezone}..."

# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_weather",
#             "description": "Get the current weather in a given location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
#                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#                 },
#                 "required": ["location", "unit"]
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "get_time",
#             "description": "Get the current time in a given timezone",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "timezone": {"type": "string", "description": "Timezone, e.g., 'America/Los_Angeles'"}
#                 },
#                 "required": ["timezone"]
#             }
#         }
#     }
# ]

# response = client.chat.completions.create(
#     model = client.models.list().data[0].id,
#     messages = [{"role":"user", "content":"오늘 판교의 날씨는 어때?"}],
#     tools = tools,
#     tool_choice = {"type":"function", "function": {"name": "get_weather"}},
#     temperature=0.0,
# )

# tool_call = response.choices[0].message.tool_calls[0].function
# print(f"function called: {tool_call.name}")
# print(f"Arguments : {tool_call.arguments}")
# assert tool_call.name == "get_weather", f"Expected 'get_weather' but got '{tool_call.name}'"
# print(f"Result: {get_weather(**json.loads(tool_call.arguments))}")
