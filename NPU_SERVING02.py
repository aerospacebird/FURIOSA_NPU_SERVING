
"""
furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct/furiosa-ai/EXAONE-4.0-32B-FP8
furiosa-llm serve furiosa-ai/gpt-oss-20b
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic  #  성공하였다.
"""
from openai import OpenAI

client = OpenAI(

    base_url = "http://127.0.0.1:8000/v1",
    api_key="EMPTY"

)

models = client.models.list()
model_id = models.data[0].id

print("모델 :", model_id)

#추론
response = client.chat.completions.create(

    model = model_id,
    messages= [

        {
            "role": "user",
            "content": "GPU와 NPU의 차이점은"

        }               
               
               ],
        max_tokens=1000
)
print(response.choices[0].message.content)


