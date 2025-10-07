import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY", "dummy-key"),
    base_url="https://api.deepinfra.com/v1/openai",
)


def get_completion(prompt, model="openai/gpt-oss-20b"):
    print("Generating using model:", model)
    response = openai.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=0.7
    )
    return response.choices[0].message.content


def generate_image_base64(prompt, model="stabilityai/sd3.5-medium"):
    response = openai.images.generate(
        prompt=prompt,
        model=model,
        n=1,
        size="512x512"
    )
    b64_data = response.data[0].b64_json
    return f"data:image/png;base64,{b64_data}"
