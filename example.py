from openai import OpenAI


client = OpenAI(base_url="http://localhost:5001/proxy/v1")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Hello, my card number is 4111-1111-1111-1111. Call me at (123) 456-7890",
        },
    ],
)

print(response.choices[0].message.content)
