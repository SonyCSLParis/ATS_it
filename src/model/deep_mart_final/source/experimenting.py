import os
import openai
OPENAI_API_KEY = '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'
# Load your API key from an environment variable or secret management service
openai.api_key = OPENAI_API_KEY

response = openai.Completion.create(model="text-davinci-003", prompt="Simplify the following code: I have been to the supermarket and I bought more than needed. I will probably contribute to food waste in the next weeks. ", temperature=0.8, max_tokens=30)

print(response)