import os
import openai
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from transformers import EncoderDecoderModel, AutoTokenizer
'''OPENAI_API_KEY = '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'
# Load your API key from an environment variable or secret management service
openai.api_key = OPENAI_API_KEY

response = openai.Completion.create(model="text-davinci-003", prompt="Simplify the following code: I have been to the supermarket and I bought more than needed. I will probably contribute to food waste in the next weeks. ", temperature=0.8, max_tokens=30)


text = 'Questa sera pulisco bene la casa e poi metto delle trappole per topo'
aug = naw.SynonymAug(aug_src='wordnet', lang= 'ita')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")

print(augmented_text)



aug = naw.RandomWordAug(action="swap")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)



aug = naw.RandomWordAug()
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)'''

