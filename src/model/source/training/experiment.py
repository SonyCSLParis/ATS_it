from transformers import (
    AutoTokenizer,
    EncoderDecoderModel, AutoModelForSeq2SeqLM)

model = AutoModelForSeq2SeqLM.from_pretrained("gsarti/it5-base")
tokenizer = AutoTokenizer.from_pretrained("gsarti/it5-base")
model.save_pretrained('/Users/francesca/Desktop/pretr')
tokenizer.save_pretrained('/Users/francesca/Desktop/pretr')


