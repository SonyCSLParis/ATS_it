from transformers import (
    AutoTokenizer,
    EncoderDecoderModel, AutoModelForSeq2SeqLM)

'''model = AutoModelForSeq2SeqLM.from_pretrained("gsarti/it5-base")
tokenizer = AutoTokenizer.from_pretrained("gsarti/it5-base")
model.save_pretrained('/Users/francesca/Desktop/pretr')
tokenizer.save_pretrained('/Users/francesca/Desktop/pretr')'''



from transformers import AutoModel, AutoTokenizer

model_name = "dbmdz/bert-base-italian-xxl-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)

tokenizer.save_pretrained('/Users/francesca/Desktop/Github/Final_final/src/model/source/bert2bert_cased')
model.save_pretrained('/Users/francesca/Desktop/Github/Final_final/src/model/source/bert2bert_cased')