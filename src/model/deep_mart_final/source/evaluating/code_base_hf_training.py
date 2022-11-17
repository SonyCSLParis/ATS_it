#in aggiunta
def compute_metrics(model,
                    tokenizer,
                    test_dataset,
                    data_collator):
    all_preds = []
    all_labels = []

    #qui ti crei un dataset sul quale compiere la valutazione
    sampled_dataset = test_dataset.shuffle().select(range(200))
    tf_generate_dataset = sampled_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=4,
    )

    #si itera sulla batch
    for batch in tf_generate_dataset:

        #si fa una predizione, generando a partire dagli input e dall'attention mask
        predictions = model.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        labels_ids = predictions.label_ids
        pred_ids = predictions.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in pred_str]
        decoded_labels = [[label.strip()] for label in label_str]
        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_labels)


    metric = evaluate.load("sacrebleu")
    result = metric.compute(predictions=all_preds, references=all_labels)
    return {"bleu": result["score"]}




'''
train_ds, eval_ds = HuggingFaceTrainer.load_dataset('/Users/francesca/Desktop/deep-martin-final/data/final_hf.csv')
data_collator = DataCollatorForSeq2Seq(tokenizer, model_deep=model_deep)
print(compute_metrics(model_deep=model_deep, tokenizer=tokenizer, test_dataset= eval_ds, data_collator=data_collator))'''