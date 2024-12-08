from transformers import BertForSequenceClassification, Trainer, TrainingArguments

def load_model(num_labels, model_name="bert-base-uncased"):
    """
    Loads the pre-trained model.

    Args:
        num_labels (int): Number of output labels.
        model_name (str): Name of the pre-trained model.

    Returns:
        model: Loaded pre-trained model.
    """
    return BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def train_model(train_dataset, val_dataset, model, tokenizer, output_dir="./model"):
    """
    Trains the model using Hugging Face's Trainer.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        model: Pre-trained model instance.
        tokenizer: Tokenizer instance.
        output_dir (str): Directory to save the model.

    Returns:
        trainer: Trained model instance.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer
