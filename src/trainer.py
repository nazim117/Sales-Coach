from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from sklearn.metrics import classification_report
from transformers import Trainer, TrainingArguments
import torch
from torch.nn import CrossEntropyLoss

class WeightedTrainer(Trainer):
    """
    Custom Trainer to handle class weights during training.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss with optional class weights.

        Args:
            model: The model instance.
            inputs: The input data.
            return_outputs (bool): Whether to return model outputs.
            **kwargs: Additional arguments passed by the Trainer.

        Returns:
            loss: Computed loss value.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        print(f"Logits: {logits}")
        print(f"Predictions: {torch.argmax(logits, dim=1)}")
    
        # Use class weights if available
        if self.class_weights is not None:
            loss_fn = CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fn = CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        print(f"Loss: {loss.item()}")
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_preds):
    """
    Computes evaluation metrics for validation/testing.
    """
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    print(f"Metrics -> Accuracy: {acc}, F1: {f1}, Precision: {precision}, Recall: {recall}")
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def train_model(train_dataset, val_dataset, model, output_dir="./model", class_weights=None):
    """
    Trains the model using a custom Trainer with class weights.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",  # Evaluate after each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=50,
        num_train_epochs=1,  # Increase epochs for better fine-tuning
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights  # Pass class weights here
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Final Evaluation Results: {eval_results}")
    return trainer
