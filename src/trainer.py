from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
import numpy as np
import torch
import torch_directml
from transformers import Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss

# Initialize DirectML device
device = torch_directml.device(0)  # Replace `0` with your intended device ID if needed

print("Using DirectML Device:", device)

class WeightedTrainer(Trainer):
    def __init__(self, model=None, *args, class_weights=None, thresholds=None, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.class_weights = class_weights
        self.thresholds = thresholds or {"neutral": 0.35}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(device)
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
        logits = outputs.logits

        # Weighted loss
        if self.class_weights is not None:
            loss_fn = CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fn = CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def predict_with_thresholds(self, logits):
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        for i, prob in enumerate(probs):
            if prob[1] > self.thresholds["neutral"]:
                preds[i] = 1  # Neutral
        return preds

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    probs = torch.softmax(torch.tensor(logits), dim=1)
    preds = torch.argmax(probs, dim=1)

    # Custom thresholds
    neutral_threshold = 0.35
    positive_threshold = 0.65

    for i, prob in enumerate(probs):
        if prob[1] > neutral_threshold and prob[1] <= positive_threshold:
            preds[i] = 1  # Neutral
        elif prob[2] > positive_threshold:
            preds[i] = 2  # Positive
        else:
            preds[i] = 0  # Negative

    report = classification_report(labels, preds.numpy(), target_names=["negative", "neutral", "positive"])
    print(report)

    return {
        "accuracy": accuracy_score(labels, preds.numpy()),
        "f1": f1_score(labels, preds.numpy(), average="weighted"),
        "precision": precision_score(labels, preds.numpy(), average="weighted"),
        "recall": recall_score(labels, preds.numpy(), average="weighted"),
    }

def train_model(train_dataset, val_dataset, model, output_dir="./model", class_weights=None):
    """
    Trains the model using a custom Trainer with class weights.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=50,
        num_train_epochs=3,
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
        class_weights=class_weights
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Final Evaluation Results: {eval_results}")

    return trainer

def optimize_thresholds(probs, labels):
    best_threshold = 0.35
    best_f1 = 0

    for threshold in np.arange(0.2, 0.5, 0.01):  # Test thresholds
        preds = []
        for prob in probs:
            if prob[1] > threshold:  # Neutral threshold
                preds.append(1)  # Neutral
            else:
                preds.append(np.argmax(prob))  # Positive/Negative
        f1 = f1_score(labels, preds, average="weighted")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Best Threshold: {best_threshold}, Best F1: {best_f1}")
    return best_threshold
