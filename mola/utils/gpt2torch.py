from transformers import TrainingArguments, Trainer

config = AutoConfig.from_pretrained("gpt2")
config.n_positions = 128          # 可按需缩短
model = PureTorchGPT2ForSequenceClassification(config, num_labels=2).to("cuda")

training_args = TrainingArguments(
    output_dir="./pure_torch_gpt2",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_steps=100,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,   # 仍是 Dataset 对象
    compute_metrics=lambda pred: {"acc": (pred.predictions.argmax(-1) == pred.label_ids).mean()},
)

trainer.train()