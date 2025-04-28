import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from main import load_model, encode_sentences, get_embedding_dimension, MultiTaskModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

task_a_labels = ["Greetings", "Sports", "Weather"]
task_b_labels = ["Positive", "Negative"]

train_sentences = [
    "Soccer is pretty cool!",
    "Hello Fetch!",
    "The weather is so gloomy.",
    "I love sunny days!",
    "Basketball is exciting.",
    "Technology is advancing fast.",
    "It's raining again."
]

task_a_ground_truth = [1, 0, 2, 0, 1, 1, 2]
task_b_ground_truth = [0, 0, 1, 0, 0, 0, 1]

num_epochs = 3
batch_size = 4
learning_rate = 1e-3

if __name__ == "__main__":
    set_seed(1467)

    # logic similar to what's in main.py
    sentence_model = load_model()
    embedding_dim = get_embedding_dimension(sentence_model)
    multitask_model = MultiTaskModel(
        embedding_dim=embedding_dim,
        num_classes_task_a=len(task_a_labels),
        num_classes_task_b=len(task_b_labels)
    )

    # freeze the sentence encoder
    for param in sentence_model.parameters():
        param.requires_grad = False

    # set up optimizer and loss functions
    optimizer = optim.Adam(multitask_model.parameters(), lr=learning_rate)
    loss_fn_task_a = nn.CrossEntropyLoss()
    loss_fn_task_b = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(num_epochs):
        multitask_model.train()
        running_loss = 0.0
        correct_task_a = 0
        correct_task_b = 0
        total = 0

        # batch training
        for batch_idx in range(0, len(train_sentences), batch_size):
            batch_sentences = train_sentences[batch_idx:batch_idx + batch_size]
            batch_task_a_labels = torch.tensor(task_a_ground_truth[batch_idx:batch_idx + batch_size])
            batch_task_b_labels = torch.tensor(task_b_ground_truth[batch_idx:batch_idx + batch_size])

            embeddings = encode_sentences(sentence_model, batch_sentences)
            task_a_logits, task_b_logits = multitask_model(embeddings)

            # compute loss
            loss_task_a = loss_fn_task_a(task_a_logits, batch_task_a_labels)
            loss_task_b = loss_fn_task_b(task_b_logits, batch_task_b_labels)
            loss = loss_task_a + loss_task_b

            # backwards/optimization pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # predictions
            preds_task_a = torch.argmax(task_a_logits, dim=1)
            preds_task_b = torch.argmax(task_b_logits, dim=1)

            # update metrics
            correct_task_a += (preds_task_a == batch_task_a_labels).sum().item()
            correct_task_b += (preds_task_b == batch_task_b_labels).sum().item()
            total += batch_task_a_labels.size(0)
            running_loss += loss.item()

        # epoch metrics
        epoch_loss = running_loss / (len(train_sentences) // batch_size)
        task_a_accuracy = correct_task_a / total
        task_b_accuracy = correct_task_b / total
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} | Task A Acc: {task_a_accuracy:.4f} | Task B Acc: {task_b_accuracy:.4f}")
