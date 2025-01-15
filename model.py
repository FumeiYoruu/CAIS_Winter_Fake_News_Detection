import pandas as pd
import numpy as np
import os
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from torchmetrics import Precision, Recall
import random

def plot_accuracies(val_accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_precision(val_precisions):
    plt.figure(figsize=(10, 6))
    plt.plot(val_precisions, label='Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Validation Precision')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_recall(val_recalls):
    plt.figure(figsize=(10, 6))
    plt.plot(val_recalls, label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Validation Recall')
    plt.legend()
    plt.grid(True)
    plt.show()



folder_path = '/content/drive/MyDrive/CAIS/F24_Curriculum/Winter Project/News_dataset'
fake = pd.read_csv(os.path.join(folder_path, 'Fake.csv'))
true = pd.read_csv(os.path.join(folder_path, 'True.csv'))
word_to_remove = 'Reuters'
true['text'] = true['text'].str.replace(rf'\({word_to_remove}\)', '', regex=True)
true['text'] = true['text'].str.replace(r'^\s*-\s*', '', regex=True)
true['text'] = true['text'].str.replace(r'^[^-]*-\s*', '', regex=True)

fake['label'] = 0
true['label'] = 1
df = pd.concat([fake, true])
df = df.sample(frac=1).reset_index(drop=True)

df['text'] = df['title'] + df['text']
df = df[['text', 'label']]

texts = df.text.values
labels = df.label.values


print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

max_len = 100

input_ids = []
attention_masks = []

for t in texts:
    encoded_dict = tokenizer.encode_plus(
                        t,
                        add_special_tokens = True,
                        max_length = max_len,
                        truncation = True,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

print('Original: ', texts[0])
print('Token IDs:', input_ids[0])

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

batch_size = 16

train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
            
        )

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2, 
    output_attentions = False, 
    output_hidden_states = False,
)

model.cuda()

optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )

epochs = 2

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


device = torch.device("cuda")

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []


for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')


    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        # Progress update every 100 batches.
        if step % 100 == 0 and not step == 0:

            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))


        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        result = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels,
                       return_dict=True)

        loss = result.loss
        logits = result.logits

        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataloader)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")


    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    precision_metric = Precision(task = "binary", num_classes=2, average='binary').to(device)
    recall_metric = Recall(task = "binary", num_classes=2, average='binary').to(device)
    for batch in validation_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
          
            result = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)
            

        loss = result.loss
        logits = result.logits

    
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        preds = torch.tensor(np.argmax(logits, axis=1).flatten())
        precision_metric.update(preds, b_labels.to('cpu'))
        recall_metric.update(preds, b_labels.to('cpu'))

        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Valid. Precision': precision,
            'Valid. Recall': recall
        }
    )

print("")
print("Training complete!")


