import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
import gensim.downloader as api
from nltk import word_tokenize
import sys

seed = 27
torch.manual_seed(seed)
np.random.seed(seed)

train_data = sys.argv[1]
test_data = sys.argv[2]
output = sys.argv[3]

# load training data
#train_data = os.path.join(os.path.dirname(__file__), "hw1_data/hw1_train.csv")
model_path = os.path.join(os.path.dirname(__file__), "trained_model")

train_df = pd.read_csv(train_data)

# resample uncommon data
counts = train_df["CORE RELATIONS"].fillna("none").value_counts()
least_common = counts[counts < 100]
most_common = counts[counts >= 100]
samples = counts[counts >= 100].sum()
minority = train_df[train_df["CORE RELATIONS"].isin(least_common.index)]
majority = train_df[train_df["CORE RELATIONS"].isin(most_common.index)]
minority_resampled = resample(minority, replace=True, n_samples=samples, random_state=27)
balanced = pd.concat([minority_resampled, majority])

x = balanced["UTTERANCES"].values
# fill all nan with none and split string into multiple labels
balanced["CORE RELATIONS"] = balanced["CORE RELATIONS"].fillna("none").str.split()
# convert labels into tuples to use label encoder
y = balanced["CORE RELATIONS"].apply(lambda relations: tuple(relations))
# convert relations into labels and one-hot encode
mlb = MultiLabelBinarizer() 
y = mlb.fit_transform(y)

# split training data into train and val set
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=27)

# convert utterances into matrix, penalize rare words
vectorizer = TfidfVectorizer()
vectorizer.fit(x_train)

# load embeddings
wv = api.load("glove-twitter-200")
text_embeddings = []
# embed text
for text in x_train:
    token_embeddings = []
    for token in word_tokenize(text.lower()):
        if token in wv:
            token_embeddings.append(wv[token])
    # take average of all emebddings (mean of all token embeddings in a sentence)
    text_embedding = np.array(token_embeddings).mean(axis=0)
    text_embeddings.append(text_embedding)
x_train_embed = np.array(text_embeddings)

text_embeddings = []
# embed text
for text in x_val:
    token_embeddings = []
    for token in word_tokenize(text.lower()):
        if token in wv:
            token_embeddings.append(wv[token])
    # take average of all emebddings
    text_embedding = np.array(token_embeddings).mean(axis=0)
    text_embeddings.append(text_embedding)
x_val_embed = np.array(text_embeddings)

# Convert to tensors
x_train = torch.FloatTensor(vectorizer.transform(x_train).toarray())
x_val = torch.FloatTensor(vectorizer.transform(x_val).toarray())
y_train = torch.FloatTensor(y_train)
y_val = torch.FloatTensor(y_val)

# concat embeddings to bag of words vector
x_train_embed = torch.tensor(x_train_embed)
x_val_embed = torch.tensor(x_val_embed)
x_train = torch.cat([x_train, x_train_embed], dim=1)
x_val = torch.cat([x_val, x_val_embed], dim=1)
train_data = TensorDataset(x_train, y_train)
val_data = TensorDataset(x_val, y_val)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)

# Classifier model
class MultiClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.ll1 = nn.Linear(input_dim, 512)
        self.ll2 = nn.Linear(512, 256)
        self.ll3 = nn.Linear(256, 128)
        self.ll4 = nn.Linear(128, 64)
        # 19 possible labels
        self.ll5 = nn.Linear(64, 19)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.ll1(x))
        x = self.gelu(self.ll2(x))
        x = self.gelu(self.ll3(x))
        x = self.gelu(self.ll4(x))
        x = self.ll5(x)
        x = x.squeeze()
        return x
    
dim = x_train.size(1)
model = MultiClassifier(dim)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.006)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)

prev_acc = None
prev_loss = None
num_epoch = 150
for epoch in range(num_epoch):
    
    epoch_loss = 0
    model.train()
    for x_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = loss_fn(preds, y_batch.squeeze())
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
    scheduler.step(epoch_loss)

    model.eval()
    with torch.no_grad():
        logits = model(x_val)
        predicted = nn.functional.sigmoid(logits)
        predicted = (logits > 0.5).int()
        acc = torch.all((predicted == y_val), dim=1).int().sum() / y_val.shape[0]
        print(f"Epoch: {epoch + 1}/{num_epoch}, Loss: {loss}, Acc: {acc}")
    
        # save model with lowest loss and highest accuracy
        if prev_acc is None or (prev_acc <= acc and loss < prev_loss):
            print("HIGHEST ACC, LOWEST LOSS : SAVING MODEL")
            torch.save(model.state_dict(), model_path)
            prev_acc = acc
            prev_loss = loss

# load test data
# test_data = os.path.join(os.path.dirname(__file__), "hw1_data/hw1_test.csv")
# results = os.path.join(os.path.dirname(__file__), "submission.csv")
test_df = pd.read_csv(test_data)
test_data = test_df["UTTERANCES"].values

text_embeddings = []
# embed text
for text in test_data:
    token_embeddings = []
    for token in word_tokenize(text.lower()):
        if token in wv:
            token_embeddings.append(wv[token])
    if token_embeddings:
        # average of all token embeddings in a sentence
        text_embedding = np.array(token_embeddings).mean(axis=0)
    # whole utterance OOV, create random embedding
    else:
        text_embedding = np.random.rand(wv.vector_size)
    text_embeddings.append(text_embedding)
x_test_embed = np.array(text_embeddings)
x_test = torch.FloatTensor(vectorizer.transform(test_data).toarray())
x_test_embed = torch.tensor(x_test_embed)
x_test = torch.cat([x_test, x_test_embed], dim=1)
x_test = x_test.float()

# load model and run on final dataset for submission
saved_model = MultiClassifier(dim)
saved_model.load_state_dict(torch.load(model_path))
saved_model.eval()
with torch.no_grad():
    logits = saved_model(x_test)
    predicted = nn.functional.sigmoid(logits)
    predicted = (predicted > 0.5).int()
    relations = mlb.inverse_transform(predicted)

# overwrite blank outputs to none
new_re = []
for tup in relations:
    new_el = []
    if len(tup) == 0:
        new_re.append(["none"])
    else:
        for el in tup:
            new_el.append(str(el))
        new_re.append(new_el)

# format output
df = pd.DataFrame()
df["Core Relations"] = new_re
df["Core Relations"] = df["Core Relations"].map(lambda row: " ".join(sorted(row)))
df.index.name = "ID"
df.to_csv(output)