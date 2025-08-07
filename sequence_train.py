
# Libraries import
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

import random

import pandas as pd
import numpy as np

import os
import sys

import math

import time

# GPU
if torch.cuda.is_available():
    print("GPU is running.")
    dev = "cuda:1"
else:
    print("CPU is running.")
    dev = "cpu"

device = torch.device(dev)

# Dataset
dataset_all_rows_total = pd.read_csv('data/data_main.tsv',sep='\t', index_col= 0)

# dataset_all_rows = dataset_all_rows.sample(frac = 0.01, random_state= 42)

count = 0
tfs = []

strength_sequence = []


dataset_all_rows = dataset_all_rows_total.sample(frac=0.2, random_state= 42)

for index, sequence in dataset_all_rows.iterrows():

    sequence = np.array(sequence.dropna().values)

    count+=1

    tfs.append(sequence[:-3])

    strength_sequence.append(sequence[-2])

# print(count)

# print(len(tfs))
# print(tfs[1])

# Separate the motifs and position
all_motifs = []
all_positions = []

for seq_index, motifs in enumerate(tfs):

    motifs_sequence = []
    position_motifs = []

    for each_item in motifs:

        each_motif = each_item.split(':')[0]

        each_motif_position = int(each_item.split(':')[1])

        motifs_sequence.append(each_motif)

        position_motifs.append(each_motif_position)

    all_motifs.append(motifs_sequence)

    all_positions.append(position_motifs)


def removeEmpty(motifs_list):

    filtered_motifs = []

    for item in motifs_list:


        if len(item) > 0:

            filtered_motifs.append(item)

    return filtered_motifs


filtered_motifs = removeEmpty(all_motifs)

position_sequences = removeEmpty(all_positions)

encoding = {
    'A': [0, 0],
    'C': [0, 1],
    'G': [1, 0],
    'T': [1, 1]
}

def one_hot_encoding(motifs_sequences):

    all_motifs_encoded = []

    for seq in motifs_sequences:

        encoded = []

        for each_motif in seq:

            encoded_seq = []

            for base in each_motif:

                if base in encoding:

                    encoded_seq.extend(encoding[base])

                # else:

                    # print("error")


            encoded.append(''.join(map(str, encoded_seq)))

        all_motifs_encoded.append(encoded)

    return all_motifs_encoded


encoded_motifs_list = one_hot_encoding(filtered_motifs)
# print(len(encoded_motifs_list))
# print(encoded_motifs_list[1])

max_len = max(len(seq[0]) for seq in encoded_motifs_list)

# Pad sequences to the maximum length
padded_motifs = [
    seq[0] + '0' * (max_len - len(seq[0])) if len(seq[0]) < max_len else seq[0]
    for seq in encoded_motifs_list
]

# Convert padded sequences to tensor
data_tensor = torch.tensor(
    [[int(bit) for bit in seq] for seq in padded_motifs],
    dtype=torch.float32
)


def filterSequenceByPosition(encoded_motifs_list, all_positions, original_motif_sequences):
    sorted_vector_representations = []
    sorted_positions = []
    sorted_original_words = []

    for index in range(len(encoded_motifs_list)):

        temp_zip = list(zip(encoded_motifs_list[index],all_positions[index], original_motif_sequences[index]))

        sorted_zip = sorted(temp_zip,key=lambda x: x[1])

        sorted_vec,sorted_position,sorted_word = zip(*sorted_zip)

        sorted_vector_representations.append(sorted_vec)
        sorted_positions.append(sorted_position)
        sorted_original_words.append(sorted_word)

    return sorted_vector_representations,sorted_positions,sorted_original_words

sorted_motif_encoding_representations , sorted_motif_positions, sorted_original_motifs = filterSequenceByPosition(encoded_motifs_list, position_sequences, filtered_motifs)

max_len = max(len(seq[0]) for seq in sorted_motif_encoding_representations)

# Padding sequences to the maximum length
padded_motifs = [
    seq[0] + '0' * (max_len - len(seq[0])) if len(seq[0]) < max_len else seq[0]
    for seq in sorted_motif_encoding_representations
]

# sequences to tensor
motif_tensor = torch.tensor(
    [[int(bit) for bit in seq] for seq in padded_motifs],
    dtype=torch.float32
)

all_indices = [val for val in range(len(sorted_motif_encoding_representations))]
train_indices = random.sample(all_indices, int(0.8 * len(all_indices)))
test_indices = list(set(all_indices).difference(set(train_indices)))

training_vec_sequences = [torch.from_numpy(np.array(motif_tensor[index])).to(dtype=torch.float32, device=device) for index in train_indices]
training_sequences_positions = [torch.from_numpy(np.array(sorted_motif_positions[index])).to(dtype=torch.long, device=device) for index in train_indices]
training_motifs_sequences = [sorted_original_motifs[index] for index in train_indices]
training_sequence_strength= [torch.from_numpy(np.array(strength_sequence[index])).to(dtype=torch.float32, device=device) for index in train_indices]

testing_vec_sequences = [torch.from_numpy(np.array(motif_tensor[index])).to(dtype=torch.float32, device=device) for index in test_indices]
testing_sequences_positions = [torch.from_numpy(np.array(sorted_motif_positions[index])).to(dtype=torch.long, device=device) for index in test_indices]
testing_motifs_sequences = [sorted_original_motifs[index] for index in test_indices]
testing_sequence_strength = [torch.from_numpy(np.array(strength_sequence[index])).to(dtype=torch.float32, device=device) for index in test_indices]

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, negative_dist = 250, positive_dist = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.negative_dist = negative_dist
        self.d_model = d_model

        self.seq_length = negative_dist + positive_dist

        position = torch.arange(0,self.seq_length,device = device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(device=device)

        pe = torch.zeros(self.seq_length, 1, d_model).to(device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term).to(device=device)
        pe[:, 0, 1::2] = torch.cos(position * div_term).to(device=device)
        self.register_buffer('pe', pe)

    def forward(self, x, positions):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        batch_size, seq_len , embedding_dim = x.shape

        x = x.transpose(1,0)

        indices = positions.reshape(-1) + self.negative_dist

        x = x + self.pe[indices]

        x = self.dropout(x)

        x.transpose(1,0)

        return x

class GeneTransformer(nn.Module):
    def __init__(self, input_dim = 50, model_dim = 64, n_output_heads = 1, nhead = 8, enc_layers_count = 3, motif_count = 1):
        super().__init__()

        self.input_dim = input_dim
        self.model_dim = model_dim

        self.motif_count = motif_count

        self.n_output_heads = n_output_heads

        self.nhead = nhead
        self.enc_layers_count = enc_layers_count

        # Linear Layers for the input.
        self.input_embedding_1 = torch.nn.Linear(self.input_dim , int(self.model_dim/2))
        self.input_embedding_2 = torch.nn.Linear(int(self.model_dim/2), self.model_dim)

        self.positional_embedding = PositionalEncoding(d_model=self.model_dim)

        # Transformer model definition.
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.nhead)
        self.encoders = torch.nn.TransformerEncoder(self.encoder_layer , num_layers = self.enc_layers_count)

        # Final output layer for the model.

        self.genetics_classifier_1 = torch.nn.Linear(self.model_dim * self.motif_count, int((self.model_dim)/2))

        self.genetics_classifier_2 = torch.nn.Linear(int((self.model_dim)/2), self.n_output_heads)


        # Activation Functions
        self.embedding_activation = torch.nn.GELU()
        # self.classifiar_activation = torch.nn.Sigmoid()

        # Dropout Functions
        self.dropout_5 = torch.nn.Dropout(p = 0.05)
        self.dropout_10 = torch.nn.Dropout(p = 0.10)


    def _initialize_classifiers(self):
        """Helper method to initialize or update the classifier layers."""
        self.genetics_classifier_1 = torch.nn.Linear(self.model_dim * self.motif_count, int(self.model_dim / 2)).to(device = dev)
        self.genetics_classifier_2 = torch.nn.Linear(int(self.model_dim / 2), self.n_output_heads).to(device = dev)

    def get_motif_count(self, motif_count):
        """
        Dynamically adjust the motif count and update related layers.
        """
        self.motif_count = motif_count
        self._initialize_classifiers()

    def forward(self,inputs, positions):

        batch_size, seq_length, input_dim  = inputs.shape



        embedded_input = self.input_embedding_1(inputs.reshape(-1,input_dim))

        embedded_input = self.embedding_activation(embedded_input)
        embedded_input = self.dropout_5(embedded_input)



        embedded_input = self.input_embedding_2(embedded_input)
        embedded_input = self.embedding_activation(embedded_input)

        embedded_input = self.dropout_5(embedded_input)

        embedded_input = embedded_input.reshape(batch_size,seq_length,self.model_dim)

        positional_embedded_input = self.positional_embedding(embedded_input, positions)

        # motifs_in_sequnece, batch_size, dimension = positional_embedded_input.shape

        encoded_x = self.encoders(positional_embedded_input)

        encoded_x = torch.mean(encoded_x,dim = 1)

        encoded_x = encoded_x.reshape(1 , int(self.motif_count * self.model_dim))

        output_x = self.genetics_classifier_1(encoded_x)

        output_x = self.genetics_classifier_2(output_x)

        output_x = output_x.reshape(-1)

        return output_x


gene_model = GeneTransformer(input_dim=50, model_dim = 256, n_output_heads= 1, nhead= 8 , enc_layers_count= 2, motif_count = 1)
gene_model = gene_model.to(device=device)

optimizer = torch.optim.AdamW(list(gene_model.parameters()), lr= 0.01, weight_decay = 0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3 ,gamma = 0.6, last_epoch= -1, verbose=False)

criterion = nn.MSELoss()

# Get data batches
def getBatches(vec_data, original_data, positions, outputs,batch_size = 1):
    counter = 0

    vec_data_batches = []
    original_data_batches = []
    position_data_batches = []
    output_data_batches = []

    while counter < len(vec_data):
        batched_vec_data = vec_data[counter: counter+batch_size][0]
        batched_original_data = original_data[counter:counter+batch_size][0]
        batched_position_data = positions[counter:counter+batch_size][0]
        batched_output_data = outputs[counter: counter+batch_size][0]

        counter = counter + batch_size

        vec_data_batches.append(batched_vec_data)
        original_data_batches.append(batched_original_data)
        position_data_batches.append(batched_position_data)
        output_data_batches.append(batched_output_data)

    return (vec_data_batches, original_data_batches, position_data_batches, output_data_batches)


# Training batches

seq_training_batches, training_motifs_batches, trainining_motifs_positions_batches, training_strength_batches = getBatches(training_vec_sequences, training_motifs_sequences, training_sequences_positions, training_sequence_strength)

# Testing batches

seq_testing_batches, testing_motifs_batches, testing_motifs_positions_batches, testing_strength_batches = getBatches(testing_vec_sequences, testing_motifs_sequences, testing_sequences_positions, testing_sequence_strength)

gene_model.train()


def TrainModel(seq_representation_inputs, motifs_inputs, position_inputs, strength_output, epoch_number):
    total_loss = 0
    total_batches = 0

    for input_index , batch_input in enumerate(seq_representation_inputs):

        total_batches +=1

        # print('input size', batch_input.size())

        batch_input = batch_input.reshape(1,-1,50)

        position_input = position_inputs[input_index].reshape(1,-1,1)


        # actual_output = strength_output[input_index].reshape(-1)

        actual_output = strength_output[input_index].reshape(-1)

        batch_size, motif_count, input_dim = position_input.shape

        gene_model.get_motif_count(motif_count)

        output = gene_model(batch_input, position_input).to(device = dev)

        loss = criterion(output,actual_output)

        total_loss+=loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if  (input_index + 1) % 100000 == 0:

        #     print('Epoch Number: {} => Batch number :{} , loss value : {} '.format(epoch_number,input_index+1,loss.item()))
        #     print("=================================================")

    print('Epoch Number: {} => Avg loss value : {}'.format(epoch_number, total_loss / (total_batches)))

    return total_loss / (total_batches)

epoch_number = 1

epoch_end_count = 0
train_epoch_avg_losses_sp = []

start_time = time.time()

while epoch_number <= 8:

    temp_holder = list(zip(seq_training_batches, training_motifs_batches, trainining_motifs_positions_batches, training_strength_batches))
    random.shuffle(temp_holder)

    seq_training_batches, training_motifs_batches, trainining_motifs_positions_batches, training_strength_batches = zip(*temp_holder)

    epoch_loss_sp = TrainModel(seq_training_batches, training_motifs_batches, trainining_motifs_positions_batches, training_strength_batches, epoch_number)

    epoch_number += 1

    if len(train_epoch_avg_losses_sp) == 0 :
        train_epoch_avg_losses_sp.append(epoch_loss_sp)
        continue

    if abs(epoch_loss_sp - train_epoch_avg_losses_sp[-1]) <= 0.1:
        epoch_end_count += 1

    train_epoch_avg_losses_sp.append(epoch_loss_sp)


    scheduler.step()

end_time = time.time()

print("Time Elapsed:", end_time - start_time)
print("Training Epoch Losses:", train_epoch_avg_losses_sp)

gene_model.eval()

def TestGeneModel(seq_representation_inputs, motifs_inputs, position_inputs, strength_output):

    actual_strength_values = []

    predicted_strength_values = []

    total_batches = 0

    with torch.no_grad():


        for input_index , batch_input in enumerate(seq_representation_inputs):

            total_batches +=1

            # print('input size', batch_input.size())

            batch_input = batch_input.reshape(1,-1,50)

            position_input = position_inputs[input_index].reshape(1,-1,1)

            actual_output = strength_output[input_index].reshape(-1)

            batch_size, motif_count, input_dim = position_input.shape

            gene_model.get_motif_count(motif_count)

            output = gene_model(batch_input, position_input).to(device = device)

            actual_strength_values.append(actual_output)

            predicted_strength_values.append(output)

        # actual_strength_values = np.concatenate(actual_strength_values, axis=0)

        # predicted_strength_values = np.concatenate(predicted_strength_values, axis=0)

    return actual_strength_values, predicted_strength_values



actual_values, predicted_values = TestGeneModel(seq_testing_batches, testing_motifs_batches, testing_motifs_positions_batches, testing_strength_batches)

import csv

print((actual_values[1]))


import torch
actual_values = torch.tensor(actual_values, device='cuda:1')
predicted_values = torch.tensor(predicted_values, device='cuda:1')


actual_values = actual_values.cpu().numpy()
predicted_values = predicted_values.cpu().numpy()

data = {
    "Actual Values": actual_values,
    "Predicted Values": predicted_values
}
df = pd.DataFrame(data)

file_name = "/local/data/sdahal_p/genome-test/result/strength_values.csv"

df.to_csv(file_name, index=False)

# with open('both.txt', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Actual Values", "Predicted Values"])
#     writer.writerows(zip(actual_values, predicted_values))