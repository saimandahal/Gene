import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

import random

import pandas as pd
import numpy as np

import os
import sys

import math

import time

import torch.nn.init as init


from dataloader_1 import load_data

from torch.nn import TransformerEncoder, TransformerEncoderLayer

    # GPU
if torch.cuda.is_available():
    print("GPU is running.")
    dev = "cuda"
else:
    print("CPU is running.")
    dev = "cpu"

device = torch.device(dev)


dataset_all_rows_total = pd.read_csv('/local/data/sdahal_p/DNA/genome-test/data/select4_Celllines_final_logScale.tsv', sep='\t', low_memory=False)
dataset_all_rows = dataset_all_rows_total[dataset_all_rows_total['cell_line'] == 'A549']

# dataset_all_rows = dataset_all_rows.sample(frac=0.05, random_state= 42)

print(dataset_all_rows.shape)

motif_columns = dataset_all_rows.columns[3:-2]
filtered_dfs_dict = {}

for motif_col in motif_columns:
    filtered_df = dataset_all_rows[pd.notna(dataset_all_rows[motif_col])]
    filtered_dfs_dict[motif_col] = filtered_df


from collections import defaultdict

motif_position_quantiles = {}

quantiles_to_extract = [0.0, 0.25, 0.5, 0.75, 1.0]

for motif_col in motif_columns:
    df = filtered_dfs_dict[motif_col].copy()

    if df[motif_col].isna().all():
        continue
    df['motif_position'] = df[motif_col].str.split(':').str[-1].astype(int)
    grouped = df.groupby('motif_position')['ReadStr']
    position_quantile_dict = {}

    for pos, values in grouped:
        values = values.dropna().astype(float).values

        if len(values) == 0:
            quantile_values = [0.0] * len(quantiles_to_extract)
        else:
            q = np.quantile(values, quantiles_to_extract)
            quantile_values = [float(x) for x in q] 

        position_quantile_dict[pos] = quantile_values

    motif_position_quantiles[motif_col] = position_quantile_dict

test_chromosomes = ['chr12', 'chr17', 'chr18', 'chr5']
testing_data_rows = dataset_all_rows[dataset_all_rows['Chr'].isin(test_chromosomes)]
training_data_rows = dataset_all_rows.drop(testing_data_rows.index)

training_data = load_data(training_data_rows, motif_position_quantiles)
testing_data = load_data(testing_data_rows, motif_position_quantiles)

training_token_batches, trainining_motifs_positions_batches, training_strength_batches, training_class_batches, training_sequence_batches  = training_data
testing_token_batches, testing_motifs_positions_batches, testing_strength_batches, testing_class_batches, testing_sequence_batches = testing_data

print("Length of training data: ", len(training_token_batches))
print("Length of testing data: ", len(testing_token_batches))


class TransGene(nn.Module):
    def __init__(self, input_dim=25, model_dim=128, n_output_heads=1, nhead=4, enc_layers_count=2, motif_count=1):
        super().__init__()
        
        self.input_dim = input_dim
        self.model_dim = 128
        self.motif_count = motif_count
        self.n_output_heads = 1
        self.nhead = 4
        self.enc_layers_count = 2
        self.class_dim = 6
        self.position_dim = 1
        self.seq_dim = 2


        self.class_embedding = torch.nn.Linear(self.class_dim, self.model_dim)
        self.position_embedding = torch.nn.Linear(self.position_dim, self.model_dim)
        self.sequence_embedding = torch.nn.Linear(self.seq_dim, self.model_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.nhead,
            dim_feedforward=1024,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.enc_layers_count)
        
        self.genetics_classifier_1 = torch.nn.Linear(self.model_dim, 1)
        self.activation_relu = torch.nn.LeakyReLU()
        self.dropout_5 = torch.nn.Dropout(p=0.05)

        self._init_weights()

    def _init_weights(self, mean=0.0, std=0.02):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=mean, std=std)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                init.normal_(module.weight, mean=mean, std=std)

    def forward(self, inputs, positions, class_seq, sequence_input_cnn):
        class_seq = class_seq.unsqueeze(0) 
        class_embed = self.activation_relu(self.class_embedding(class_seq))
        pos_embed= self.activation_relu(self.position_embedding(positions))
        seq_embed = self.activation_relu(self.sequence_embedding(sequence_input_cnn))

        seq_embed = seq_embed.sum(dim=0, keepdim=True)

        input_encoder = class_embed + pos_embed

        input_encoder = input_encoder + seq_embed
        
        encoded_x = input_encoder

        for layer in self.encoder.layers:
            encoded_x, attention_output = self._encoder_layer_forward_with_attention(layer, encoded_x)

        classifier_input = (encoded_x + input_encoder)

        all_preds = self.genetics_classifier_1(classifier_input) 

        weighted_output = all_preds.sum(dim=1, keepdim=True) 

        return all_preds, weighted_output.reshape(-1,), attention_output

    def _encoder_layer_forward_with_attention(self, layer, src):

        # Self attention part
        src2, attn_weights = layer.self_attn(
            src, src, src,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=True
        )

        src = src + layer.dropout1(src2)
        src = layer.norm1(src)
        
        # Feedforward part
        src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
        src = src + layer.dropout2(src2)
        src = layer.norm2(src)
        
        return src, attn_weights


from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from IPython.display import clear_output

gene_model = TransGene(input_dim = 25, model_dim = 128, n_output_heads= 1, nhead= 4 , enc_layers_count= 2, motif_count = 1)

gene_model = gene_model.to(device=device)

optimizer_gene = torch.optim.AdamW(gene_model.parameters(), lr= 1e-5, weight_decay=1e-6, betas=(0.9, 0.999), eps=1e-08)

scheduler_gene = CosineAnnealingLR(optimizer_gene, T_max = 32)

# print(gene_model)

# for name,param in gene_model.named_parameters():
#     if param.requires_grad:
#         print(name)

# torch.save(gene_model, "/local/data/sdahal_p/Motif/models/main_bining_model.pth") 
# quit(0)

epoch_number = 1
epoch_end_count = 0
train_epoch_avg_losses_sp = []

start_time = time.time()

gene_model.train(True)

weight_all_iteration_encoder = []

weight_all_iteration_classifier = []

patience = 5

early_stop_counter = 0

best_loss = 9999

training_loss = []
validation_loss= []

for epoch in range (32):

    temp_holder = list(zip(training_token_batches, trainining_motifs_positions_batches,training_class_batches, training_strength_batches, training_sequence_batches))
    random.shuffle(temp_holder)
    training_token_batches, trainining_motifs_positions_batches,training_class_batches, training_strength_batches, training_sequence_batches = zip(*temp_holder)

    total_loss = 0

    correct = 0

    total_batches = 0
    batch_size = 1

    val_loss= 0 

    weight_changes_classifier = []

    weight_changes_encoder = []

    for name, param in gene_model.named_parameters():

        if name == 'genetics_classifier_1.weight':
        
            previous_weight_classifier = param.clone()

    for name, param in gene_model.named_parameters():

        if name == 'encoder.layers.1.self_attn.in_proj_weight':
        
            previous_weight_encoder = param.clone()

    gene_model.train()

    for seq_index , batch_input in enumerate(training_token_batches):


        total_batches +=1

        position_input = trainining_motifs_positions_batches[seq_index].reshape(1,-1,1).to(device)

        class_sequence = training_class_batches[seq_index].to(device)

        actual_output = training_strength_batches[seq_index].reshape(-1).to(device)

        sequence_input = training_sequence_batches[seq_index].to(device)

        tokens, features = batch_input.shape
        
        batch_input = batch_input.reshape(batch_size,tokens,features)
        batch_input = batch_input.to(device)

        optimizer_gene.zero_grad()

        all_preds, weighted_output , attention_output = gene_model(batch_input, position_input, class_sequence, sequence_input)

        loss_fn = nn.MSELoss()

        loss = loss_fn(weighted_output,actual_output)

        if total_batches % 1000 == 0:

            for name, param in gene_model.named_parameters():

                if name == 'genetics_classifier_1.weight':
                
                    current_weight = param.clone()

                    weight_change = torch.norm(current_weight - previous_weight_classifier , p='fro')

                    weight_changes_classifier.append(weight_change.item())

                    previous_weight_classifier = current_weight.clone()

            for name, param in gene_model.named_parameters():

                if name == 'encoder.layers.1.self_attn.in_proj_weight':
                
                    current_weight = param.clone()

                    weight_change = torch.norm(current_weight - previous_weight_encoder , p='fro')

                    weight_changes_encoder.append(weight_change.item())

                    previous_weight_encoder = current_weight.clone()

        loss.backward()

        optimizer_gene.step()

        total_loss+=loss.item()

    scheduler_gene.step()

    training_loss.append(total_loss/total_batches)

    gene_model.eval()

    val_loss, val_correct = 0, 0

    with torch.no_grad():

        val_batches = 0

        for input_index , batch_input in enumerate(testing_token_batches):

            val_batches +=1

            batch_size = 1

            position_input = testing_motifs_positions_batches[input_index].reshape(1,-1,1)

            class1 = testing_class_batches[input_index]

            class_input = testing_class_batches[input_index]

            sequence_input = testing_sequence_batches[input_index].to(device)

            actual_output = testing_strength_batches[input_index].reshape(-1).to(device)

            tokens, features = batch_input.shape

            batch_input = batch_input.reshape(batch_size,tokens,features)

            batch_input = batch_input.to(device)
            position_input = position_input.to(device)
            class_input = class_input.to(device)    
            
            all_preds, weighted_output , attention_output = gene_model(batch_input, position_input, class_input, sequence_input)
            
            loss_fn = nn.MSELoss()

            val_loss_epoch = loss_fn(weighted_output,actual_output)

            val_loss += val_loss_epoch.item()

            # val_correct += (output.argmax(dim=1) == actual_output).sum().item()

        validation_loss.append(val_loss/val_batches)

        currentScore = (val_loss/val_batches)
        
    print('Epoch Number: {} Training Loss : {}'.format(epoch + 1, total_loss / (total_batches)))
    print('Validation Loss : {}'.format( val_loss / (val_batches)))

    train_epoch_avg_losses_sp.append(total_loss / total_batches)
    
    weight_all_iteration_encoder.append(weight_changes_encoder)

    weight_all_iteration_classifier.append(weight_changes_classifier)

    torch.save(gene_model, "/local/data/sdahal_p/Motif/models/main_bining_model.pth") 

    flattened_encoder = [item for sublist in weight_all_iteration_encoder for item in sublist]
    flattened_classifier = [item for sublist in weight_all_iteration_classifier for item in sublist]
    max_length = max(len(flattened_encoder), len(flattened_classifier))
    flattened_encoder.extend([None] * (max_length - len(flattened_encoder)))
    flattened_classifier.extend([None] * (max_length - len(flattened_classifier)))
    df = pd.DataFrame({'Encoder': flattened_encoder, 'Classifier': flattened_classifier})
    df.to_csv('/local/data/sdahal_p/Motif/results/weights_new.csv', index=False)


    if currentScore < best_loss:
        
        best_loss = currentScore
        early_stop_counter = 0

    else:
        
        early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print("Early stopping triggered")
            break

end_time = time.time()

print("Time Elapsed:", end_time - start_time)
print("")
print("Training Epoch Losses:", train_epoch_avg_losses_sp)
print("Val loss", validation_loss)

# Eval

gene_model.eval()


test_loss = []

def TestGeneModel(seq_representation_inputs, position_inputs,class_inputs, strength_output, testing_sequences):

    actual_strength_values = []

    predicted_strength_values = []

    total_batches = 0

    with torch.no_grad():

        for input_index , batch_input in enumerate(seq_representation_inputs):

            total_batches +=1

            batch_size = 1

            position_input = position_inputs[input_index].reshape(1,-1,1)

            class1 = class_inputs[input_index]

            class_input = class_inputs[input_index]

            sequence_input = testing_sequences[input_index].to(device)

            actual_output = strength_output[input_index].reshape(-1).to(device)

            tokens, features = batch_input.shape

            batch_input = batch_input.reshape(batch_size,tokens,features)

            batch_input = batch_input.to(device)
            position_input = position_input.to(device)
            class_input = class_input.to(device)    
            sequence_input = sequence_input.to(device)

            all_preds, weighted_output , attention_output = gene_model(batch_input, position_input, class_input, sequence_input)

            actual_strength_values.append(actual_output)

            predicted_strength_values.append(weighted_output)

    return actual_strength_values, predicted_strength_values, attention_output

actual_values, predicted_values, attention_output = TestGeneModel(testing_token_batches, testing_motifs_positions_batches,testing_class_batches, testing_strength_batches, training_sequence_batches)


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

file_name = "/local/data/sdahal_p/Motif/results/strength_main_bining_model.csv"

df.to_csv(file_name, index=False)