from tqdm import tqdm
import re
import pickle
from sentence_transformers import SentenceTransformer

path = '/home/whut4/liyafei/dataset/TB/'
name = 'TB'

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')


# log_embeddings = []  # List to store log embeddings
# # Read the first 2 million lines from the "normal.log" file
# with open(path + 'normal.log', 'r', encoding='utf-8') as file:
#     lines = file.readlines()[:2000000]
# print('read')
# # Iterate over each line and embed the log
# for line in lines:
#     line = line.strip()  # Remove leading/trailing whitespaces
#     embedded_line = model.encode([line])  # Embed the log using SentenceTransformer
#     log_embeddings.append(embedded_line[0])  # Store the embedded log
# print('embedding')
# # Save the log embeddings to a pickle file
# with open(path + name + '_normal.pickle', 'wb') as file:
#     pickle.dump(log_embeddings, file)
# print('dump')

log_embeddings = []  # List to store log embeddings
# Read the first 2 million lines from the "normal.log" file
with open(path + 'abnormal.log', 'r', encoding='utf-8') as file:
    lines = file.readlines()
print('read')
# Iterate over each line and embed the log
for line in lines:
    line = line.strip()  # Remove leading/trailing whitespaces
    embedded_line = model.encode([line])  # Embed the log using SentenceTransformer
    log_embeddings.append(embedded_line[0])  # Store the embedded log
print('embedding')
# Save the log embeddings to a pickle file
with open(path + name + '_abnormal.pickle', 'wb') as file:
    pickle.dump(log_embeddings, file)
print('dump')