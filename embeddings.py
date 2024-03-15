import pandas as pd
import json
import torch
from transformers import AutoModel, AutoTokenizer

df = pd.read_csv('yields_data_1000.csv')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(1)

with torch.no_grad():
 model = AutoModel.from_pretrained('mistralai/Mistral-7B-v0.1', torch_dtype=torch.float16).to(device)
 tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
 results = []
 for idx in df.index:
   smiles = df.loc[idx, 'reaction']
   yield_class = df.loc[idx, 'High-yielding']
   enc = tokenizer(smiles, return_tensors="pt", truncation=True, max_length=512).to(device)
   emb = model(**enc).last_hidden_state.cpu()[0][-1]
   results.append({
           "input": smiles,
           "class": yield_class,
           "embedding": emb
   })

with open("results.json", "w") as file:
   json.dump(
       list(results),
       file)