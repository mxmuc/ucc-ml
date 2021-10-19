from tkinter import *
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageTk

BERT_MODEL  = 'bert-base-multilingual-cased'
DROPOUT     = 0.3
HIDDEN_SIZE = 768

class Classifier(nn.Module):
  def __init__(self):
    super(Classifier, self).__init__()
    self.bert    = BertModel.from_pretrained(BERT_MODEL)
    self.drop    = nn.Dropout(p = DROPOUT)
    self.linear  = nn.Linear(HIDDEN_SIZE, 2)

  def forward(self, input_ids, attention_mask):
    last_hidden_state, pooler_output = \
      self.bert(
        input_ids       = input_ids,
        attention_mask  = attention_mask,
        return_dict     = False
      )

    corrupted = self.drop(pooler_output)
    out       = self.linear(corrupted)
    
    return out

  def encode(self, input_ids, attention_mask):
    return self.bert(
      input_ids       = input_ids,
      attention_mask  = attention_mask,
      return_dict     = False
    )

def predict_new_message(message, model, tokenizer, max_len=256):
        encoding = tokenizer.encode_plus(
            message,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            )

        input_ids = encoding['input_ids'].flatten().unsqueeze(0)

        #print(input_ids)
        attention_mask = encoding['attention_mask'].flatten().unsqueeze(0)

        outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        _, preds = torch.max(outputs, dim=1)

        return preds.item()

class Dialog:
    def __init__(self):
        self.model = Classifier()
        self.model.load_state_dict(torch.load('../model/clf.bin', map_location=torch.device('cpu')))
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        def store():
            self.new = self.entry.get()
            prediction = predict_new_message(self.new, self.model, self.tokenizer)
            if prediction == 0:
                a.change('That is clearly a FIRST level task!')
            else: 
                a.change('This is something for SECOND level!')
            
        self.win = Toplevel()
        self.win.title("UCC Support Level Classifier")
        self.win.configure()


        path = "./ucc_logo.gif"

        self.img = ImageTk.PhotoImage(Image.open(path))

        self.panel = Label(self.win, image = self.img)
        self.panel.image = self.img
        self.panel.pack()
        
        self.label = Label(self.win, text="Enter a message to classify:")
        self.label.pack()

        self.l = Label(self.win)

        self.entry = Entry(self.win, width=50)
        self.entry.pack()

        self.b1 = Button(self.win, text='Classify', width=10,command=store)
        self.b1.pack()
        
    def __str__(self): 
        return str(self.new)

    def change(self,ran_text):
        self.l.config(text=ran_text,font=(0,12))
        self.l.pack()


root = Tk()
root.withdraw()

a = Dialog()

root.mainloop()

