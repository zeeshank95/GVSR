import json

with open('data/vsitu_vocab/verb_voc.json', 'r') as f:
    v=json.load(f)  

class vb_vocab:
    def __init__(self):
        self.indices = v
        self.idx2vb = {}
        self.symbols = []
        for key,value in self.indices.items():
            self.idx2vb[value] = key
            self.symbols.append(key)
        self.unk_index = v['<unk>']
        self.pad_index = v['<pad>']

    def __getitem__(self, arg):
        return self.idx2vb[arg]
    
    def __len__(self):
         return self.indices.__len__()
    
