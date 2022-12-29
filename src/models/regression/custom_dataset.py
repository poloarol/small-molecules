""" custom_dataset.py """

import os
from typing import Any

import numpy as np
import pandas as pd
import deepchem as dc
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm


"""
Creating a custom dataset for the torch_geometric models
using aqueous solubility dataset from: https://doi.org/10.1038/s41597-019-0151-1
"""

class MoleculeDataset(Dataset):
    def __init__(self,
                 root: str,
                 filename: str,
                 test: bool = False,
                 transform: Any = None,
                 pre_transform: Any = None,
                 length: int = 0):
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        self.test = test
        self.filename = filename
        self.length = length
    
    @property
    def processed_file_names(self):
        """
        If these files are founf in raw_dir, processing is skipped
        """
        
        processed_files = [file for file in os.listdir(self.processed_dir)\
            if not file.endswith("pre")]
        
        if self.test:
            processed_files = [file for file in processed_files if "test" in file]
            if not processed_files:
                return ["no_files.dummy"]
            self.length = len(processed_files)
            return [f"data_{i}.pt" for i in list(range(self.length))]
        else:
            processed_files = [file for file in processed_files if not "test" in file]
            if not processed_files:
                return ["no_files.dummy"]
            self.length = len(processed_files)
            return [f"data_{i}.pt" for i in list(range(self.length))]
    
    def download(self):
        """
        Implement if needed to trigger raw file download from the web.
        """
    
    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        count: int = 0
        for idx, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            try:
                feat = featurizer.featurize(mol["SMILES"])
                data = feat[0].to_pyg_graph()
                count = count + 1
            except:
                continue
            
            data.y = self.get_label(mol["isSoluble"]) # binary classification label
            data.smiles = mol["SMILES"]
            
            if self.text:
                torch.save(data, 
                           os.path.join(self.processed_dir, f"data_test_{count-1}.pt")
                        )
            else:
                torch.save(data, 
                           os.path.join(self.processed_dir, f"data_{count-1}.pt")
                        )
        
        print(f"Number of molecules included: {count}")
    
    
    def get_label(self, label):
        """
        Returns the label (0/1) for the model: data.y
        """
        label = np.asarray([label])
        return torch.Tensor(label, dtype=torch.int64)
    
    def __len__(self):
        return self.length
    
    def get(self, index):
        """
        Equivalen to __getitem__ in pytorch, not needed for
        pyG InMemoryDataset
        """
        
        if self.test:
            data = torch.load(
                os.path.join(self.processed_dir, f"data_test_{index}.pt")
            )
        else:
            data = torch.load(
                os.path.join(self.processed_dir, f"data_{index}.pt")
            )
            
        return data