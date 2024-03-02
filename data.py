import torch, gdown, os ,  cv2, random, pandas as pd, numpy as np, pickle as p
from matplotlib import pyplot as plt
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset , random_split
from glob import glob
from PIL import Image
class CustomDataset(Dataset):
    def __init__(self, root, transformations = None):
        
        self.transformations = transformations
      
        self.class_name, self.im_ids ,self.im_classes , count = {}, [],[], 0
        
        
        ds = pd.read_csv(f"{root}/cloud_classification_export.csv")
        for ind in range(len(ds)):
            
            im_id = f"{root}/{ds.iloc[ind]['image']}"; im_class = ds.iloc[ind]["choice"]
            self.im_ids.append(im_id); self.im_classes.append(im_class)
            self.im_ids.append(im_id); self.im_classes.append(im_class)
            
            if im_class not in self.class_name: self.class_name[im_class] = count; count+=1


    def __len__(self): return len(self.im_ids)

    def __getitem__(self, idx):
        im = Image.open(self.im_ids[idx]).convert("RGB")
        gt = self.class_name[self.im_classes[idx]]

        if self.transformations is not None: im = self.transformations(im)
        
        return im, gt
 

def get_dls(root, transformations, bs, split = [0.9, 0.05, 0.05]):
    
    ds = CustomDataset(root = root, transformations = transformations)
    
    total_len = len(ds)
    tr_len = int(total_len * split[0])
    vl_len = int(total_len * split[1])
    ts_len = total_len - (tr_len + vl_len)
    
    tr_ds, vl_ds, ts_ds = random_split(dataset = ds, lengths = [tr_len, vl_len, ts_len])
    save_prefix = "Clouds"
    with open(f"{save_prefix}_classes_names.pickle", "wb") as f: p.dump(ds.class_name, f, protocol = p.HIGHEST_PROTOCOL)
    tr_dl = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = 0)
    val_dl = DataLoader(vl_ds, batch_size = bs, shuffle = False, num_workers = 0)
    ts_dl   = DataLoader(ts_ds, batch_size = 1, shuffle = False, num_workers = 0)
    
    return tr_dl, val_dl, ts_dl, ds.class_name

