import torch, gdown, os ,  cv2, random, pandas as pd, numpy as np, pickle as p
from matplotlib import pyplot as plt
from torchvision import transforms as T

def tr_2_im(t, type = "rgb"):
    gray = T.Compose([T.Normalize(mean = [ 0.], std = [1/0.5]), T.Normalize(mean = [-0.5], std = [1])])
    rgb = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.2505, 1/0.2505, 1/0.2505]),
                         T.Normalize(mean = [ -0.2250, -0.2250, -0.2250 ], std = [ 1., 1., 1. ])])
    inp = gray if type == "gray" else rgb
    return(inp(t)*255).detach().squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8) if type == "gray" else (inp(t)*255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def visualize(data, num_im, rows, cmap =None, class_name =None, save_file =None, data_name = None):
    os.makedirs(save_file, exist_ok = True)
    assert cmap in ["rgb", "gray"]
    if cmap == "rgb": cmap = "RdBu"
    plt.figure(figsize=(18, 10))
    index = [random.randint(0, len(data)-1) for _ in range(num_im)]
    for i, idx in enumerate(index):
        im, gt = data[idx]
        plt.subplot(rows, num_im//rows, i+1)
        plt.imshow(tr_2_im(im, cmap), cmap="RdBu")
        plt.imshow(tr_2_im(im))
        plt.axis("off")
        plt.title(f"GT -> {class_name[gt]}")
        plt.savefig(f"{save_file}/{data_name}.png")
#visualize(data=tr_dl.dataset, num_im= 20, rows=4, cmap = 'rgb', class_name=list(classes.keys()))


def Visualization(i, save_file=None, data_name=None):
    os.makedirs(save_file, exist_ok=True)

    # Plot and save accuracy figure
    plt.figure(figsize=(8, 5))
    plt.plot(i["tr_acc_sc"], label="Train Accuracy Score")
    plt.plot(i["val_acc_sc"], label="Validation Accuracy Score")
    plt.title("Train and Validation Accuracy Score")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Score")
    plt.xticks(np.arange(len(i["val_acc_sc"])), [i for i in range(1, len(i["val_acc_sc"]) + 1)])
    plt.legend()
    plt.savefig(f"{save_file}/{data_name}_Accuracy.png")
  

    # Plot and save loss figure
    plt.figure(figsize=(8, 5))
    plt.plot(i["tr_loss_cs"], label="Train Losses")
    plt.plot(i["val_loss_cs"], label="Validation Losses")
    plt.title("Train and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Score")
    plt.xticks(np.arange(len(i["val_acc_sc"])), [i for i in range(1, len(i["val_acc_sc"]) + 1)])
    plt.legend()
    plt.savefig(f"{save_file}/{data_name}_loss.png")
   
            
