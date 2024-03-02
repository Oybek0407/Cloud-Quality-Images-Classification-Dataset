import argparse, timm, torch, random, pandas as pd, numpy as np,  pickle as p
from matplotlib import pyplot as plt
from torchvision import transforms as T
from data import get_dls
from utils import visualize, Visualization
from train import train
from train import set_up
from inference import inference
from tqdm import tqdm

def run(args):
        
    mean, std, im_size = [0.2250, 0.2250, 0.2250], [0.2505, 0.2505, 0.2505], 224

    tfs = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])

    tr_dl, val_dl, ts_dl, classes = get_dls(root=args.data_path, transformations= tfs, bs=args.batch_size)
    data_name = {tr_dl: "train", val_dl: "valid", ts_dl: "test"}
    for data, name in data_name.items():
            visualize(data=data.dataset, num_im=args.namber_im, rows=args.rows, cmap= args.rgb, class_name= list(classes.keys()), save_file= args.save_file, data_name=name)

    print(f"\n Sample images are being saved in a file named {args.save_file}!\n")
    # Model  
    model =timm.create_model(model_name= args.model_nomi, pretrained=True, num_classes = len(classes))
    set_up(model = model)
    device, model, optimazer, loss_fn, epochs = set_up(model)
        
    result = train(model = model, tr_dl = tr_dl, val_dl = val_dl, loss_fn = loss_fn, epochs = epochs, opt = optimazer,
          device = device, save_prefix = args.save_file, save_dir = args.save_dir, threshold = 0.001)
    
    print("\nTrain finished !\n")  
        
    Visualization(i = result,  save_file = args.learn_rate, data_name = args.save_file)
        
    print(f"\n학습률 조정이 완료되고 결과가  {args.save_file}에 저장 되었습니다\n")
        
    model.load_state_dict(torch.load(f"{args.save_dir}/{args.save_file}_best_model.pth"))
    model.eval()
    inference(model = model.eval(), data = ts_dl, device = device ,num_im = 20, row =4, class_name = list(classes.keys()), 
              im_dim = 224, save_inference_images  = args.save_dirs, data_name =args.save_file)

    print(f"\nInference 과정이 완료되었고 GrandCAM의 결과를 {args.save_dirs}  파일에서  확인 가능합니다!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lentils Types Classification Demo")
    parser.add_argument("-dp", "--data_path", type=str, default="cloud_classification", help="path of dataset")
    parser.add_argument("-nm", "--namber_im", type=str, default= 20 , help=" number of images")
    parser.add_argument("-rw", "--rows", type=str, default= 4, help="Classes names")
    parser.add_argument("-rb", "--rgb", type=str, default= 'rgb', help=" red green blue ")
    parser.add_argument("-sd", "--save_dir", type=str, default= "saved_models", help="Save the model")
    parser.add_argument("-sp", "--save_dirs", type=str, default= "save_inference_images", help="Save the model")
    parser.add_argument("-lr", "--learn_rate", type=str, default= "save_learning_rate", help="Save the model")
    parser.add_argument("-sv", "--save_file", type=str, default="Clouds", help="File for saving visualization sample images") 
    parser.add_argument("-bs", "--batch_size", type=str, default= 32, help="Batch_size")         
    parser.add_argument("-mn", "--model_nomi", type=str, default="rexnet_150", help="AI Model Nomi")
    parser.add_argument("-my", "--model_yulagi", type=str, default="saved_models/Clouds_best_model.pth", help="Trained model")

    args = parser.parse_args()
    run(args)



