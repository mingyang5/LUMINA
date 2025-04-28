import os
# os.environ['CUDA_VISIBLE_DEVICES']='6'

import json
import random
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score, accuracy_score
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import clip
from colorama import Fore, Style

warnings.filterwarnings("ignore")


def adjust_unfreeze_rate(epoch, adjust_after=12, increase_rate=2):
    return 1 if epoch < adjust_after else increase_rate

def unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=False):
    if unfreeze_all:
        for param in model.parameters():
            param.requires_grad = True
    else:
        unfreeze_every_n_epochs = adjust_unfreeze_rate(epoch)
        layers_to_unfreeze = min((epoch // unfreeze_every_n_epochs) % total_layers, total_layers)
        for i, (name, param) in enumerate(model.named_parameters()):
            param.requires_grad = i >= total_layers - layers_to_unfreeze

def calculate_metrics(logits, ground_truth):
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(ground_truth.cpu(), preds.cpu())
    f1 = f1_score(ground_truth.cpu(), preds.cpu(), average='weighted')
    return acc, f1

def plot_gradient_norms(gradient_norms, epoch, use_log_scale=True):
    plt.figure(figsize=(20, 10))
    cmap = plt.get_cmap('Spectral')
    sorted_layers = sorted(gradient_norms.items(), key=lambda item: max(item[1]), reverse=True)
    colors = cmap(range(len(sorted_layers)))

    for (layer_name, norms), color in zip(sorted_layers, colors):
        plt.plot(norms, label=layer_name, color=color)

    plt.xlabel('Batch')
    plt.ylabel('Gradient Norm')
    if use_log_scale:
        plt.yscale('log')
    plt.title(f'Gradient Norms for Epoch {epoch}')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/gradient_norms_epoch_{epoch}.png")
    plt.close()

def plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts):
    epochs_x = range(1, epoch + 2)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs_x, training_losses, label='Training Loss')
    plt.plot(epochs_x, validation_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs_x, logits_images, label='Image Logits')
    plt.plot(epochs_x, logits_texts, label='Text Logits')
    plt.title('Logits Over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/combined_plot_epoch_{epoch + 1}.png")
    plt.close()


class ImageTextDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = self.annotations[self.image_paths[idx]]
        label = random.choice(labels) if len(labels) >= 2 else (labels[0] if labels else '')
        
        if len(label) > 512:
            label = label[:512]
        text = clip.tokenize([label], truncate=True)

        return image, text.squeeze(0)



class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits_per_image, logits_per_text):
        labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
        loss_img = self.criterion(logits_per_image, labels)
        loss_txt = self.criterion(logits_per_text, labels)
        return (loss_img + loss_txt) / 2



def trainloop(model, train_dataloader, optimizer, scheduler, scaler, device="cuda", unfreeze_all=True):
    contrastive_loss = ContrastiveLoss().to(device)
    logits_images, logits_texts = [], []

    for epoch in range(EPOCHS):
        unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=unfreeze_all)
        model.train()
        total_train_loss, train_accs, train_f1s = 0.0, [], []
        gradient_norms = {}

        for batch_idx, (images, texts) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{EPOCHS}'):
            images, texts = images.to(device), texts.to(device)
            optimizer.zero_grad()
            with autocast():
                logits_per_image, logits_per_text = model(images, texts)
                loss = contrastive_loss(logits_per_image, logits_per_text)
                acc, f1 = calculate_metrics(logits_per_image, torch.arange(images.size(0), device=device))
                train_accs.append(acc)
                train_f1s.append(f1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_train_loss += loss.item()

            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    gradient_norms.setdefault(name, []).append(parameter.grad.norm().item())

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_losses.append(avg_train_loss)

        # Validation logic (currently commented out, can be enabled later)
        # model.eval()
        # total_val_loss, val_accs, val_f1s = 0.0, [], []
        # with torch.no_grad():
        #     for images, texts in val_dataloader:
        #         images, texts = images.to(device), texts.to(device)
        #         logits_per_image, logits_per_text = model(images, texts)
        #         val_loss = contrastive_loss(logits_per_image, logits_per_text)
        #         acc, f1 = calculate_metrics(logits_per_image, torch.arange(images.size(0), device=device))
        #         val_accs.append(acc)
        #         val_f1s.append(f1)
        #         total_val_loss += val_loss.item()
        # avg_val_loss = total_val_loss / len(val_dataloader)
        # validation_losses.append(avg_val_loss)
        # plot_gradient_norms(gradient_norms, epoch)
        # if epoch >= 1:
        #     plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts)
        # print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {sum(val_accs)/len(val_accs):.4f} | Val F1: {sum(val_f1s)/len(val_f1s):.4f}" + Style.RESET_ALL)

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            model_path = f"{ft_checkpoints_folder}/clip_ft_epoch_{epoch+1}.pt"
            torch.save(model, model_path)
            print(Fore.GREEN + f"Model saved to {model_path}" + Style.RESET_ALL)


if __name__ == "__main__":
    plots_folder = './datasets/finetuning_results/ft-plots'
    ft_checkpoints_folder = './datasets/finetuning_results/ft-checkpoints'
    text_logs_folder = './datasets/finetuning_results/ft-logs'
    for folder in [plots_folder, ft_checkpoints_folder, text_logs_folder]:
        os.makedirs(folder, exist_ok=True)

    clip_model_name = 'ViT-L/14'
    # clip_model_name = "ViT-B/32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 50
    learning_rate = 5e-7
    batch_size = 64
    unfreeze_all = True

    train_image_folders = [
        "datasets/DiffusionDB"
    ]
    train_labels_path = "datasets/finetuning_data/aigc_diffusiondb_labels.json"
    # val_image_folder = "path/to/validation/image/folder"
    # val_labels_path = "path/to/validation-labels.json"
    
    model, preprocess = clip.load(clip_model_name, device=device)
    model = model.float()

    datasets = [ImageTextDataset(folder, train_labels_path, transform=preprocess) for folder in train_image_folders]
    train_dataloader = DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=True)
    # val_dataset = ImageTextDataset(val_image_folder, val_labels_path, transform=preprocess)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9, 0.995), weight_decay=1e-3)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.1, anneal_strategy='linear')

    scaler = GradScaler()

    training_losses = []
    # validation_losses = []
    
    print(f"Precision: {model.dtype}")
    print(f"Total batches: {len(train_dataloader)} @ Batch Size: {batch_size}")
    print("== START TRAINING ==\n")
    trainloop(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        unfreeze_all=unfreeze_all
    )