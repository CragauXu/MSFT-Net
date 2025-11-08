import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from model.MSFTNet import MSFT_Net
from utils.dataset import Dataset_Multimodal
import albumentations as A
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for the model.")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for the optimizer.")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--model_name', type=str, default='MMtransformer_forzen_MRnet', help="Name of the model.")
    parser.add_argument('--data_name', type=str, default='all', help="Dataset name.")
    parser.add_argument('--root_path', type=str, required=True, help="Root path of the dataset.")
    parser.add_argument('--train_csv', type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument('--val_csv', type=str, required=True, help="Path to the validation CSV file.")
    parser.add_argument('--output_dir', type=str, default='result', help="Directory to save the model and logs.")
    return parser.parse_args()

def train(args):
    # Initialize model
    model = MSFT_Net(img_size=224, num_classes=2, patch_size=16, embed_dim=768, num_heads=12, sparse_ratio1=0.7, sparse_ratio2=0.7)
    model.load_state_dict(torch.load(f'{args.output_dir}/{args.data_name}_{args.model_name}_abnormal.pth'))
    model = model.cuda()

    # Define transforms
    train_transforms = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2()
    ])
    val_transforms = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2()
    ])

    # Prepare datasets and dataloaders
    train_dataset = Dataset_Multimodal(args.root_path, 16, args.train_csv, train_transforms)
    val_dataset = Dataset_Multimodal(args.root_path, 16, args.val_csv, val_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)

    # Training loop
    best_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} - Training")
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0

        for video_b, video_c, video_e, labels in train_dataloader:
            video_b, video_c, video_e, labels = video_b.cuda(), video_c.cuda(), video_e.cuda(), labels.cuda()
            with torch.amp.autocast('cuda'):
                outputs = model(video_b, video_c, video_e)
                loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_accuracy += (outputs.argmax(1) == labels).sum().item()

        print(f"Training Loss: {total_train_loss:.4f}, Accuracy: {total_train_accuracy / len(train_dataset):.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        with torch.no_grad():
            for video_b, video_c, video_e, labels in val_dataloader:
                video_b, video_c, video_e, labels = video_b.cuda(), video_c.cuda(), video_e.cuda(), labels.cuda()
                outputs = model(video_b, video_c, video_e)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
                total_val_accuracy += (outputs.argmax(1) == labels).sum().item()

        val_accuracy = total_val_accuracy / len(val_dataset)
        print(f"Validation Loss: {total_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), f"{args.output_dir}/{args.data_name}_{args.model_name}_best.pth")
            print("Best model saved.")

if __name__ == "__main__":
    args = parse_args()
    train(args)