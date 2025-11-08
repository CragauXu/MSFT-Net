import argparse
import torch
from torch.utils.data import DataLoader
from model.MSFTNet import MSFT_Net
from utils.dataset import Dataset_Multimodal
import albumentations as A
from utils.metrics  import calculate_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Testing script for the model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model.")
    parser.add_argument('--test_csv', type=str, required=True, help="Path to the test CSV file.")
    parser.add_argument('--root_path', type=str, required=True, help="Root path of the dataset.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for testing.")
    return parser.parse_args()

def test(args):
    # Initialize model
    model = MSFT_Net(img_size=224, num_classes=2, patch_size=16, embed_dim=768, num_heads=12, sparse_ratio1=0.7, sparse_ratio2=0.7)
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    model.eval()

    # Define transforms
    test_transforms = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2()
    ])

    # Prepare dataset and dataloader
    test_dataset = Dataset_Multimodal(args.root_path, 16, args.test_csv, test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Testing loop
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for video_b, video_c, video_e, labels in test_dataloader:
            video_b, video_c, video_e, labels = video_b.cuda(), video_c.cuda(), video_e.cuda(), labels.cuda()
            outputs = model(video_b, video_c, video_e)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
            preds = outputs.argmax(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    metrics = calculate_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs)
    )

    # Print metrics
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

if __name__ == "__main__":
    args = parse_args()
    test(args)