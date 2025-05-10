import os
import time
import datetime
import json
import io
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import random
from contextlib import redirect_stdout
import deepspeed
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_transform():
    return T.Compose([T.ToTensor()])

def collate_fn(batch):
    return tuple(zip(*batch))

class CocoSubsetDetection(torch.utils.data.Dataset):
    def __init__(self, root, annFile, img_ids, transform):
        self.root = root
        self.coco = COCO(annFile)
        self.img_ids = img_ids
        self.transform = transform

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowds = [], [], [], []
        for ann in anns:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowds.append(ann['iscrowd'])

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = torch.tensor(labels, dtype=torch.int64)[keep]
            areas = torch.tensor(areas, dtype=torch.float32)[keep]
            iscrowds = torch.tensor(iscrowds, dtype=torch.int64)[keep]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)
            areas = torch.tensor([], dtype=torch.float32)
            iscrowds = torch.tensor([], dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowds
        }

        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.img_ids)

def get_dataset(root, annFile, subset_size=400):
    coco = COCO(annFile)
    all_img_ids = list(coco.imgs.keys())
    selected_img_ids = random.sample(all_img_ids, min(subset_size, len(all_img_ids)))
    return CocoSubsetDetection(root, annFile, selected_img_ids, get_transform())

def get_val_loader(root, annFile, transform, batch_size=8, subset_size=2000):
    coco = COCO(annFile)
    all_img_ids = list(coco.imgs.keys())
    selected_img_ids = set(random.sample(all_img_ids, subset_size))
    dataset = CocoSubsetDetection(root, annFile, list(selected_img_ids), transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

def evaluate_model(model, dataloader, device, annFile):
    model.eval()
    coco_gt = COCO(annFile)
    results = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for target, output in zip(targets, outputs):
                image_id = int(target['image_id'].item())
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                boxes[:, 2:] -= boxes[:, :2]  # Convert to [x, y, w, h]
                for box, score, label in zip(boxes, scores, labels):
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box.tolist(),
                        "score": float(score)
                    })

    with open("results_deepspeed.json", "w") as f:
        json.dump(results, f, indent=2)

    coco_dt = coco_gt.loadRes("results_deepspeed.json")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed from deepspeed")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Load model
    base_model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    base_model.train()

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=base_model,
        model_parameters=[p for p in base_model.parameters() if p.requires_grad],
        config=args.deepspeed_config
    )

    # Load training and validation datasets
    dataset = get_dataset('coco_dataset/train2017', 'coco_dataset/annotations/instances_train2017.json')
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=8, collate_fn=collate_fn, num_workers=4)

    val_loader = get_val_loader('coco_dataset/val2017', 'coco_dataset/annotations/instances_val2017.json', get_transform())

    if local_rank == 0 and not os.path.exists("results_zero.csv"):
        with open("results_zero.csv", "w") as f:
            f.write("epoch,avg_loss,epoch_time_sec,gpu_count\n")

    for epoch in range(5):
        sampler.set_epoch(epoch)
        total_loss = 0
        start_time = time.time()

        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model_engine(images, targets)

            if isinstance(loss_dict, dict):
                loss = sum(loss_dict.values())
            else:
                raise TypeError(f"[Rank {local_rank}] Expected dict for loss_dict, got {type(loss_dict)} with content: {loss_dict}")

            model_engine.backward(loss)
            model_engine.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(dataloader)

        if local_rank == 0:
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Time: {str(datetime.timedelta(seconds=int(epoch_time)))}")
            with open("results_zero.csv", "a") as f:
                f.write(f"{epoch+1},{avg_loss:.4f},{int(epoch_time)},{dist.get_world_size()}\n")

            # Use base model (not DeepSpeed wrapped) for evaluation
            base_model.eval()
            mAP = evaluate_model(base_model, val_loader, device, 'coco_dataset/annotations/instances_val2017.json')
            base_model.train()

            with open("results_zero.csv", "a") as f:
                f.write(f"epoch_{epoch+1}_mAP,{mAP:.4f}\n")

if __name__ == "__main__":
    train()
