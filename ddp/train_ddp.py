import os
import time
import datetime
import json
import io
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import random
from contextlib import redirect_stdout

try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TENSORBOARD = True
except ImportError:
    USE_TENSORBOARD = False

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

        boxes = []
        labels = []
        areas = []
        iscrowds = []

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

class CocoValSubset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform, subset_ids):
        self.dataset = CocoDetection(root, annFile, transform=transform)
        self.indices = [i for i, (img, target) in enumerate(self.dataset)
                        if target and target[0]['image_id'] in subset_ids]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def get_dataset(root, annFile, subset_size=12000):
    coco = COCO(annFile)
    all_img_ids = list(coco.imgs.keys())
    selected_img_ids = random.sample(all_img_ids, min(subset_size, len(all_img_ids)))
    return CocoSubsetDetection(root, annFile, selected_img_ids, get_transform())

def get_val_loader(root, annFile, transform, batch_size=8, subset_size=2000):
    coco = COCO(annFile)
    all_img_ids = list(coco.imgs.keys())
    selected_img_ids = set(random.sample(all_img_ids, subset_size))

    dataset = CocoValSubset(root, annFile, transform, selected_img_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=4)

def evaluate_model(model, dataloader, device, annFile):
    model.eval()
    coco_gt = COCO(annFile)
    results = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                if len(target) == 0:
                    continue
                image_id = int(target[0]['image_id'])
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                boxes[:, 2:] -= boxes[:, :2]
                for box, score, label in zip(boxes, scores, labels):
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box.tolist(),
                        "score": float(score)
                    })

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    coco_dt = coco_gt.loadRes("results.json")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    print(f"[GPU {rank}] Starting process")
    if world_size > 1:
        setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dataset = get_dataset(args['data_root'], args['ann_file'], args['subset_size'])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(dataset, batch_size=args['batch_size'], sampler=sampler,
                            shuffle=(sampler is None), collate_fn=collate_fn,
                            num_workers=4, pin_memory=True, persistent_workers=False)

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args['lr'], momentum=0.9, weight_decay=0.0005)

    writer = SummaryWriter(log_dir=f'logs/ddp_rank{rank}') if rank == 0 and USE_TENSORBOARD else None

    if rank == 0 and not os.path.exists("results_ddp.csv"):
        with open("results_ddp.csv", "w") as f:
            f.write("epoch,avg_loss,epoch_time_sec,gpu_count\n")

    for epoch in range(args['epochs']):
        model.train()
        if sampler: sampler.set_epoch(epoch)
        epoch_start = time.time()
        total_loss = 0

        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(dataloader)

        if rank == 0:
            print(f"[GPU {rank}] Epoch {epoch+1}/{args['epochs']}, Avg Loss: {avg_loss:.4f}, Time: {str(datetime.timedelta(seconds=int(epoch_time)))}")
            with open("results_ddp.csv", "a") as f:
                f.write(f"{epoch+1},{avg_loss:.4f},{int(epoch_time)},{world_size}\n")
            if writer:
                writer.add_scalar("Loss/train", avg_loss, epoch)
                writer.add_scalar("Time/epoch", epoch_time, epoch)

            val_loader = get_val_loader(args['val_root'], args['val_ann'], get_transform(), args['batch_size'])
            print(f"[GPU {rank}] Evaluating after epoch {epoch+1}...")
            f_output = io.StringIO()
            with redirect_stdout(f_output):
                mAP = evaluate_model(model.module if world_size > 1 else model, val_loader, device, args['val_ann'])
            print(f_output.getvalue())
            with open("results_ddp.csv", "a") as f:
                f.write(f"epoch_{epoch+1}_mAP,{mAP:.4f}\n")

    if rank == 0:
        torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), "fasterrcnn_ddp.pth")
        if writer:
            writer.close()

    if world_size > 1:
        cleanup()

def run_training(world_size):
    args = {
        'data_root': 'coco_dataset/train2017',
        'ann_file': 'coco_dataset/annotations/instances_train2017.json',
        'val_root': 'coco_dataset/val2017',
        'val_ann': 'coco_dataset/annotations/instances_val2017.json',
        'batch_size': 8,
        'epochs': 5,
        'lr': 0.005,
        'subset_size': 12000
    }
    if world_size > 1:
        mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train(rank=0, world_size=1, args=args)

if __name__ == "__main__":
    run_training(world_size=4)
