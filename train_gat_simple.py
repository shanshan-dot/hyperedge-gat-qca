import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import re
from collections import defaultdict
from edge_gat import GATEncoder


# ====================== Focal Loss ======================
def focal_loss(logits, targets, gamma=2.0, alpha=None, reduction='mean'):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    if alpha is not None:
        alpha_t = alpha[targets]
        focal = alpha_t * (1 - pt) ** gamma * ce_loss
    else:
        focal = (1 - pt) ** gamma * ce_loss
    if reduction == 'mean':
        return focal.mean()
    elif reduction == 'sum':
        return focal.sum()
    else:
        return focal


# ====================== Model: Block X/Y + Offset X/Y + Clock ======================
class SimpleGATModel(nn.Module):
    def __init__(self, gat_encoder, num_blocks=10, block_size=6, num_clock_classes=4):
        super().__init__()
        self.gat_encoder = gat_encoder
        self.num_block_classes = num_blocks
        self.num_offset_classes = block_size
        self.num_clock_classes = num_clock_classes

        emb_dim = gat_encoder.out_emb_dim

        def make_head(out_dim):
            return nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(emb_dim * 2, emb_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(emb_dim, out_dim),
            )

        self.block_x_head = make_head(self.num_block_classes)
        self.block_y_head = make_head(self.num_block_classes)
        self.offset_x_head = make_head(self.num_offset_classes)
        self.offset_y_head = make_head(self.num_offset_classes)

        self.clock_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, self.num_clock_classes),
        )

    def forward(self, graph_data):
        emb = self.gat_encoder(graph_data)
        return (
            self.block_x_head(emb),
            self.block_y_head(emb),
            self.offset_x_head(emb),
            self.offset_y_head(emb),
            self.clock_head(emb),
        )


# ====================== Dataset ======================
class QCASingleStepDataset(Dataset):
    def __init__(self, root_dir, num_blocks=10, block_size=6):
        self.root_dir = Path(root_dir)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.samples = []
        self.function_to_idx = {}
        self._load_data()

    def _load_data(self):
        func_set = set()
        success_dirs = list(self.root_dir.rglob("*_success"))
        if not success_dirs:
            raise ValueError(f"No *_success directory found under {self.root_dir}")

        for ep_dir in success_dirs:
            raw_dir = ep_dir / "raw_graph_data"
            if not raw_dir.exists():
                continue
            step_files = sorted(
                raw_dir.glob("step*_raw.pt"),
                key=lambda x: int(re.search(r'step(\d+)_raw', x.stem).group(1)),
            )
            if not step_files:
                continue
            data = torch.load(step_files[0], weights_only=False)
            func = data.get("function", None)
            if func is not None:
                func_set.add(func)

        self.function_to_idx = {func: idx for idx, func in enumerate(sorted(func_set))}

        max_coord = self.num_blocks * self.block_size - 1
        for ep_dir in success_dirs:
            raw_dir = ep_dir / "raw_graph_data"
            if not raw_dir.exists():
                continue
            step_files = sorted(
                raw_dir.glob("step*_raw.pt"),
                key=lambda x: int(re.search(r'step(\d+)_raw', x.stem).group(1)),
            )
            if not step_files:
                continue

            for pt_file in step_files:
                data = torch.load(pt_file, weights_only=False)
                if 'target_x' not in data or 'target_y' not in data:
                    continue
                x = min(max(int(data['target_x']), 0), max_coord)
                y = min(max(int(data['target_y']), 0), max_coord)

                graph_dict = {
                    "node_features": data["node_features"],
                    "edge_index": data["edge_index"],
                    "hyperedge_adj": data.get("hyperedge_adj"),
                }
                self.samples.append((
                    graph_dict,
                    x // self.block_size,
                    y // self.block_size,
                    x % self.block_size,
                    y % self.block_size,
                    int(data.get("clock_phase", 0)),
                    data.get("function", "unknown"),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_single_step(batch):
    graph_dicts = [item[0] for item in batch]
    block_x_labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    block_y_labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    offset_x_labels = torch.tensor([item[3] for item in batch], dtype=torch.long)
    offset_y_labels = torch.tensor([item[4] for item in batch], dtype=torch.long)
    clock_labels = torch.tensor([item[5] for item in batch], dtype=torch.long)
    func_names = [item[6] for item in batch]
    return graph_dicts, block_x_labels, block_y_labels, offset_x_labels, offset_y_labels, clock_labels, func_names


# ====================== Data augmentation ======================
def drop_edges(edge_index, edge_attr, drop_prob=0.15):
    if drop_prob <= 0 or edge_index.shape[1] == 0:
        return edge_index, edge_attr
    num_edges = edge_index.shape[1]
    keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_prob
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    return edge_index[:, keep_mask], edge_attr[keep_mask] if edge_attr is not None else None


def add_node_noise(node_feat, noise_std=0.01):
    if noise_std <= 0:
        return node_feat
    return node_feat + torch.randn_like(node_feat) * noise_std


def augment_graph(graph_dict, drop_edge_prob=0.15, noise_std=0.01):
    if drop_edge_prob > 0 and graph_dict["edge_index"].shape[1] > 0:
        edge_idx, edge_attr = drop_edges(
            graph_dict["edge_index"],
            graph_dict.get("edge_features", None),
            drop_edge_prob,
        )
        graph_dict["edge_index"] = edge_idx
        if edge_attr is not None:
            graph_dict["edge_features"] = edge_attr
    if noise_std > 0:
        graph_dict["node_features"] = add_node_noise(graph_dict["node_features"], noise_std)
    return graph_dict


# ====================== Loss composition ======================
def compute_loss(block_x_logits, block_y_logits,
                 offset_x_logits, offset_y_logits, clock_logits,
                 block_x_labels, block_y_labels,
                 offset_x_labels, offset_y_labels, clock_labels,
                 block_weight=1.0, offset_weight=3.0, clock_weight=5.0,
                 focal_gamma=2.0, focal_alpha=None):
    loss_block_x = focal_loss(block_x_logits, block_x_labels, gamma=focal_gamma, alpha=focal_alpha)
    loss_block_y = focal_loss(block_y_logits, block_y_labels, gamma=focal_gamma, alpha=focal_alpha)
    loss_offset_x = focal_loss(offset_x_logits, offset_x_labels, gamma=focal_gamma, alpha=focal_alpha)
    loss_offset_y = focal_loss(offset_y_logits, offset_y_labels, gamma=focal_gamma, alpha=focal_alpha)
    loss_clock = F.cross_entropy(clock_logits, clock_labels)
    total = (
        block_weight * (loss_block_x + loss_block_y)
        + offset_weight * (loss_offset_x + loss_offset_y)
        + clock_weight * loss_clock
    )
    return total, loss_block_x, loss_block_y, loss_offset_x, loss_offset_y, loss_clock


# ====================== Evaluation ======================
@torch.no_grad()
def evaluate(model, dataloader, device,
             block_weight=1.0, offset_weight=3.0, clock_weight=5.0):
    model.eval()
    total_loss = 0.0
    total_block_x_acc = 0.0
    total_block_y_acc = 0.0
    total_offset_x_acc = 0.0
    total_offset_y_acc = 0.0
    total_clock_acc = 0.0
    total_steps = 0

    for graph_dicts, block_x_labels, block_y_labels, offset_x_labels, offset_y_labels, clock_labels, _ in dataloader:
        embs = []
        for g in graph_dicts:
            g_data = {
                "node_features": g["node_features"].to(device),
                "edge_index": g["edge_index"].to(device),
                "hyperedge_adj": g["hyperedge_adj"].to(device) if g.get("hyperedge_adj") is not None else None,
            }
            embs.append(model.gat_encoder(g_data).unsqueeze(0))
        embs = torch.cat(embs, dim=0)

        block_x_logits = model.block_x_head(embs)
        block_y_logits = model.block_y_head(embs)
        offset_x_logits = model.offset_x_head(embs)
        offset_y_logits = model.offset_y_head(embs)
        clock_logits = model.clock_head(embs)

        block_x_labels = block_x_labels.to(device)
        block_y_labels = block_y_labels.to(device)
        offset_x_labels = offset_x_labels.to(device)
        offset_y_labels = offset_y_labels.to(device)
        clock_labels = clock_labels.to(device)

        loss, _, _, _, _, _ = compute_loss(
            block_x_logits, block_y_logits,
            offset_x_logits, offset_y_logits, clock_logits,
            block_x_labels, block_y_labels,
            offset_x_labels, offset_y_labels, clock_labels,
            block_weight=block_weight, offset_weight=offset_weight, clock_weight=clock_weight,
        )
        total_loss += loss.item() * block_x_labels.size(0)

        total_block_x_acc += (block_x_logits.argmax(1) == block_x_labels).sum().item()
        total_block_y_acc += (block_y_logits.argmax(1) == block_y_labels).sum().item()
        total_offset_x_acc += (offset_x_logits.argmax(1) == offset_x_labels).sum().item()
        total_offset_y_acc += (offset_y_logits.argmax(1) == offset_y_labels).sum().item()
        total_clock_acc += (clock_logits.argmax(1) == clock_labels).sum().item()
        total_steps += block_x_labels.size(0)

    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
    block_x_acc = total_block_x_acc / total_steps if total_steps > 0 else 0.0
    block_y_acc = total_block_y_acc / total_steps if total_steps > 0 else 0.0
    offset_x_acc = total_offset_x_acc / total_steps if total_steps > 0 else 0.0
    offset_y_acc = total_offset_y_acc / total_steps if total_steps > 0 else 0.0
    clock_acc = total_clock_acc / total_steps if total_steps > 0 else 0.0

    return avg_loss, block_x_acc, block_y_acc, offset_x_acc, offset_y_acc, clock_acc


# ====================== Training ======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_BLOCKS = 10
    BLOCK_SIZE = 6
    BATCH_SIZE = 64
    EPOCHS = 150
    LR = 5e-4
    WEIGHT_DECAY = 1e-5
    GAT_HIDDEN = 64
    GAT_OUT = 64
    GAT_HEADS = 8
    GAT_LAYERS = 2
    DROPOUT = 0.1
    BLOCK_WEIGHT = 1.0
    OFFSET_WEIGHT = 2.0
    CLOCK_WEIGHT = 2.0
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = None
    MODEL_SAVE_PATH = "model/simple_gat_model_split_block_xy_offset_clock.pth"
    LOG_SAVE_PATH = "model/training_logs_split_block_xy_offset_clock.pt"
    PATIENCE = 15

    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(LOG_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

    root_dir = Path(__file__).parent/ "data"

    temp_dataset = QCASingleStepDataset(root_dir, num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    node_feat_dim = temp_dataset[0][0]["node_features"].shape[1]

    max_block_x = max((s[1] for s in temp_dataset.samples), default=0)
    max_block_y = max((s[2] for s in temp_dataset.samples), default=0)
    max_offset_x = max((s[3] for s in temp_dataset.samples), default=0)
    max_offset_y = max((s[4] for s in temp_dataset.samples), default=0)
    max_clock = max((s[5] for s in temp_dataset.samples), default=0)

    NUM_BLOCKS = max(NUM_BLOCKS, max_block_x + 1, max_block_y + 1)
    BLOCK_SIZE = max(BLOCK_SIZE, max_offset_x + 1, max_offset_y + 1)
    NUM_CLOCK_CLASSES = max(4, max_clock + 1)

    full_dataset = QCASingleStepDataset(root_dir, num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_single_step, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_single_step, pin_memory=True)

    gat_encoder = GATEncoder(
        node_feat_dim=node_feat_dim,
        hidden_dim=GAT_HIDDEN,
        out_emb_dim=GAT_OUT,
        dropout_rate=DROPOUT,
        hyperedge_emb_dim=32,
        gat_heads=GAT_HEADS,
        num_gat_layers=GAT_LAYERS,
    ).to(device)

    model = SimpleGATModel(
        gat_encoder=gat_encoder,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        num_clock_classes=NUM_CLOCK_CLASSES,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    no_improve = 0

    logs = {
        'train_loss': [], 'val_loss': [],
        'block_x_acc': [], 'block_y_acc': [],
        'offset_x_acc': [], 'offset_y_acc': [], 'clock_acc': [],
        'lr': [], 'grad_norm': [],
    }

    print(f"Device: {device} | Node feat dim: {node_feat_dim} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for graph_dicts, block_x_labels, block_y_labels, offset_x_labels, offset_y_labels, clock_labels, _ in progress:
            block_x_labels = block_x_labels.to(device)
            block_y_labels = block_y_labels.to(device)
            offset_x_labels = offset_x_labels.to(device)
            offset_y_labels = offset_y_labels.to(device)
            clock_labels = clock_labels.to(device)

            embs = []
            for g in graph_dicts:
                g_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in g.items()}
                g_aug = augment_graph(g_copy, drop_edge_prob=0.15, noise_std=0.01)
                g_data = {
                    "node_features": g_aug["node_features"].to(device),
                    "edge_index": g_aug["edge_index"].to(device),
                    "hyperedge_adj": g_aug["hyperedge_adj"].to(device) if g_aug.get("hyperedge_adj") is not None else None,
                }
                embs.append(model.gat_encoder(g_data).unsqueeze(0))
            embs = torch.cat(embs, dim=0)

            block_x_logits = model.block_x_head(embs)
            block_y_logits = model.block_y_head(embs)
            offset_x_logits = model.offset_x_head(embs)
            offset_y_logits = model.offset_y_head(embs)
            clock_logits = model.clock_head(embs)

            loss, *_ = compute_loss(
                block_x_logits, block_y_logits,
                offset_x_logits, offset_y_logits, clock_logits,
                block_x_labels, block_y_labels,
                offset_x_labels, offset_y_labels, clock_labels,
                block_weight=BLOCK_WEIGHT,
                offset_weight=OFFSET_WEIGHT,
                clock_weight=CLOCK_WEIGHT,
                focal_gamma=FOCAL_GAMMA,
                focal_alpha=FOCAL_ALPHA,
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_grad_norm = total_grad_norm / len(train_loader) if len(train_loader) > 0 else 0.0

        val_loss, block_x_acc, block_y_acc, offset_x_acc, offset_y_acc, clock_acc = evaluate(
            model, val_loader, device,
            block_weight=BLOCK_WEIGHT,
            offset_weight=OFFSET_WEIGHT,
            clock_weight=CLOCK_WEIGHT,
        )
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        logs['train_loss'].append(avg_train_loss)
        logs['val_loss'].append(val_loss)
        logs['block_x_acc'].append(block_x_acc)
        logs['block_y_acc'].append(block_y_acc)
        logs['offset_x_acc'].append(offset_x_acc)
        logs['offset_y_acc'].append(offset_y_acc)
        logs['clock_acc'].append(clock_acc)
        logs['lr'].append(current_lr)
        logs['grad_norm'].append(avg_grad_norm)

        print(f"Epoch {epoch}: train={avg_train_loss:.4f} val={val_loss:.4f} "
              f"bx={block_x_acc:.4f} by={block_y_acc:.4f} "
              f"ox={offset_x_acc:.4f} oy={offset_y_acc:.4f} clk={clock_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'config': {
                    'num_blocks': NUM_BLOCKS,
                    'block_size': BLOCK_SIZE,
                    'num_clock_classes': NUM_CLOCK_CLASSES,
                    'out_emb_dim': GAT_OUT,
                    'hidden_dim': GAT_HIDDEN,
                    'gat_heads': GAT_HEADS,
                    'num_gat_layers': GAT_LAYERS,
                    'node_feat_dim': node_feat_dim,
                    'block_weight': BLOCK_WEIGHT,
                    'offset_weight': OFFSET_WEIGHT,
                    'clock_weight': CLOCK_WEIGHT,
                }
            }, MODEL_SAVE_PATH)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stop at epoch {epoch} (best val loss {best_val_loss:.4f})")
                break

    torch.save(logs, LOG_SAVE_PATH)

    final_loss, block_x_acc, block_y_acc, offset_x_acc, offset_y_acc, clock_acc = evaluate(
        model, val_loader, device,
        block_weight=BLOCK_WEIGHT,
        offset_weight=OFFSET_WEIGHT,
        clock_weight=CLOCK_WEIGHT,
    )
    print(f"Final: loss={final_loss:.4f} bx={block_x_acc:.4f} by={block_y_acc:.4f} "
          f"ox={offset_x_acc:.4f} oy={offset_y_acc:.4f} clk={clock_acc:.4f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()