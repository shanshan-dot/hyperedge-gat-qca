import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim=24,
        hidden_dim=128,
        out_emb_dim=128,
        dropout_rate=0.1,
        hyperedge_emb_dim=32,
        gat_heads=8,
        num_gat_layers=2,
    ):
        super().__init__()
        self.out_emb_dim = out_emb_dim
        self.edge_fusion_weight = nn.Parameter(torch.tensor(0.1))
        self.hyperedge_fusion_weight = nn.Parameter(torch.tensor(0.2))

        self.node_feat_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Edge feature projection
        self.edge_feat_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.hyperedge_proj = nn.Sequential(
            nn.Linear(hidden_dim, hyperedge_emb_dim),
            nn.LayerNorm(hyperedge_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.hyperedge_to_node = nn.Sequential(
            nn.Linear(hyperedge_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.hyperedge_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.gat_layers = nn.ModuleList()
        for _ in range(num_gat_layers):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=gat_heads,
                    concat=False,
                    dropout=dropout_rate,
                )
            )

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.graph_project = nn.Sequential(
            nn.Linear(hidden_dim, out_emb_dim),
            nn.LayerNorm(out_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def _fusion_hyperedge_features(self, x, hyperedge_adj):
        num_hyperedges = hyperedge_adj.shape[1]
        if num_hyperedges == 0:
            return x

        hyperedge_emb_list = []
        for i in range(num_hyperedges):
            mask = hyperedge_adj[:, i] > 0
            node_features = x[mask]
            if len(node_features) == 0:
                continue
            attn = self.hyperedge_attention(node_features)
            attn_w = torch.softmax(attn, dim=0)
            he_emb = (node_features * attn_w).sum(dim=0, keepdim=True)
            hyperedge_emb_list.append(he_emb)

        if not hyperedge_emb_list:
            return x

        hyperedge_emb = torch.cat(hyperedge_emb_list, dim=0)
        hyperedge_emb = self.hyperedge_proj(hyperedge_emb)
        node_hyperedge_emb = hyperedge_adj @ hyperedge_emb
        node_hyperedge_emb = self.hyperedge_to_node(node_hyperedge_emb)
        return x + self.hyperedge_fusion_weight * node_hyperedge_emb

    def _attention_pool(self, x, batch):
        attn = self.attention_pool(x)
        attn_exp = torch.exp(attn)
        max_batch = batch.max().item() + 1
        sum_exp = torch.zeros(max_batch, 1, device=x.device)
        sum_exp.scatter_add_(0, batch.unsqueeze(-1), attn_exp)
        attn_softmax = attn_exp / sum_exp[batch]
        graph_emb = torch.zeros(max_batch, x.size(1), device=x.device)
        graph_emb.scatter_add_(
            0, batch.unsqueeze(-1).expand_as(x), x * attn_softmax
        )
        return graph_emb

    def forward(self, graph_data, return_attn=False):
        x = graph_data["node_features"]
        edge_index = graph_data["edge_index"]
        hyperedge_adj = graph_data.get("hyperedge_adj")
        batch = graph_data.get("batch")
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        hyperedge_adj = (
            hyperedge_adj.to(device) if hyperedge_adj is not None else None
        )
        batch = batch.to(device)

        # 1. Node feature projection
        x = self.node_feat_proj(x)

        # 2. Hyperedge fusion
        if hyperedge_adj is not None and hyperedge_adj.shape[1] > 0:
            x = self._fusion_hyperedge_features(x, hyperedge_adj)

        # Regular edge feature fusion
        edge_attr = graph_data.get("edge_features")
        if edge_attr is not None and edge_index.shape[1] > 0:
            if edge_attr.dim() == 3:
                edge_attr = edge_attr.squeeze(0)
            edge_emb = self.edge_feat_proj(edge_attr)
            row, col = edge_index
            edge_aggr = torch.zeros_like(x)
            edge_aggr.index_add_(0, col, edge_emb)
            x = x + self.edge_fusion_weight * edge_aggr

        # GAT layers. Extract attention weights from the last layer when requested.
        attn_weights = None
        edge_idx_out = None

        for i, layer in enumerate(self.gat_layers):
            residual = x

            # Use the last layer to return attention weights.
            if return_attn and i == len(self.gat_layers) - 1:
                x, (edge_idx_out, attn) = layer(
                    x, edge_index, return_attention_weights=True
                )
                # PyG returns attn with shape [num_edges, num_heads].
                # Average over heads and move to CPU for downstream plotting.
                attn_weights = attn.mean(dim=1).detach().cpu()
            else:
                x = layer(x, edge_index)

            x = F.relu(x)
            x = residual + x

        # 4. Attention pooling for graph embeddings
        graph_emb_raw = self._attention_pool(x, batch)

        # 5. Graph projection and normalization
        graph_emb_proj = self.graph_project(graph_emb_raw)
        if graph_emb_proj.shape[0] == 1:
            graph_emb_proj = graph_emb_proj.squeeze(0)
        graph_emb_norm = F.normalize(graph_emb_proj, dim=-1)

        if return_attn:
            # Return: (graph embedding, edge index, attention weights)
            # edge_idx_out and attn_weights have already been moved to CPU.
            return graph_emb_norm, edge_idx_out.cpu(), attn_weights

        return graph_emb_norm