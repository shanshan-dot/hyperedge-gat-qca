import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F


class QCAHyperGraphBuilder:
    def __init__(self, env, num_blocks=10, block_size=6):
        """
        Initialize the hypergraph builder.

        :param env: NanoPlacementEnv instance.
        :param num_blocks: (deprecated, kept for compatibility) Theoretical number of blocks.
        :param block_size: Size of each block (default 6).
        """
        self.env = env
        self.num_blocks = num_blocks  # Kept for compatibility
        self.block_size = block_size   # Used to compute block/offset
        self.node_list = env.actions
        self.node_type_map = env.node_to_action
        self.dg = env.DG
        self.num_nodes = len(self.node_list)
        self._precompute_static()

    # ---------- Static feature construction (unchanged) ----------
    def _precompute_static(self):
        # 1. Node type one-hot encoding (4 dimensions)
        type_codes = []
        for node in self.node_list:
            t = self.node_type_map[node]
            if t == "INPUT":
                code = [1.0, 0.0, 0.0, 0.0]
            elif t == "OUTPUT":
                code = [0.0, 1.0, 0.0, 0.0]
            elif t in ["AND", "OR", "XOR"]:
                code = [0.0, 0.0, 1.0, 0.0]
            else:
                code = [0.0, 0.0, 0.0, 1.0]
            type_codes.append(code)
        self.type_codes = torch.tensor(type_codes, dtype=torch.float32)  # (num_nodes, 4)

        # 2. Single edges (directed edges from predecessor to current node)
        edges = []
        for idx, node_name in enumerate(self.node_list):
            preds = list(self.dg.predecessors(node_name))
            if len(preds) == 1 and preds[0] in self.node_list:
                edges.append((self.node_list.index(preds[0]), idx))
        if edges:
            self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)

        # 3. Hyperedge construction (fan-in and fan-out)
        he_map = defaultdict(list)
        hyperedge_id = 0

        # 3.1 Fan-in hyperedges: predecessors of multi-input gates + current node
        for idx, node_name in enumerate(self.node_list):
            preds = list(self.dg.predecessors(node_name))
            if len(preds) >= 2:
                pred_indices = [self.node_list.index(p) for p in preds if p in self.node_list]
                if len(pred_indices) >= 2:
                    members = pred_indices + [idx]
                    he_map[hyperedge_id] = members
                    hyperedge_id += 1

        # 3.2 Fan-out hyperedges: one node driving multiple successors
        for idx, node_name in enumerate(self.node_list):
            succs = list(self.dg.successors(node_name))
            if len(succs) >= 2:
                succ_indices = [self.node_list.index(s) for s in succs if s in self.node_list]
                if len(succ_indices) >= 2:
                    members = [idx] + succ_indices
                    he_map[hyperedge_id] = members
                    hyperedge_id += 1

        # Build hyperedge incidence matrix
        num_he = hyperedge_id
        he_adj = np.zeros((self.num_nodes, num_he), dtype=np.float32)
        for hid, nodes in he_map.items():
            for n in nodes:
                he_adj[n, hid] = 1.0
        self.hyperedge_adj = torch.from_numpy(he_adj)

    # ---------- Graph data assembly ----------
    def parse_qca_netlist(self, exclude_node_idx=None):
        """
        Build graph data.

        :param exclude_node_idx: Node index to exclude (forced as unplaced).
        :return: Dictionary containing 'node_features' (24-dim), 'edge_index', 'hyperedge_adj'.
        """
        dynamic_feats = self._build_dynamic_features(exclude_node_idx)  # (num_nodes, 20)
        node_feats = torch.cat([self.type_codes, dynamic_feats], dim=1)  # (num_nodes, 4+20=24)
        return {
            "node_features": node_feats,
            "edge_index": self.edge_index,
            "hyperedge_adj": self.hyperedge_adj,
        }

    # ---------- Dynamic feature construction (clock uses sin/cos) ----------
    def _build_dynamic_features(self, exclude_node_idx=None):
        W = self.env.layout_width
        H = self.env.layout_height
        num_nodes = self.num_nodes
        # Dynamic feature dimensions: 20
        # 0: placed
        # 1-2: predecessor 1 clock sin, cos
        # 3-4: predecessor 2 clock sin, cos
        # 5-8: predecessor 1 block/offset (bx, by, ox, oy)
        # 9-12: predecessor 2 block/offset (bx, by, ox, oy)
        # 13-16: predicted current node block/offset (bx, by, ox, oy)
        # 17-18: predicted current node clock sin, cos
        # 19: global placement progress (0~1)
        feats = np.zeros((num_nodes, 20), dtype=np.float32)

        node_dict = self.env.node_dict
        layout = self.env.layout

        # Global placement progress: placed nodes / total nodes
        placed_count = sum(1 for node in self.node_list if node in node_dict)
        progress = placed_count / num_nodes if num_nodes > 0 else 0.0

        for idx, node_name in enumerate(self.node_list):
            # ---------- Placement status ----------
            if exclude_node_idx is not None and idx == exclude_node_idx:
                placed = 0.0
            else:
                placed = 1.0 if node_name in node_dict else 0.0

            # ---------- Get predecessor info (up to two) ----------
            preds = list(self.dg.predecessors(node_name))[:2]

            phases = [0, 0]  # Clock phases of the two predecessors (0~3)
            pred_coords = []  # Coordinates of placed predecessors
            pred_block_offsets = [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)]  # Predecessor block/offset

            for i, p in enumerate(preds):
                if p in node_dict:
                    p_id = node_dict[p]
                    try:
                        ptile = layout.get_tile(int(p_id))
                        phases[i] = layout.get_clock_number((ptile.x, ptile.y, 0))
                        # Record coordinates for prediction
                        pred_coords.append((ptile.x, ptile.y))
                        # Compute predecessor block/offset
                        bx = ptile.x // self.block_size
                        by = ptile.y // self.block_size
                        ox = ptile.x % self.block_size
                        oy = ptile.y % self.block_size
                        pred_block_offsets[i] = (float(bx), float(by), float(ox), float(oy))
                    except:
                        pass

            # ---------- Predict current node coordinates ----------
            # For now, use the average of predecessor coordinates as a simple prediction.
            # Replace this block with an actual prediction model if available.
            if len(pred_coords) == 0:
                predicted_x = W // 2
                predicted_y = H // 2
            elif len(pred_coords) == 1:
                predicted_x, predicted_y = pred_coords[0]
            else:
                x1, y1 = pred_coords[0]
                x2, y2 = pred_coords[1]
                predicted_x = (x1 + x2) // 2
                predicted_y = (y1 + y2) // 2

            predicted_x = min(max(predicted_x, 0), W - 1)
            predicted_y = min(max(predicted_y, 0), H - 1)

            # Predicted current node block/offset
            pred_block_x = predicted_x // self.block_size
            pred_block_y = predicted_y // self.block_size
            pred_offset_x = predicted_x % self.block_size
            pred_offset_y = predicted_y % self.block_size

            # Predicted current node clock phase: use first predecessor's phase (or 0 if none)
            predicted_phase = phases[0] if len(pred_coords) > 0 else 0

            # Compute sin/cos encoding for clock
            pred1_sin = np.sin(np.pi * phases[0] / 2.0)
            pred1_cos = np.cos(np.pi * phases[0] / 2.0)
            pred2_sin = np.sin(np.pi * phases[1] / 2.0)
            pred2_cos = np.cos(np.pi * phases[1] / 2.0)
            pred_sin = np.sin(np.pi * predicted_phase / 2.0)
            pred_cos = np.cos(np.pi * predicted_phase / 2.0)

            # ---------- Fill dynamic features ----------
            feats[idx, 0] = placed
            # Predecessor 1 clock sin/cos
            feats[idx, 1] = pred1_sin
            feats[idx, 2] = pred1_cos
            # Predecessor 2 clock sin/cos
            feats[idx, 3] = pred2_sin
            feats[idx, 4] = pred2_cos
            # Predecessor 1 block/offset
            feats[idx, 5] = pred_block_offsets[0][0]
            feats[idx, 6] = pred_block_offsets[0][1]
            feats[idx, 7] = pred_block_offsets[0][2]
            feats[idx, 8] = pred_block_offsets[0][3]
            # Predecessor 2 block/offset
            feats[idx, 9]  = pred_block_offsets[1][0]
            feats[idx, 10] = pred_block_offsets[1][1]
            feats[idx, 11] = pred_block_offsets[1][2]
            feats[idx, 12] = pred_block_offsets[1][3]
            # Predicted current node block/offset
            feats[idx, 13] = float(pred_block_x)
            feats[idx, 14] = float(pred_block_y)
            feats[idx, 15] = float(pred_offset_x)
            feats[idx, 16] = float(pred_offset_y)
            # Predicted current node clock sin/cos
            feats[idx, 17] = pred_sin
            feats[idx, 18] = pred_cos
            # Global placement progress
            feats[idx, 19] = progress

        return torch.from_numpy(feats)