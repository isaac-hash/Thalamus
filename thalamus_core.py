import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class GlobalWorkspace(nn.Module):
    def __init__(self, module_names, workspace_dim, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.module_names = module_names
        # Projection includes extra space for 'fatigue' awareness
        self.projections = nn.ModuleDict({
            name: nn.Linear(workspace_dim + 2, workspace_dim) for name in module_names
        })
        self.gating = nn.Linear(workspace_dim, 1)

    def forward(self, expert_outs, fatigue):
        scores, votes = [], []
        for i, name in enumerate(self.module_names):
            # Apply fatigue: suppress signal of the current winner
            # Higher fatigue = lower signal strength
            weakened_out = expert_outs[name] * (1.0 - fatigue[:, i:i+1])
            
            proj = self.projections[name](weakened_out)
            votes.append(proj)
            scores.append(self.gating(proj))
        
        # Softmax competition with low temperature
        weights = F.softmax(torch.stack(scores, dim=1) / self.temperature, dim=1)
        global_context = torch.sum(torch.stack(votes, dim=1) * weights, dim=1)
        return global_context, weights

class ThalamusModel(nn.Module):
    def __init__(self, input_dim, workspace_dim=64):
        super().__init__()
        self.workspace_dim = workspace_dim
        
        # 3 Specialized Brain Regions
        self.visual_module = Mamba(d_model=input_dim+workspace_dim, d_state=16, d_conv=4, expand=2)
        self.semantic_module = Mamba(d_model=input_dim+workspace_dim, d_state=64, d_conv=4, expand=2)
        self.episodic_module = Mamba(d_model=input_dim+workspace_dim, d_state=128, d_conv=4, expand=2)
        
        self.modules_dict = {'visual': self.visual_module, 'semantic': self.semantic_module, 'episodic': self.episodic_module}
        self.workspace = GlobalWorkspace(list(self.modules_dict.keys()), workspace_dim)
        self.norm = nn.LayerNorm(workspace_dim)
        self.output_head = nn.Linear(workspace_dim, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Persistent States & Fatigue
        caches = {name: (torch.zeros(batch_size, mod.d_model*mod.expand, mod.d_conv, device=device),
                         torch.zeros(batch_size, mod.d_model*mod.expand, mod.d_state, device=device)) 
                  for name, mod in self.modules_dict.items()}
        
        prev_workspace = torch.zeros(batch_size, self.workspace_dim, device=device)
        fatigue = torch.zeros(batch_size, len(self.modules_dict), device=device)
        
        outputs, gate_history = [], []

        for t in range(seq_len):
            combined = torch.cat([x[:, t, :], prev_workspace], dim=-1)
            
            expert_outs = {}
            for name, mod in self.modules_dict.items():
                c, s = caches[name]
                out_t, nc, ns = mod.step(combined.unsqueeze(1), c, s)
                caches[name] = (nc, ns)
                expert_outs[name] = out_t.squeeze(1)
            
            # Workspace decision influenced by fatigue
            global_context, weights = self.workspace(expert_outs, fatigue)
            
            # --- PERSISTENT FATIGUE LOGIC ---
            # Winners gain fatigue; all experts recover slightly over time
            fatigue = (fatigue + weights.squeeze(-1) * 0.15) * 0.85
            fatigue = torch.clamp(fatigue, 0.0, 0.6) # Cap at 60% suppression
            
            prev_workspace = self.norm(global_context)
            outputs.append(self.output_head(prev_workspace))
            gate_history.append(weights.detach().cpu().numpy())
            
        return torch.stack(outputs, dim=1), np.array(gate_history)