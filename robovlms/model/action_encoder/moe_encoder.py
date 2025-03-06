import torch
from torch import nn
import torch.nn.functional as F


class MoEActionEncoder(nn.Module):
    def __init__(self, c_dim, d_dim, num_experts=4, k=2, **kwargs):
        """
        Mixture of Experts Action Encoder
        
        Args:
            c_dim: dimension of continuous actions (arm)
            d_dim: dimension of discrete actions (gripper)
            num_experts: number of expert networks
            k: number of experts to route to for each input
            **kwargs: additional arguments, including hidden_size
        """
        super().__init__()
        self.hidden_size = kwargs.get("hidden_size")
        self.c_dim = c_dim
        self.d_dim = d_dim
        self.num_experts = num_experts
        self.k = k  # top-k experts to select
        
        # Router network to decide which experts to use
        self.router = nn.Sequential(
            nn.Linear(c_dim + d_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, num_experts)
        )
        
        # Expert networks for processing actions
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.ModuleDict({
                "arm_mlp": nn.Linear(c_dim, self.hidden_size // 2),
                "gripper_mlp": nn.Linear(d_dim, self.hidden_size // 2),
                "act_mlps": nn.ModuleList([
                    nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                    nn.Linear(self.hidden_size // 2, self.hidden_size)
                ])
            })
            self.experts.append(expert)
            
    def forward(self, action, **kwargs):
        """
        action: B x L x (c_dim + d_dim)
        
        Return:
            action_embed: B x L x h_act
            router_loss: load balancing loss for the router
        """
        batch_shape = action.shape[:-1]  # B x L
        c_action = action[..., :self.c_dim]
        d_action = action[..., self.c_dim:]
        assert c_action.shape[-1] == self.c_dim and d_action.shape[-1] == self.d_dim
        
        # Calculate routing probabilities
        router_logits = self.router(action)  # B x L x num_experts
        
        # Get routing probabilities with softmax
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # For load balancing loss calculation
        router_loss = self._compute_load_balancing_loss(routing_probs)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(routing_probs, self.k, dim=-1)
        
        # Normalize the routing weights for the selected experts
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        combined_output = torch.zeros(*batch_shape, self.hidden_size, device=action.device)
        
        # Process input with each selected expert and combine outputs
        for k_idx in range(self.k):
            # For each item in the batch, get the expert index and weight
            expert_indices = selected_experts[..., k_idx]  # B x L
            expert_weights = routing_weights[..., k_idx].unsqueeze(-1)  # B x L x 1
            
            # Process each item with its selected expert
            # This is a bit complex with batching since different items use different experts
            flat_batch_size = batch_shape.numel()
            flat_c_action = c_action.reshape(flat_batch_size, -1)
            flat_d_action = d_action.reshape(flat_batch_size, -1)
            flat_expert_indices = expert_indices.reshape(-1)
            flat_expert_weights = expert_weights.reshape(flat_batch_size, -1)
            
            # Initialize output for this k value
            flat_output = torch.zeros(flat_batch_size, self.hidden_size, device=action.device)
            
            # Process each sample with its assigned expert
            for i in range(flat_batch_size):
                expert_idx = flat_expert_indices[i].item()
                expert = self.experts[expert_idx]
                
                # Process this sample
                c_embed = expert["arm_mlp"](flat_c_action[i:i+1])
                d_embed = expert["gripper_mlp"](flat_d_action[i:i+1])
                action_embed = c_embed + d_embed
                for mlp in expert["act_mlps"]:
                    action_embed = mlp(action_embed)
                
                flat_output[i] = action_embed.squeeze(0)
            
            # Reshape output and apply weights
            output = flat_output.reshape(*batch_shape, self.hidden_size)
            combined_output += output * expert_weights
            
        return combined_output, router_loss
    
    def _compute_load_balancing_loss(self, routing_probs):
        """
        Calculates load balancing loss to ensure all experts are utilized
        
        Args:
            routing_probs: routing probabilities, shape [B, L, num_experts]
            
        Returns:
            load_balancing_loss
        """
        # Calculate mean routing probability for each expert across the batch
        router_prob_per_expert = routing_probs.mean(dim=[0, 1])  # [num_experts]
        
        # We want a uniform distribution for optimal load balancing
        target_distribution = torch.ones_like(router_prob_per_expert) / self.num_experts
        
        # Calculate KL divergence between actual and uniform distributions
        load_balancing_loss = F.kl_div(
            router_prob_per_expert.log(), target_distribution, reduction='batchmean'
        )
        
        return load_balancing_loss