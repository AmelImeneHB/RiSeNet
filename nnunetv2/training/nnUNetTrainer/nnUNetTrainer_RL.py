#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnUNet Trainer with Reinforcement Learning for Dynamic Deep Supervision
Compatible with nnUNet v2 architecture
Created by: HBAI
FIXED VERSION - Corrected gradient computation issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class DynamicWeightAgent(nn.Module):
    """Reinforcement learning agent to dynamically adjust deep supervision weights"""
    
    def __init__(self, num_levels, hidden_dim=64, dropout_rate=0.1):
        super(DynamicWeightAgent, self).__init__()
        self.num_levels = num_levels
        self.lstm = nn.LSTM(num_levels * 2, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_levels)
        
        # Alternative to softmax for better torch.compile compatibility
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, state_history):
        # state_history shape: (batch_size, sequence_length, num_levels * 2)
        lstm_out, _ = self.lstm(state_history)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out[:, -1, :])
        
        # Use temperature-scaled softmax for better stability
        scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
        
        # Manual softmax implementation that's more compile-friendly
        exp_logits = torch.exp(scaled_logits - torch.max(scaled_logits, dim=-1, keepdim=True)[0])
        weights = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
        
        return weights


class ConvergenceOptimizer:
    """Convergence optimizer using reinforcement learning"""
    
    def __init__(self, num_levels, learning_rate=1e-4, history_length=10, baseline_momentum=0.9):
        self.agent = DynamicWeightAgent(num_levels)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate)
        self.history_length = history_length
        self.state_history = []
        
        # Baseline for variance reduction
        self.baseline_momentum = baseline_momentum
        self.reward_baseline = 0.0
        
        # Storage for previous values
        self.previous_losses = None
        self.previous_dices = None
        
        # Flag to disable compile for RL agent to avoid warnings
        self.compile_agent = False

    def get_weights(self, losses, dices):
        """Get optimal weights based on performance history"""
        # Normalize inputs for stability
        losses_tensor = torch.tensor(losses, dtype=torch.float32, device=next(self.agent.parameters()).device)
        dices_tensor = torch.tensor(dices, dtype=torch.float32, device=next(self.agent.parameters()).device)
        
        # Normalize to [0, 1] range
        if len(losses) > 1:
            losses_norm = (losses_tensor - losses_tensor.min()) / (losses_tensor.max() - losses_tensor.min() + 1e-8)
            dices_norm = (dices_tensor - dices_tensor.min()) / (dices_tensor.max() - dices_tensor.min() + 1e-8)
        else:
            losses_norm = losses_tensor
            dices_norm = dices_tensor
        
        state = torch.cat([losses_norm, dices_norm]).unsqueeze(0)
        self.state_history.append(state)
        
        if len(self.state_history) > self.history_length:
            self.state_history.pop(0)

        if len(self.state_history) < 2:
            # Return uniform weights with small exploration noise at the beginning
            weights = np.ones(len(losses)) / len(losses)
            noise = np.random.normal(0, 0.01, len(losses))
            weights += noise
            weights = np.maximum(weights, 1e-6)  # Ensure positivity
            weights = weights / weights.sum()  # Renormalize
            return weights

        # Suppress softmax warnings for RL agent
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.lowering")
            with torch.no_grad():
                state_history_tensor = torch.cat(self.state_history, dim=0).unsqueeze(0)
                weights = self.agent(state_history_tensor).squeeze(0)

        return weights.detach().cpu().numpy()

    def compute_reward(self, losses, dices, previous_losses, previous_dices):
        """Compute reward based on performance improvement"""
        # Individual level improvements
        loss_improvements = [prev - curr for prev, curr in zip(previous_losses, losses)]
        dice_improvements = [curr - prev for prev, curr in zip(previous_dices, dices)]
        
        # Weighted combination (Dice is more important for segmentation)
        total_reward = sum(loss_improvements) + 2.0 * sum(dice_improvements)
        
        # Update baseline with exponential moving average
        self.reward_baseline = (self.baseline_momentum * self.reward_baseline + 
                               (1 - self.baseline_momentum) * total_reward)
        
        # Return advantage (reward - baseline)
        return total_reward - self.reward_baseline

    def update(self, losses, dices, previous_losses, previous_dices):
        """Update agent based on performance improvement with corrected REINFORCE"""
        if len(self.state_history) < 2:
            return

        # CORRECTION PRINCIPALE : Séparer complètement le forward pass pour les gradients
        device = next(self.agent.parameters()).device
        
        # Préparer les données d'entrée
        state_history_tensor = torch.cat(self.state_history, dim=0).unsqueeze(0).to(device)
        
        # Calcul du reward (scalaire Python, pas de gradients)
        reward = self.compute_reward(losses, dices, previous_losses, previous_dices)
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
        
        # Suppression des warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.lowering")
            
            # Forward pass avec gradients activés
            self.agent.train()  # S'assurer que l'agent est en mode training
            weights = self.agent(state_history_tensor).squeeze(0)
            
            # Calcul de la policy gradient loss
            log_probs = torch.log(weights + 1e-8)
            
            # REINFORCE avec baseline : -log(π(a)) * advantage
            policy_loss = -torch.sum(log_probs * reward_tensor)
            
            # Bonus d'entropie pour l'exploration
            entropy = -torch.sum(weights * torch.log(weights + 1e-8))
            entropy_bonus = 0.01 * entropy
            
            # Loss totale
            total_loss = policy_loss - entropy_bonus
            
            # Mise à jour de l'agent
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping pour la stabilité
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            self.optimizer.step()


class RLDeepSupervisionWrapper:
    """Custom wrapper to handle RL weights in deep supervision"""
    
    def __init__(self, base_loss, convergence_optimizer):
        self.base_loss = base_loss
        self.convergence_optimizer = convergence_optimizer
        
    def __call__(self, output, target):
        if not isinstance(output, (list, tuple)):
            # No deep supervision
            return self.base_loss(output, target)
        
        # Compute individual losses
        individual_losses = [self.base_loss(o, t) for o, t in zip(output, target)]
        
        # Compute individual Dice scores
        dices = []
        for o, t in zip(output, target):
            dice_score = self._compute_dice_score(o, t)
            dices.append(dice_score)
        
        # Get RL weights (with warning suppression)
        loss_values = [l.item() for l in individual_losses]
        weights = self.convergence_optimizer.get_weights(loss_values, dices)
        
        # Apply weights
        weighted_losses = [w * l for w, l in zip(weights, individual_losses)]
        total_loss = sum(weighted_losses)
        
        # CORRECTION : Mise à jour de l'optimiseur RL APRÈS le calcul de la loss principale
        # Cela évite les conflits de graphes de calcul
        if (hasattr(self.convergence_optimizer, 'previous_losses') and 
            self.convergence_optimizer.previous_losses is not None):
            # Utiliser des copies détachées pour éviter les conflits de gradients
            current_losses_detached = [l for l in loss_values]  # Déjà détachés via .item()
            current_dices_detached = [d for d in dices]  # Déjà détachés via _compute_dice_score
            
            # Programmer la mise à jour pour après le backward pass principal
            # On sauvegarde les données nécessaires
            self.convergence_optimizer._pending_update = {
                'current_losses': current_losses_detached,
                'current_dices': current_dices_detached,
                'previous_losses': self.convergence_optimizer.previous_losses.copy(),
                'previous_dices': self.convergence_optimizer.previous_dices.copy()
            }
        
        # Save for next iteration
        self.convergence_optimizer.previous_losses = loss_values
        self.convergence_optimizer.previous_dices = dices
        
        return total_loss
    
    def _compute_dice_score(self, output, target):
        """Compute Dice score for a given output level"""
        with torch.no_grad():
            # Ensure target is properly formatted
            if isinstance(target, list):
                # If target is a list (multiple scales), use the first one
                target = target[0] if len(target) > 0 else target
            
            # Convert to probabilities and predictions
            if output.shape[1] > 1:  # Multi-class
                # Use manual softmax to avoid warnings
                exp_output = torch.exp(output - torch.max(output, dim=1, keepdim=True)[0])
                output_probs = exp_output / torch.sum(exp_output, dim=1, keepdim=True)
                pred = output_probs.argmax(dim=1).float()
            else:  # Binary
                output_probs = torch.sigmoid(output)
                pred = (output_probs > 0.5).float().squeeze(1)  # Remove channel dimension for binary
            
            # Ensure target is in the right format
            if target.ndim > pred.ndim:
                target = target.squeeze(1)  # Remove channel dimension if present
            target = target.float()
            
            # Handle case where target might have different shape
            if pred.shape != target.shape:
                # Resize target to match prediction if needed
                if pred.ndim == 3 and target.ndim == 3:  # Both are 3D
                    target = F.interpolate(target.unsqueeze(0).unsqueeze(0), 
                                         size=pred.shape[-3:], 
                                         mode='nearest').squeeze(0).squeeze(0)
                elif pred.ndim == 4 and target.ndim == 4:  # Both are 4D batches
                    target = F.interpolate(target.unsqueeze(1), 
                                         size=pred.shape[-3:], 
                                         mode='nearest').squeeze(1)
            
            # Flatten for dice calculation
            pred_flat = pred.view(pred.shape[0], -1)  # Batch x Voxels
            target_flat = target.view(target.shape[0], -1)  # Batch x Voxels
            
            # Compute Dice coefficient per batch element
            intersection = torch.sum(pred_flat * target_flat, dim=1)
            union = torch.sum(pred_flat, dim=1) + torch.sum(target_flat, dim=1)
            
            # Avoid division by zero
            dice_per_sample = torch.where(union > 0, 
                                        (2.0 * intersection) / union,
                                        torch.tensor(1.0, device=output.device))  # Perfect score when both are empty
            
            # Return mean dice as a regular Python float
            mean_dice = dice_per_sample.mean().cpu().item()
            
            # Ensure we return a regular float, not np.float32
            return float(mean_dice)


class nnUNetTrainer_RL(nnUNetTrainer):
    """
    nnUNet Trainer with reinforcement learning optimization of deep supervision weights
    Compatible with nnUNet v2
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # RL variables - will be initialized later
        self.convergence_optimizer = None
        self.rl_wrapper = None
        self.original_loss = None
        
        # Initialize RL logging keys in the logger
        self._initialize_rl_logging()
        
        self.print_to_log_file("\n" + "="*70 + "\n"
                              "nnU-Net with Reinforcement Learning for Dynamic Deep Supervision\n"
                              "Dynamic weight optimization based on performance feedback\n"
                              "WARNING SUPPRESSION: Softmax compilation warnings disabled for RL agent\n"
                              + "="*70 + "\n",
                              also_print_to_console=True, add_timestamp=False)

    def _initialize_rl_logging(self):
        """Initialize RL-specific logging keys in the nnUNet logger"""
        # Add RL-specific keys to the logger's dictionary
        rl_keys = [
            'rl_weights',
            'reward_baseline', 
            'individual_losses_mean',
            'individual_dices_mean'
        ]
        
        for key in rl_keys:
            if key not in self.logger.my_fantastic_logging:
                self.logger.my_fantastic_logging[key] = []

    def _build_loss(self):
        """Build loss function with RL integration"""
        # Build base loss as in parent
        if self.label_manager.has_regions:
            from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
                                 {}, weight_ce=1, weight_dice=1,
                                 ignore_label=self.label_manager.ignore_label, 
                                 dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # Initialize RL optimizer if deep supervision is enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            if deep_supervision_scales is not None:
                num_levels = len(deep_supervision_scales)
                self.convergence_optimizer = ConvergenceOptimizer(num_levels=num_levels)
                
                # Create RL wrapper instead of standard wrapper
                self.rl_wrapper = RLDeepSupervisionWrapper(loss, self.convergence_optimizer)
                self.print_to_log_file(f"RL optimizer initialized with {num_levels} deep supervision levels")
                self.print_to_log_file("RL agent will NOT be compiled to avoid softmax warnings")
                
                return self.rl_wrapper
            else:
                self.print_to_log_file("Deep supervision disabled, no RL optimizer")
        
        # If no deep supervision, use standard logic
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def train_step(self, batch: dict) -> dict:
        """Training step with enhanced RL logging and warning suppression"""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        from nnunetv2.utilities.helpers import dummy_context
        from torch import autocast
        
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # Suppress warnings during loss computation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.lowering")
                l = self.loss(output, target)

        # Backward pass pour le réseau principal
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        # CORRECTION : Mise à jour de l'agent RL APRÈS le backward pass principal
        if (self.convergence_optimizer is not None and 
            hasattr(self.convergence_optimizer, '_pending_update')):
            pending = self.convergence_optimizer._pending_update
            self.convergence_optimizer.update(
                pending['current_losses'],
                pending['current_dices'],
                pending['previous_losses'],
                pending['previous_dices']
            )
            # Nettoyer les données en attente
            delattr(self.convergence_optimizer, '_pending_update')

        # Prepare result with RL information
        result = {'loss': l.detach().cpu().numpy()}
        
        # Add RL information if available
        if self.convergence_optimizer is not None and hasattr(self.convergence_optimizer, 'previous_losses'):
            if self.convergence_optimizer.previous_losses is not None:
                result.update({
                    'dice': np.mean(self.convergence_optimizer.previous_dices) if self.convergence_optimizer.previous_dices else 0.0,
                    'individual_losses': self.convergence_optimizer.previous_losses,
                    'individual_dices': self.convergence_optimizer.previous_dices
                })
                
                # Compute current weights for logging (with warning suppression)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.lowering")
                    weights = self.convergence_optimizer.get_weights(
                        self.convergence_optimizer.previous_losses,
                        self.convergence_optimizer.previous_dices
                    )
                result['rl_weights'] = weights.tolist()
                result['reward_baseline'] = self.convergence_optimizer.reward_baseline

        return result

    def on_train_epoch_end(self, train_outputs: List[dict]):
        """End of training epoch with RL logging"""
        super().on_train_epoch_end(train_outputs)
        
        if self.convergence_optimizer is not None and train_outputs:
            # Log RL information if available - use proper logging methods
            if 'rl_weights' in train_outputs[-1]:
                weights = train_outputs[-1]['rl_weights']
                # Log mean of weights instead of the full array to avoid logger issues
                mean_weight = np.mean(weights) if weights else 0.0
                self.logger.log('rl_weights', mean_weight, self.current_epoch)
                self.print_to_log_file(f'RL weights: {[f"{w:.3f}" for w in weights]}')
            
            # Log individual losses mean
            if 'individual_losses' in train_outputs[-1]:
                losses = train_outputs[-1]['individual_losses']
                if losses:
                    mean_loss = np.mean(losses)
                    self.logger.log('individual_losses_mean', mean_loss, self.current_epoch)
                self.print_to_log_file(f'Individual losses: {[f"{l:.4f}" for l in losses]}')
            
            # Log individual dice scores mean
            if 'individual_dices' in train_outputs[-1]:
                dices = train_outputs[-1]['individual_dices']
                if dices:
                    mean_dice = np.mean(dices)
                    self.logger.log('individual_dices_mean', mean_dice, self.current_epoch)
                self.print_to_log_file(f'Individual dices: {[f"{d:.4f}" for d in dices]}')
                
            # Log baseline for monitoring
            if 'reward_baseline' in train_outputs[-1]:
                baseline = train_outputs[-1]['reward_baseline']
                self.logger.log('reward_baseline', baseline, self.current_epoch)
                self.print_to_log_file(f'Reward baseline: {baseline:.4f}')

    def save_checkpoint(self, filename: str) -> None:
        """Save checkpoint with RL agent state"""
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                # Get base checkpoint
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                    
                from torch._dynamo import OptimizedModule
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                
                # Add RL agent state if available
                if self.convergence_optimizer is not None:
                    checkpoint['convergence_optimizer_state'] = self.convergence_optimizer.agent.state_dict()
                    checkpoint['convergence_optimizer_optim'] = self.convergence_optimizer.optimizer.state_dict()
                    checkpoint['reward_baseline'] = self.convergence_optimizer.reward_baseline
                    if hasattr(self.convergence_optimizer, 'previous_losses'):
                        checkpoint['previous_losses'] = self.convergence_optimizer.previous_losses
                        checkpoint['previous_dices'] = self.convergence_optimizer.previous_dices
                
                torch.save(checkpoint, filename)
                self.print_to_log_file(f'Checkpoint saved: {filename}')
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """Load checkpoint with RL agent state"""
        super().load_checkpoint(filename_or_checkpoint)
        
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint
            
        # Load RL agent state if available
        if 'convergence_optimizer_state' in checkpoint and self.convergence_optimizer is not None:
            self.convergence_optimizer.agent.load_state_dict(checkpoint['convergence_optimizer_state'])
            if 'convergence_optimizer_optim' in checkpoint:
                self.convergence_optimizer.optimizer.load_state_dict(checkpoint['convergence_optimizer_optim'])
            if 'reward_baseline' in checkpoint:
                self.convergence_optimizer.reward_baseline = checkpoint['reward_baseline']
            if 'previous_losses' in checkpoint:
                self.convergence_optimizer.previous_losses = checkpoint['previous_losses']
            if 'previous_dices' in checkpoint:
                self.convergence_optimizer.previous_dices = checkpoint['previous_dices']
            self.print_to_log_file('RL agent state loaded from checkpoint')
