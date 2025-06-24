#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnUNet Trainer avec Reinforcement Learning pour Deep Supervision Dynamique
Créé par: François
Basé sur le nnUNetTrainer original
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class DynamicWeightAgent(nn.Module):
    """Agent de reinforcement learning pour ajuster dynamiquement les poids de deep supervision"""
    
    def __init__(self, num_levels, hidden_dim=64):
        super(DynamicWeightAgent, self).__init__()
        self.num_levels = num_levels
        self.lstm = nn.LSTM(num_levels * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_levels)

    def forward(self, state_history):
        # state_history shape: (batch_size, sequence_length, num_levels * 2)
        lstm_out, _ = self.lstm(state_history)
        weights = F.softmax(self.fc(lstm_out[:, -1, :]), dim=-1)
        return weights


class ConvergenceOptimizer:
    """Optimiseur de convergence utilisant le reinforcement learning"""
    
    def __init__(self, num_levels, learning_rate=1e-4, history_length=10):
        self.agent = DynamicWeightAgent(num_levels)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate)
        self.history_length = history_length
        self.state_history = []

    def get_weights(self, losses, dices):
        """Obtient les poids optimaux basés sur l'historique des performances"""
        state = torch.cat([torch.tensor(losses, dtype=torch.float32), 
                          torch.tensor(dices, dtype=torch.float32)]).unsqueeze(0)
        self.state_history.append(state)
        
        if len(self.state_history) > self.history_length:
            self.state_history.pop(0)

        if len(self.state_history) < 2:
            # Retourner des poids uniformes au début
            return np.ones(len(losses)) / len(losses)

        with torch.no_grad():
            state_history_tensor = torch.cat(self.state_history, dim=0).unsqueeze(0)
            weights = self.agent(state_history_tensor).squeeze(0)

        return weights.detach().numpy()

    def update(self, losses, dices, previous_losses, previous_dices):
        """Met à jour l'agent basé sur l'amélioration des performances"""
        if len(self.state_history) < 2:
            return

        state_history_tensor = torch.cat(self.state_history, dim=0).unsqueeze(0)
        weights = self.agent(state_history_tensor).squeeze(0)

        # Calculer la récompense basée sur l'amélioration
        loss_improvement = sum(p - c for p, c in zip(previous_losses, losses))
        dice_improvement = sum(c - p for p, c in zip(previous_dices, dices))
        reward = loss_improvement + dice_improvement

        # Calculer la perte pour l'agent (REINFORCE)
        log_probs = torch.log(weights + 1e-8)  # Éviter log(0)
        loss = -torch.sum(log_probs * reward)

        # Mettre à jour l'agent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class nnUNetTrainer_RL(nnUNetTrainer):
    """
    nnUNet Trainer avec optimisation par reinforcement learning des poids de deep supervision
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # Variables pour RL - seront initialisées plus tard
        self.convergence_optimizer = None
        self.previous_losses = None
        self.previous_dices = None
        
        self.print_to_log_file("\n" + "="*70 + "\n"
                              "nnU-Net avec Reinforcement Learning pour Deep Supervision\n"
                              "Optimisation dynamique des poids basée sur les performances\n"
                              + "="*70 + "\n",
                              also_print_to_console=True, add_timestamp=False)

    def initialize(self):
        """Initialise le trainer et l'optimiseur RL"""
        super().initialize()
        
        # Initialiser l'optimiseur de convergence après que le réseau soit construit
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            if deep_supervision_scales is not None:
                num_levels = len(deep_supervision_scales)
                self.convergence_optimizer = ConvergenceOptimizer(num_levels=num_levels)
                self.print_to_log_file(f"Optimiseur RL initialisé avec {num_levels} niveaux de deep supervision")

    def compute_dice_score(self, output, target):
        """Calcule le score Dice pour un niveau de sortie donné"""
        with torch.no_grad():
            if self.label_manager.has_regions:
                output = (torch.sigmoid(output) > 0.5).float()
            else:
                output = output.argmax(dim=1, keepdim=True).float()
                target = target.float()
            
            # Calcul Dice simple
            axes = [0] + list(range(2, output.ndim))
            intersection = torch.sum(output * target, dim=axes)
            union = torch.sum(output, dim=axes) + torch.sum(target, dim=axes)
            dice = (2.0 * intersection) / (union + 1e-8)
            
        return dice.mean().item()

    def train_step(self, batch: dict) -> dict:
        """Étape d'entraînement avec RL pour ajuster les poids de deep supervision"""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass avec autocast si CUDA
        from nnunetv2.utilities.helpers import dummy_context
        from torch import autocast
        
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            
            # Si pas de deep supervision ou pas d'optimiseur RL, utiliser la perte standard
            if not self.enable_deep_supervision or self.convergence_optimizer is None:
                l = self.loss(output, target)
            else:
                # Traitement avec RL pour deep supervision
                if not isinstance(output, list):
                    output = [output]
                if not isinstance(target, list):
                    target = [target] * len(output)

                # Calculer les pertes individuelles pour chaque niveau
                individual_losses = []
                dices = []
                
                # Utiliser la fonction de perte de base sans wrapper DeepSupervision
                base_loss_fn = self._get_base_loss_function()
                
                for o, t in zip(output, target):
                    loss_val = base_loss_fn(o, t)
                    dice_val = self.compute_dice_score(o, t)
                    
                    individual_losses.append(loss_val)
                    dices.append(dice_val)

                # Obtenir les poids de l'agent RL
                loss_values = [l.item() for l in individual_losses]
                weights = self.convergence_optimizer.get_weights(loss_values, dices)

                # Appliquer les poids aux pertes
                weighted_losses = [w * l for w, l in zip(weights, individual_losses)]
                l = sum(weighted_losses)

                # Mettre à jour l'agent RL
                if self.previous_losses is not None and self.previous_dices is not None:
                    self.convergence_optimizer.update(
                        loss_values, dices, 
                        self.previous_losses, self.previous_dices
                    )

                # Sauvegarder pour la prochaine itération
                self.previous_losses = loss_values
                self.previous_dices = dices

        # Backward pass
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

        # Préparer le résultat
        result = {'loss': l.detach().cpu().numpy()}
        
        # Ajouter les informations RL si disponibles
        if self.convergence_optimizer is not None and hasattr(self, 'previous_losses'):
            result.update({
                'dice': np.mean(self.previous_dices) if self.previous_dices else 0.0,
                'rl_weights': weights.tolist() if 'weights' in locals() else None,
                'individual_losses': self.previous_losses,
                'individual_dices': self.previous_dices
            })

        return result

    def _get_base_loss_function(self):
        """Retourne la fonction de perte de base sans wrapper DeepSupervision"""
        if self.label_manager.has_regions:
            from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
            base_loss = DC_and_BCE_loss({}, 
                                      {'batch_dice': self.configuration_manager.batch_dice,
                                       'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                      use_ignore_label=self.label_manager.ignore_label is not None,
                                      dice_class=MemoryEfficientSoftDiceLoss)
        else:
            from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
            base_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                      'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
                                     {}, weight_ce=1, weight_dice=1,
                                     ignore_label=self.label_manager.ignore_label, 
                                     dice_class=MemoryEfficientSoftDiceLoss)
        return base_loss

    def on_train_epoch_end(self, train_outputs: List[dict]):
        """Fin d'époque d'entraînement avec logging RL"""
        super().on_train_epoch_end(train_outputs)
        
        if self.convergence_optimizer is not None and train_outputs:
            # Logger les informations RL si disponibles
            if 'rl_weights' in train_outputs[-1] and train_outputs[-1]['rl_weights'] is not None:
                weights = train_outputs[-1]['rl_weights']
                self.logger.log('rl_weights', weights, self.current_epoch)
                self.print_to_log_file(f'Poids RL: {[f"{w:.3f}" for w in weights]}')

    def save_checkpoint(self, filename: str) -> None:
        """Sauvegarde avec état de l'agent RL"""
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                # Obtenir le checkpoint de base
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
                
                # Ajouter l'état de l'agent RL si disponible
                if self.convergence_optimizer is not None:
                    checkpoint['convergence_optimizer_state'] = self.convergence_optimizer.agent.state_dict()
                    checkpoint['convergence_optimizer_optim'] = self.convergence_optimizer.optimizer.state_dict()
                    checkpoint['previous_losses'] = self.previous_losses
                    checkpoint['previous_dices'] = self.previous_dices
                
                torch.save(checkpoint, filename)
                self.print_to_log_file(f'Checkpoint sauvegardé: {filename}')
            else:
                self.print_to_log_file('Pas de checkpoint écrit, la sauvegarde est désactivée')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """Chargement avec état de l'agent RL"""
        super().load_checkpoint(filename_or_checkpoint)
        
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        else:
            checkpoint = filename_or_checkpoint
            
        # Charger l'état de l'agent RL si disponible
        if 'convergence_optimizer_state' in checkpoint and self.convergence_optimizer is not None:
            self.convergence_optimizer.agent.load_state_dict(checkpoint['convergence_optimizer_state'])
            if 'convergence_optimizer_optim' in checkpoint:
                self.convergence_optimizer.optimizer.load_state_dict(checkpoint['convergence_optimizer_optim'])
            if 'previous_losses' in checkpoint:
                self.previous_losses = checkpoint['previous_losses']
            if 'previous_dices' in checkpoint:
                self.previous_dices = checkpoint['previous_dices']
            self.print_to_log_file('État de l\'agent RL chargé depuis le checkpoint')
