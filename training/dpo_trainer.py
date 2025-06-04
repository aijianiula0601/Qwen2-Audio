import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DPOTrainer(Trainer):
    """
    Direct Preference Optimization (DPO) Trainer for Qwen2-Audio
    """
    
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset=None,
        tokenizer=None,
        beta: float = 0.1,
        reference_model=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs
        )
        self.beta = beta
        self.reference_model = reference_model
        
        # Create reference model if not provided
        if self.reference_model is None:
            self.reference_model = self._create_reference_model()
    
    def _create_reference_model(self):
        """Create reference model for DPO"""
        # Clone the model for reference
        reference_model = type(self.model)(self.model.config)
        reference_model.load_state_dict(self.model.state_dict())
        
        # Freeze reference model
        for param in reference_model.parameters():
            param.requires_grad = False
        
        reference_model.eval()
        return reference_model
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute DPO loss
        """
        # Extract chosen and rejected inputs
        chosen_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'audio_values': inputs['audio_values'],
            'labels': inputs['labels']
        }
        
        rejected_inputs = {
            'input_ids': inputs['rejected_input_ids'],
            'attention_mask': inputs['rejected_attention_mask'],
            'audio_values': inputs['audio_values'],  # Same audio for both
            'labels': inputs['rejected_labels']
        }
        
        # Forward pass through model
        chosen_outputs = model(**chosen_inputs)
        rejected_outputs = model(**rejected_inputs)
        
        # Forward pass through reference model
        with torch.no_grad():
            ref_chosen_outputs = self.reference_model(**chosen_inputs)
            ref_rejected_outputs = self.reference_model(**rejected_inputs)
        
        # Compute log probabilities
        chosen_logps = self._get_log_probs(chosen_outputs.logits, chosen_inputs['labels'])
        rejected_logps = self._get_log_probs(rejected_outputs.logits, rejected_inputs['labels'])
        
        ref_chosen_logps = self._get_log_probs(ref_chosen_outputs.logits, chosen_inputs['labels'])
        ref_rejected_logps = self._get_log_probs(ref_rejected_outputs.logits, rejected_inputs['labels'])
        
        # Compute DPO loss
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits).mean()
        
        # Metrics for logging
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()
        
        metrics = {
            'dpo_loss': loss.detach(),
            'chosen_rewards': chosen_rewards.mean(),
            'rejected_rewards': rejected_rewards.mean(),
            'reward_accuracy': (chosen_rewards > rejected_rewards).float().mean(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean()
        }
        
        # Add metrics to logs
        if hasattr(self, '_log_metrics'):
            self._log_metrics.update(metrics)
        else:
            self._log_metrics = metrics
        
        if return_outputs:
            return loss, {'chosen_outputs': chosen_outputs, 'rejected_outputs': rejected_outputs}
        
        return loss
    
    def _get_log_probs(self, logits, labels):
        """
        Compute log probabilities for given logits and labels
        """
        # Shift labels and logits for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probabilities for actual tokens
        gathered_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out padding tokens
        mask = (shift_labels != -100).float()
        masked_log_probs = gathered_log_probs * mask
        
        # Sum over sequence length
        return masked_log_probs.sum(dim=-1)
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log metrics including DPO-specific ones
        """
        if hasattr(self, '_log_metrics'):
            logs.update(self._log_metrics)
            # Reset metrics
            self._log_metrics = {}
        
        super().log(logs)
    
    def prediction_step(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None
    ):
        """
        Prediction step for evaluation
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Extract logits for metrics
        chosen_logits = outputs['chosen_outputs'].logits
        rejected_logits = outputs['rejected_outputs'].logits
        
        return (loss, (chosen_logits, rejected_logits), None)


def create_dpo_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    training_args=None,
    beta: float = 0.1,
    **kwargs
) -> DPOTrainer:
    """
    Create DPO trainer with given parameters
    """
    return DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=beta,
        **kwargs
    ) 