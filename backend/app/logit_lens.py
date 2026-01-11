"""
Phase 2: Logit Lens Decoder

Maps hidden states at any layer directly to the vocabulary to see the model's
"current guess" at each position in the residual stream.

Reference: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class LogitLensResult:
    """Results from logit lens analysis."""
    layer_idx: int
    token_idx: int
    top_tokens: List[str]
    top_probs: List[float]
    top_logits: List[float]
    entropy: float


class LogitLensDecoder:
    """
    Decodes hidden states at any layer to vocabulary predictions.

    The logit lens allows us to see what token the model would predict
    if we "read off" the residual stream at any intermediate layer.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        model_type: str = "auto"
    ):
        """
        Initialize the Logit Lens decoder.

        Args:
            model: The transformer model
            tokenizer: The tokenizer for decoding tokens
            model_type: Type of model ('gpt2', 'llama', 'mistral', 'auto')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type

        # Get the unembedding matrix and final layer norm
        self._setup_projection_components()

    def _setup_projection_components(self):
        """Extract the components needed for logit lens projection."""
        model_name = self.model.__class__.__name__.lower()

        # Get the language model head (unembedding matrix)
        if hasattr(self.model, 'lm_head'):
            self.lm_head = self.model.lm_head
        elif hasattr(self.model, 'cls'):
            self.lm_head = self.model.cls
        elif hasattr(self.model, 'embed_out'):
            # Some models use embed_out
            self.lm_head = self.model.embed_out
        else:
            # For GPT-2 style models that tie embeddings
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                self.lm_head = lambda x: F.linear(x, self.model.transformer.wte.weight)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                self.lm_head = lambda x: F.linear(x, self.model.model.embed_tokens.weight)
            else:
                raise ValueError("Could not find language model head")

        # Get the final layer normalization
        # GPT-2 style
        if 'gpt2' in model_name or self.model_type == 'gpt2':
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'ln_f'):
                self.final_ln = self.model.transformer.ln_f
            else:
                self.final_ln = nn.Identity()
        # GPT-Neo/GPT-J style
        elif 'gptneo' in model_name or 'gptj' in model_name:
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'ln_f'):
                self.final_ln = self.model.transformer.ln_f
            else:
                self.final_ln = nn.Identity()
        # LLaMA/Mistral/Qwen/Phi/Gemma style (most modern models)
        elif any(x in model_name for x in ['llama', 'mistral', 'mixtral', 'qwen', 'phi', 'gemma']):
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
                self.final_ln = self.model.model.norm
            else:
                self.final_ln = nn.Identity()
        # OPT style
        elif 'opt' in model_name:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder') and hasattr(self.model.model.decoder, 'final_layer_norm'):
                self.final_ln = self.model.model.decoder.final_layer_norm
            else:
                self.final_ln = nn.Identity()
        # Falcon style
        elif 'falcon' in model_name:
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'ln_f'):
                self.final_ln = self.model.transformer.ln_f
            else:
                self.final_ln = nn.Identity()
        # BLOOM style
        elif 'bloom' in model_name:
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'ln_f'):
                self.final_ln = self.model.transformer.ln_f
            else:
                self.final_ln = nn.Identity()
        else:
            # Auto-detect based on common patterns
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
                self.final_ln = self.model.model.norm
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'ln_f'):
                self.final_ln = self.model.transformer.ln_f
            else:
                self.final_ln = nn.Identity()

    def decode_hidden_state(
        self,
        hidden_state: torch.Tensor,
        apply_ln: bool = True,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project a hidden state to vocabulary logits.

        Args:
            hidden_state: Tensor of shape [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
            apply_ln: Whether to apply final layer norm before projection
            top_k: Number of top tokens to return

        Returns:
            Tuple of (top_k_indices, top_k_logits)
        """
        device = next(self.model.parameters()).device

        # Ensure 3D tensor
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)

        # Move to model device if needed
        hidden_state = hidden_state.to(device)

        with torch.no_grad():
            # Apply final layer norm
            if apply_ln:
                normalized = self.final_ln(hidden_state)
            else:
                normalized = hidden_state

            # Project to vocabulary
            if callable(self.lm_head):
                logits = self.lm_head(normalized)
            else:
                logits = self.lm_head(normalized)

            # Get top-k predictions
            top_logits, top_indices = torch.topk(logits, k=top_k, dim=-1)

        return top_indices.cpu(), top_logits.cpu()

    def analyze_layer(
        self,
        hidden_state: torch.Tensor,
        layer_idx: int,
        token_indices: Optional[List[int]] = None,
        top_k: int = 10,
        apply_ln: bool = True
    ) -> List[LogitLensResult]:
        """
        Analyze hidden states at a specific layer.

        Args:
            hidden_state: Hidden state tensor [batch, seq_len, hidden_dim]
            layer_idx: Which layer this hidden state is from
            token_indices: Specific token positions to analyze (None = all)
            top_k: Number of top predictions to return
            apply_ln: Whether to apply final layer norm

        Returns:
            List of LogitLensResult for each analyzed position
        """
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)

        batch_size, seq_len, hidden_dim = hidden_state.shape

        if token_indices is None:
            token_indices = list(range(seq_len))

        results = []

        top_indices, top_logits = self.decode_hidden_state(
            hidden_state, apply_ln=apply_ln, top_k=top_k
        )

        for token_idx in token_indices:
            if token_idx >= seq_len:
                continue

            # Get predictions for this position (first batch element)
            indices = top_indices[0, token_idx].tolist()
            logits = top_logits[0, token_idx].tolist()

            # Decode tokens
            tokens = [self.tokenizer.decode([idx]) for idx in indices]

            # Calculate probabilities
            probs = F.softmax(top_logits[0, token_idx], dim=-1).tolist()

            # Calculate entropy
            full_logits = self.decode_hidden_state(
                hidden_state[:, token_idx:token_idx+1, :],
                apply_ln=apply_ln,
                top_k=hidden_state.shape[-1] if hidden_state.shape[-1] < 50257 else 50257
            )[1]
            full_probs = F.softmax(full_logits[0, 0], dim=-1)
            entropy = -torch.sum(full_probs * torch.log(full_probs + 1e-10)).item()

            results.append(LogitLensResult(
                layer_idx=layer_idx,
                token_idx=token_idx,
                top_tokens=tokens,
                top_probs=probs,
                top_logits=logits,
                entropy=entropy
            ))

        return results

    def analyze_all_layers(
        self,
        hidden_states: Dict[int, torch.Tensor],
        token_idx: int = -1,
        top_k: int = 10,
        apply_ln: bool = True
    ) -> Dict[int, LogitLensResult]:
        """
        Analyze the residual stream across all captured layers for a specific token.

        Args:
            hidden_states: Dictionary mapping layer index to hidden state tensor
            token_idx: Which token position to analyze (-1 = last token)
            top_k: Number of top predictions to return
            apply_ln: Whether to apply final layer norm

        Returns:
            Dictionary mapping layer index to LogitLensResult
        """
        results = {}

        for layer_idx, hidden_state in sorted(hidden_states.items()):
            if hidden_state.dim() == 2:
                hidden_state = hidden_state.unsqueeze(0)

            seq_len = hidden_state.shape[1]
            actual_idx = token_idx if token_idx >= 0 else seq_len + token_idx

            layer_results = self.analyze_layer(
                hidden_state,
                layer_idx=layer_idx,
                token_indices=[actual_idx],
                top_k=top_k,
                apply_ln=apply_ln
            )

            if layer_results:
                results[layer_idx] = layer_results[0]

        return results

    def get_prediction_trajectory(
        self,
        hidden_states: Dict[int, torch.Tensor],
        token_idx: int = -1,
        target_token: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Track how the model's prediction changes across layers.

        Args:
            hidden_states: Dictionary mapping layer index to hidden state tensor
            token_idx: Which token position to analyze
            target_token: Optional specific token to track probability of

        Returns:
            Dictionary with 'layers', 'top_predictions', 'confidences', and optionally 'target_probs'
        """
        layer_results = self.analyze_all_layers(hidden_states, token_idx=token_idx)

        trajectory = {
            'layers': [],
            'top_predictions': [],
            'confidences': [],
            'entropies': []
        }

        if target_token is not None:
            trajectory['target_probs'] = []
            target_id = self.tokenizer.encode(target_token, add_special_tokens=False)
            if target_id:
                target_id = target_id[0]
            else:
                target_id = None

        for layer_idx in sorted(layer_results.keys()):
            result = layer_results[layer_idx]
            trajectory['layers'].append(layer_idx)
            trajectory['top_predictions'].append(result.top_tokens[0])
            trajectory['confidences'].append(result.top_probs[0])
            trajectory['entropies'].append(result.entropy)

            if target_token is not None and target_id is not None:
                # Find target token probability if in top-k
                try:
                    idx = [self.tokenizer.encode(t, add_special_tokens=False)[0]
                           for t in result.top_tokens].index(target_id)
                    trajectory['target_probs'].append(result.top_probs[idx])
                except (ValueError, IndexError):
                    trajectory['target_probs'].append(0.0)

        return trajectory

    def to_serializable(self, result: Union[LogitLensResult, Dict]) -> Dict:
        """Convert results to JSON-serializable format."""
        if isinstance(result, LogitLensResult):
            return {
                'layer_idx': result.layer_idx,
                'token_idx': result.token_idx,
                'top_tokens': result.top_tokens,
                'top_probs': [float(p) for p in result.top_probs],
                'top_logits': [float(l) for l in result.top_logits],
                'entropy': float(result.entropy)
            }
        elif isinstance(result, dict):
            return {k: self.to_serializable(v) if isinstance(v, LogitLensResult) else v
                    for k, v in result.items()}
        return result
