"""
Phase 1: Hook Manager Class

Handles generic input/output interception for transformer models using PyTorch hooks.
Supports capturing activations and injecting interventions without modifying model weights.
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from contextlib import contextmanager


@dataclass
class InterventionConfig:
    """Configuration for a single intervention."""
    layer: int
    component: str  # 'mlp_output', 'attn_output', 'residual'
    token_index: Optional[int] = None
    dim_index: Optional[int] = None
    value: Optional[float] = None
    head: Optional[int] = None  # For head ablation
    ablate: bool = False  # Whether to zero out


@dataclass
class CapturedActivations:
    """Container for captured model activations."""
    hidden_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    attention_weights: Dict[int, torch.Tensor] = field(default_factory=dict)
    mlp_outputs: Dict[int, torch.Tensor] = field(default_factory=dict)
    attn_outputs: Dict[int, torch.Tensor] = field(default_factory=dict)
    logits: Optional[torch.Tensor] = None


class HookManager:
    """
    Manages PyTorch forward hooks for capturing and intervening on model activations.

    Key features:
    - Capture hidden states at any layer
    - Capture attention weights
    - Inject interventions (patching, ablation)
    - Proper cleanup to prevent memory leaks
    """

    def __init__(self, model: nn.Module, model_type: str = "auto"):
        """
        Initialize the Hook Manager.

        Args:
            model: The transformer model to hook into
            model_type: Type of model ('gpt2', 'llama', 'mistral', 'auto')
        """
        self.model = model
        self.model_type = model_type
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.captured = CapturedActivations()
        self.interventions: List[InterventionConfig] = []
        self.skip_layers: List[int] = []

        # Detect model architecture
        self._detect_model_structure()

    def _detect_model_structure(self):
        """Detect the model's internal structure for proper hook placement."""
        model_name = self.model.__class__.__name__.lower()

        # GPT-2 style models
        if 'gpt2' in model_name or self.model_type == 'gpt2':
            self.layer_attr = 'transformer.h'
            self.attn_attr = 'attn'
            self.mlp_attr = 'mlp'
            self.ln_attr = 'ln_1'
        # GPT-Neo/GPT-J style
        elif 'gptneo' in model_name or 'gptj' in model_name:
            self.layer_attr = 'transformer.h'
            self.attn_attr = 'attn'
            self.mlp_attr = 'mlp'
            self.ln_attr = 'ln_1'
        # LLaMA/LLaMA-2/LLaMA-3 style
        elif 'llama' in model_name or self.model_type == 'llama':
            self.layer_attr = 'model.layers'
            self.attn_attr = 'self_attn'
            self.mlp_attr = 'mlp'
            self.ln_attr = 'input_layernorm'
        # Mistral/Mixtral style
        elif 'mistral' in model_name or 'mixtral' in model_name or self.model_type == 'mistral':
            self.layer_attr = 'model.layers'
            self.attn_attr = 'self_attn'
            self.mlp_attr = 'mlp'
            self.ln_attr = 'input_layernorm'
        # Qwen/Qwen2 style
        elif 'qwen' in model_name:
            self.layer_attr = 'model.layers'
            self.attn_attr = 'self_attn'
            self.mlp_attr = 'mlp'
            self.ln_attr = 'input_layernorm'
        # Phi/Phi-2/Phi-3 style
        elif 'phi' in model_name:
            self.layer_attr = 'model.layers'
            self.attn_attr = 'self_attn'
            self.mlp_attr = 'mlp'
            self.ln_attr = 'input_layernorm'
        # Gemma style
        elif 'gemma' in model_name:
            self.layer_attr = 'model.layers'
            self.attn_attr = 'self_attn'
            self.mlp_attr = 'mlp'
            self.ln_attr = 'input_layernorm'
        # OPT style
        elif 'opt' in model_name:
            self.layer_attr = 'model.decoder.layers'
            self.attn_attr = 'self_attn'
            self.mlp_attr = 'fc1'  # OPT uses fc1/fc2 instead of mlp
            self.ln_attr = 'self_attn_layer_norm'
        # Falcon style
        elif 'falcon' in model_name:
            self.layer_attr = 'transformer.h'
            self.attn_attr = 'self_attention'
            self.mlp_attr = 'mlp'
            self.ln_attr = 'input_layernorm'
        # BLOOM style
        elif 'bloom' in model_name:
            self.layer_attr = 'transformer.h'
            self.attn_attr = 'self_attention'
            self.mlp_attr = 'mlp'
            self.ln_attr = 'input_layernorm'
        else:
            # Try to auto-detect based on model structure
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                self.layer_attr = 'model.layers'
                self.attn_attr = 'self_attn'
                self.mlp_attr = 'mlp'
                self.ln_attr = 'input_layernorm'
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                self.layer_attr = 'transformer.h'
                self.attn_attr = 'attn'
                self.mlp_attr = 'mlp'
                self.ln_attr = 'ln_1'
            else:
                # Default fallback
                self.layer_attr = 'model.layers'
                self.attn_attr = 'self_attn'
                self.mlp_attr = 'mlp'
                self.ln_attr = 'input_layernorm'

    def _get_layers(self) -> nn.ModuleList:
        """Get the list of transformer layers."""
        attrs = self.layer_attr.split('.')
        module = self.model
        for attr in attrs:
            module = getattr(module, attr)
        return module

    def _create_capture_hook(self, layer_idx: int, capture_type: str) -> Callable:
        """Create a hook function that captures activations."""
        def hook(module: nn.Module, input: Tuple, output: Any):
            if capture_type == 'hidden_state':
                # Output is typically (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                self.captured.hidden_states[layer_idx] = hidden.detach().cpu()

            elif capture_type == 'attention':
                # Attention output is typically (attn_output, attn_weights, ...)
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        self.captured.attention_weights[layer_idx] = attn_weights.detach().cpu()
                    attn_out = output[0]
                    self.captured.attn_outputs[layer_idx] = attn_out.detach().cpu()

            elif capture_type == 'mlp':
                if isinstance(output, tuple):
                    mlp_out = output[0]
                else:
                    mlp_out = output
                self.captured.mlp_outputs[layer_idx] = mlp_out.detach().cpu()

            # Explicitly return output to ensure it's not modified
            return output

        return hook

    def _create_intervention_hook(self, intervention: InterventionConfig) -> Callable:
        """Create a hook function that modifies activations."""
        def hook(module: nn.Module, input: Tuple, output: Any):
            # Handle tuple outputs - preserve the tuple structure
            if isinstance(output, tuple):
                modified = list(output)
                tensor_to_modify = modified[0].clone()

                # Apply intervention
                if intervention.ablate:
                    if intervention.head is not None:
                        tensor_to_modify.zero_()
                    else:
                        tensor_to_modify = torch.zeros_like(tensor_to_modify)
                elif intervention.value is not None:
                    if intervention.token_index is not None and intervention.dim_index is not None:
                        tensor_to_modify[:, intervention.token_index, intervention.dim_index] = intervention.value
                    elif intervention.token_index is not None:
                        tensor_to_modify[:, intervention.token_index, :] = intervention.value

                modified[0] = tensor_to_modify
                return tuple(modified)
            else:
                # Non-tuple output
                tensor_to_modify = output.clone()
                if intervention.ablate:
                    return torch.zeros_like(tensor_to_modify)
                elif intervention.value is not None:
                    if intervention.token_index is not None and intervention.dim_index is not None:
                        tensor_to_modify[:, intervention.token_index, intervention.dim_index] = intervention.value
                    elif intervention.token_index is not None:
                        tensor_to_modify[:, intervention.token_index, :] = intervention.value
                return tensor_to_modify

        return hook

    def _create_skip_layer_hook(self, layer_idx: int) -> Callable:
        """Create a hook that skips a layer by returning the input unchanged."""
        def hook(module: nn.Module, input: Tuple, output: Any):
            # Get input hidden states
            if isinstance(input, tuple):
                input_hidden = input[0]
            else:
                input_hidden = input

            # Preserve output tuple structure but replace hidden states with input
            if isinstance(output, tuple):
                modified = list(output)
                modified[0] = input_hidden
                return tuple(modified)
            return input_hidden

        return hook

    def register_capture_hooks(
        self,
        capture_hidden_states: bool = True,
        capture_attention: bool = True,
        capture_mlp: bool = False,
        layers: Optional[List[int]] = None
    ):
        """
        Register hooks to capture activations.

        Args:
            capture_hidden_states: Whether to capture hidden states after each layer
            capture_attention: Whether to capture attention weights
            capture_mlp: Whether to capture MLP outputs
            layers: Specific layers to capture (None = all layers)
        """
        self.clear_hooks()
        self.captured = CapturedActivations()

        model_layers = self._get_layers()
        target_layers = layers if layers is not None else range(len(model_layers))

        for layer_idx in target_layers:
            if layer_idx >= len(model_layers):
                continue

            layer = model_layers[layer_idx]

            if capture_hidden_states:
                handle = layer.register_forward_hook(
                    self._create_capture_hook(layer_idx, 'hidden_state')
                )
                self.handles.append(handle)

            if capture_attention:
                attn_module = getattr(layer, self.attn_attr, None)
                if attn_module is not None:
                    handle = attn_module.register_forward_hook(
                        self._create_capture_hook(layer_idx, 'attention')
                    )
                    self.handles.append(handle)

            if capture_mlp:
                mlp_module = getattr(layer, self.mlp_attr, None)
                if mlp_module is not None:
                    handle = mlp_module.register_forward_hook(
                        self._create_capture_hook(layer_idx, 'mlp')
                    )
                    self.handles.append(handle)

    def register_intervention_hooks(
        self,
        interventions: List[InterventionConfig],
        skip_layers: Optional[List[int]] = None
    ):
        """
        Register hooks for interventions.

        Args:
            interventions: List of intervention configurations
            skip_layers: List of layer indices to skip entirely
        """
        model_layers = self._get_layers()

        # Register skip layer hooks
        if skip_layers:
            for layer_idx in skip_layers:
                if layer_idx < len(model_layers):
                    layer = model_layers[layer_idx]
                    handle = layer.register_forward_hook(
                        self._create_skip_layer_hook(layer_idx)
                    )
                    self.handles.append(handle)

        # Register intervention hooks
        for intervention in interventions:
            if intervention.layer >= len(model_layers):
                continue

            layer = model_layers[intervention.layer]

            if intervention.component == 'attn_output':
                target_module = getattr(layer, self.attn_attr, None)
            elif intervention.component == 'mlp_output':
                target_module = getattr(layer, self.mlp_attr, None)
            else:
                target_module = layer

            if target_module is not None:
                handle = target_module.register_forward_hook(
                    self._create_intervention_hook(intervention)
                )
                self.handles.append(handle)

    def clear_hooks(self):
        """Remove all registered hooks to prevent memory leaks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def get_captured_activations(self) -> CapturedActivations:
        """Return the captured activations."""
        return self.captured

    @contextmanager
    def capture_context(
        self,
        capture_hidden_states: bool = True,
        capture_attention: bool = True,
        capture_mlp: bool = False,
        layers: Optional[List[int]] = None
    ):
        """
        Context manager for safe hook registration and cleanup.

        Usage:
            with hook_manager.capture_context():
                outputs = model(inputs)
            activations = hook_manager.get_captured_activations()
        """
        try:
            self.register_capture_hooks(
                capture_hidden_states=capture_hidden_states,
                capture_attention=capture_attention,
                capture_mlp=capture_mlp,
                layers=layers
            )
            yield self
        finally:
            self.clear_hooks()

    @contextmanager
    def intervention_context(
        self,
        interventions: List[InterventionConfig],
        skip_layers: Optional[List[int]] = None,
        also_capture: bool = True
    ):
        """
        Context manager for safe intervention registration and cleanup.

        Usage:
            interventions = [InterventionConfig(layer=5, component='mlp_output', ablate=True)]
            with hook_manager.intervention_context(interventions):
                outputs = model(inputs)
        """
        try:
            if also_capture:
                self.register_capture_hooks()
            self.register_intervention_hooks(interventions, skip_layers)
            yield self
        finally:
            self.clear_hooks()
