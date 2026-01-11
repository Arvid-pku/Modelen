"""
API Schemas following the Data Protocol from project specification.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class HeadAblation(BaseModel):
    """Configuration for ablating a specific attention head."""
    layer: int
    head: int


class ActivationPatch(BaseModel):
    """Configuration for patching a specific activation value."""
    layer: int
    component: Literal["mlp_output", "attn_output"]
    token_index: int
    dim_index: int
    value: float


class InterventionConfig(BaseModel):
    """Configuration for all interventions."""
    skip_layers: List[int] = Field(default_factory=list)
    ablate_heads: List[HeadAblation] = Field(default_factory=list)
    activation_patching: List[ActivationPatch] = Field(default_factory=list)


class ResponseFormat(BaseModel):
    """Configuration for what to include in the response."""
    include_logits: bool = True
    include_attentions: bool = True
    include_hidden_states: bool = False
    include_logit_lens: bool = True
    top_k: int = 10
    analyze_token_idx: int = -1  # Which token to analyze (-1 = last)


class InferenceRequest(BaseModel):
    """Request for inference with optional interventions."""
    request_type: Literal["inference", "inference_with_intervention"] = "inference"
    prompt: str
    interventions: Optional[InterventionConfig] = None
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    max_new_tokens: int = 1


class TokenPrediction(BaseModel):
    """Prediction for a single token position."""
    token: str
    probability: float
    logit: float


class LogitLensLayerResult(BaseModel):
    """Logit lens result for a single layer."""
    layer_idx: int
    token_idx: int
    top_tokens: List[str]
    top_probs: List[float]
    top_logits: List[float]
    entropy: float


class AttentionData(BaseModel):
    """Attention weights for a layer."""
    layer_idx: int
    weights: List[List[List[float]]]  # [num_heads, seq_len, seq_len]


class InferenceResponse(BaseModel):
    """Response from inference endpoint."""
    prompt: str
    input_tokens: List[str]
    generated_tokens: List[str]
    top_predictions: Optional[List[List[TokenPrediction]]] = None
    attention_maps: Optional[List[AttentionData]] = None
    logit_lens_results: Optional[Dict[int, LogitLensLayerResult]] = None
    hidden_state_norms: Optional[Dict[int, List[float]]] = None


class ComparisonRequest(BaseModel):
    """Request to compare original vs intervened inference."""
    prompt: str
    interventions: InterventionConfig
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)


class DiffResult(BaseModel):
    """Difference between original and intervened results."""
    logit_diff: Optional[List[float]] = None
    attention_diff: Optional[Dict[int, List[List[List[float]]]]] = None
    prediction_changed: bool
    original_prediction: str
    intervened_prediction: str


class ComparisonResponse(BaseModel):
    """Response comparing original vs intervened inference."""
    original: InferenceResponse
    intervened: InferenceResponse
    diff: DiffResult


class ModelInfoResponse(BaseModel):
    """Information about the loaded model."""
    model_name: str
    num_layers: int
    num_heads: int
    hidden_size: int
    vocab_size: int
    device: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
