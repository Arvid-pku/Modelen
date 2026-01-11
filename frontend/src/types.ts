export interface TokenPrediction {
  token: string;
  probability: number;
  logit: number;
}

export interface AttentionData {
  layer_idx: number;
  weights: number[][][]; // [num_heads, seq_len, seq_len]
}

export interface LogitLensLayerResult {
  layer_idx: number;
  token_idx: number;
  top_tokens: string[];
  top_probs: number[];
  top_logits: number[];
  entropy: number;
}

export interface InferenceResponse {
  prompt: string;
  input_tokens: string[];
  generated_tokens: string[];
  top_predictions: TokenPrediction[][] | null;
  attention_maps: AttentionData[] | null;
  logit_lens_results: Record<number, LogitLensLayerResult> | null;
  hidden_state_norms: Record<number, number[]> | null;
}

export interface HeadAblation {
  layer: number;
  head: number;
}

export interface ActivationPatch {
  layer: number;
  component: 'mlp_output' | 'attn_output';
  token_index: number;
  dim_index: number;
  value: number;
}

export interface InterventionConfig {
  skip_layers: number[];
  ablate_heads: HeadAblation[];
  activation_patching: ActivationPatch[];
}

export interface ModelInfo {
  model_name: string;
  num_layers: number;
  num_heads: number;
  hidden_size: number;
  vocab_size: number;
  device: string;
}

export interface ComparisonResponse {
  original: InferenceResponse;
  intervened: InferenceResponse;
  diff: {
    logit_diff: number[] | null;
    prediction_changed: boolean;
    original_prediction: string;
    intervened_prediction: string;
  };
}

export interface LogitLensTrajectory {
  layers: number[];
  top_predictions: string[];
  confidences: number[];
  entropies: number[];
  target_probs?: number[];
}
