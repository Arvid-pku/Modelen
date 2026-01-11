"""
FastAPI Backend for LLM Mechanistic Interpretability Workbench

Provides endpoints for:
- Model inference with activation capture
- Interventions (layer skipping, head ablation, activation patching)
- Logit lens analysis
- Comparison between original and intervened outputs
"""

import os
import logging
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    InferenceRequest, InferenceResponse, ComparisonRequest, ComparisonResponse,
    ModelInfoResponse, HealthResponse, TokenPrediction, AttentionData,
    LogitLensLayerResult, DiffResult
)
from .model_manager import model_manager
from .hook_manager import InterventionConfig as HookInterventionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    model_name = os.environ.get("MODEL_NAME", "gpt2")
    device = os.environ.get("DEVICE", None)
    dtype = os.environ.get("DTYPE", "float16")

    try:
        model_manager.load_model(
            model_name=model_name,
            device=device,
            dtype=dtype
        )
        logger.info(f"Model {model_name} loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    yield

    # Cleanup
    if model_manager.model is not None:
        del model_manager.model
        torch.cuda.empty_cache()


app = FastAPI(
    title="LLM Interpretability Workbench",
    description="API for visualizing and intervening on LLM internal states",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded(),
        device=str(model_manager.device) if model_manager.device else "none"
    )


@app.get("/memory")
async def get_memory_info():
    """Get GPU/system memory information."""
    memory_info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": []
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = props.total_memory

            memory_info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": total / (1024**3),
                "allocated_memory_gb": allocated / (1024**3),
                "reserved_memory_gb": reserved / (1024**3),
                "free_memory_gb": (total - reserved) / (1024**3),
                "utilization_percent": (allocated / total) * 100 if total > 0 else 0,
            })

    return memory_info


@app.post("/memory/clear")
async def clear_memory():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        return {"status": "success", "message": "GPU cache cleared"}
    return {"status": "info", "message": "No GPU available"}


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    return model_manager.get_cache_stats()


@app.post("/cache/clear")
async def clear_caches():
    """Clear all caches."""
    model_manager.clear_caches()
    return {"status": "success", "message": "All caches cleared"}


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="No model loaded")

    info = model_manager.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/model/load")
async def load_model(
    model_name: str = "gpt2",
    device: Optional[str] = None,
    quantization: Optional[str] = None
):
    """
    Load a different model.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load on ('cuda', 'cpu', or None for auto)
        quantization: '4bit', '8bit', or None for no quantization
    """
    try:
        model_manager.load_model(
            model_name=model_name,
            device=device,
            quantization=quantization,
            trust_remote_code=True  # Enable for models like Qwen
        )
        return {"status": "success", "model": model_name, "quantization": quantization}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """
    Run inference with optional interventions and return activations.

    This is the main endpoint for the workbench.
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="No model loaded")

    try:
        # Tokenize input
        inputs = model_manager.tokenize(request.prompt)
        input_ids = inputs["input_ids"]
        input_tokens = model_manager.decode_tokens(input_ids)

        # Prepare interventions if specified
        hook_interventions = []
        skip_layers = []

        if request.interventions:
            skip_layers = request.interventions.skip_layers

            # Convert head ablations
            for ablation in request.interventions.ablate_heads:
                hook_interventions.append(HookInterventionConfig(
                    layer=ablation.layer,
                    component="attn_output",
                    head=ablation.head,
                    ablate=True
                ))

            # Convert activation patches
            for patch in request.interventions.activation_patching:
                hook_interventions.append(HookInterventionConfig(
                    layer=patch.layer,
                    component=patch.component,
                    token_index=patch.token_index,
                    dim_index=patch.dim_index,
                    value=patch.value
                ))

        # Run inference - use model's native outputs instead of hooks for basic capture
        with torch.no_grad():
            if hook_interventions or skip_layers:
                # Only use hooks for interventions
                with model_manager.hook_manager.intervention_context(
                    hook_interventions,
                    skip_layers=skip_layers,
                    also_capture=False
                ):
                    outputs = model_manager.model(
                        **inputs,
                        output_attentions=request.response_format.include_attentions,
                        output_hidden_states=request.response_format.include_logit_lens
                    )
            else:
                # No hooks needed - use model's native outputs
                outputs = model_manager.model(
                    **inputs,
                    output_attentions=request.response_format.include_attentions,
                    output_hidden_states=request.response_format.include_logit_lens
                )

        # Get hidden states from model output (not hooks)
        captured_hidden_states = {}
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                captured_hidden_states[layer_idx] = hidden.detach().cpu()

        # Process logits for top predictions
        top_predictions = None
        if request.response_format.include_logits:
            logits = outputs.logits  # [batch, seq, vocab]
            probs = torch.softmax(logits, dim=-1)
            top_k = request.response_format.top_k

            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            top_predictions = []

            for pos in range(logits.shape[1]):
                pos_predictions = []
                for k in range(top_k):
                    token_id = top_indices[0, pos, k].item()
                    prob = top_probs[0, pos, k].item()
                    logit = logits[0, pos, token_id].item()
                    token = model_manager.tokenizer.decode([token_id])
                    pos_predictions.append(TokenPrediction(
                        token=token,
                        probability=float(prob),
                        logit=float(logit)
                    ))
                top_predictions.append(pos_predictions)

        # Process attention maps
        attention_maps = None
        logger.info(f"include_attentions={request.response_format.include_attentions}, has_attentions={hasattr(outputs, 'attentions')}, attentions={outputs.attentions is not None if hasattr(outputs, 'attentions') else 'N/A'}")
        if request.response_format.include_attentions and hasattr(outputs, 'attentions') and outputs.attentions:
            attention_maps = []
            logger.info(f"Processing {len(outputs.attentions)} attention layers")
            for layer_idx, attn in enumerate(outputs.attentions):
                logger.info(f"Layer {layer_idx}: attn type={type(attn)}, is_none={attn is None}")
                if attn is None:
                    continue
                # attn shape: [batch, num_heads, seq, seq]
                logger.info(f"Layer {layer_idx}: attn shape={attn.shape}")
                weights = attn[0].cpu().float().numpy().tolist()
                attention_maps.append(AttentionData(
                    layer_idx=layer_idx,
                    weights=weights
                ))
            logger.info(f"Captured {len(attention_maps)} attention maps")

        # Process logit lens results
        logit_lens_results = None
        if request.response_format.include_logit_lens and captured_hidden_states:
            # Use specified token index or default to last token
            analyze_idx = request.response_format.analyze_token_idx
            lens_results = model_manager.logit_lens.analyze_all_layers(
                captured_hidden_states,
                token_idx=analyze_idx,
                top_k=request.response_format.top_k
            )
            logit_lens_results = {}
            for layer_idx, result in lens_results.items():
                logit_lens_results[layer_idx] = LogitLensLayerResult(
                    layer_idx=result.layer_idx,
                    token_idx=result.token_idx,
                    top_tokens=result.top_tokens,
                    top_probs=[float(p) for p in result.top_probs],
                    top_logits=[float(l) for l in result.top_logits],
                    entropy=float(result.entropy)
                )

        # Process hidden state norms
        hidden_state_norms = None
        if request.response_format.include_hidden_states and captured_hidden_states:
            hidden_state_norms = {}
            for layer_idx, hidden in captured_hidden_states.items():
                norms = torch.norm(hidden[0], dim=-1).tolist()
                hidden_state_norms[layer_idx] = [float(n) for n in norms]

        # Generate token(s) if requested
        generated_tokens = []
        if request.max_new_tokens > 0:
            with torch.no_grad():
                gen_outputs = model_manager.model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=False,
                    pad_token_id=model_manager.tokenizer.pad_token_id
                )
                new_tokens = gen_outputs[0, input_ids.shape[1]:]
                if new_tokens.numel() > 0:
                    generated_tokens = model_manager.decode_tokens(new_tokens)

        return InferenceResponse(
            prompt=request.prompt,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            top_predictions=top_predictions,
            attention_maps=attention_maps,
            logit_lens_results=logit_lens_results,
            hidden_state_norms=hidden_state_norms
        )

    except Exception as e:
        import traceback
        logger.error(f"Inference error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=ComparisonResponse)
async def compare_inference(request: ComparisonRequest):
    """
    Compare original inference with intervened inference.

    Returns both results plus the difference (diff-view).
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="No model loaded")

    try:
        # Run original inference
        original_request = InferenceRequest(
            prompt=request.prompt,
            response_format=request.response_format,
            max_new_tokens=1
        )
        original = await run_inference(original_request)

        # Run intervened inference
        intervened_request = InferenceRequest(
            prompt=request.prompt,
            interventions=request.interventions,
            response_format=request.response_format,
            max_new_tokens=1
        )
        intervened = await run_inference(intervened_request)

        # Calculate differences
        logit_diff = None
        if original.top_predictions and intervened.top_predictions:
            # Diff at last position
            orig_logits = {p.token: p.logit for p in original.top_predictions[-1]}
            interv_logits = {p.token: p.logit for p in intervened.top_predictions[-1]}
            all_tokens = set(orig_logits.keys()) | set(interv_logits.keys())
            logit_diff = [
                interv_logits.get(t, 0) - orig_logits.get(t, 0)
                for t in sorted(all_tokens)
            ]

        # Check if prediction changed
        orig_pred = original.generated_tokens[0] if original.generated_tokens else ""
        interv_pred = intervened.generated_tokens[0] if intervened.generated_tokens else ""

        diff = DiffResult(
            logit_diff=logit_diff,
            prediction_changed=(orig_pred != interv_pred),
            original_prediction=orig_pred,
            intervened_prediction=interv_pred
        )

        return ComparisonResponse(
            original=original,
            intervened=intervened,
            diff=diff
        )

    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/logit_lens")
async def analyze_logit_lens(prompt: str, token_idx: int = -1, top_k: int = 10):
    """
    Dedicated endpoint for logit lens analysis.

    Returns predictions at each layer for a specific token position.
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="No model loaded")

    try:
        inputs = model_manager.tokenize(prompt)

        with torch.no_grad():
            with model_manager.hook_manager.capture_context(capture_hidden_states=True):
                _ = model_manager.model(**inputs)

        captured = model_manager.hook_manager.get_captured_activations()

        # Get trajectory
        trajectory = model_manager.logit_lens.get_prediction_trajectory(
            captured.hidden_states,
            token_idx=token_idx
        )

        # Get full results
        results = model_manager.logit_lens.analyze_all_layers(
            captured.hidden_states,
            token_idx=token_idx,
            top_k=top_k
        )

        serializable_results = {
            str(k): model_manager.logit_lens.to_serializable(v)
            for k, v in results.items()
        }

        return {
            "trajectory": trajectory,
            "layer_results": serializable_results,
            "input_tokens": model_manager.decode_tokens(inputs["input_ids"])
        }

    except Exception as e:
        logger.error(f"Logit lens error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export")
async def export_scenario(
    prompt: str,
    name: str = "scenario",
    include_interventions: bool = False
):
    """
    Export a scenario as JSON for static GitHub Pages demo.

    Captures all visualization data for offline viewing.
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="No model loaded")

    try:
        # Run full inference
        request = InferenceRequest(
            prompt=prompt,
            response_format={
                "include_logits": True,
                "include_attentions": True,
                "include_hidden_states": True,
                "include_logit_lens": True,
                "top_k": 20
            },
            max_new_tokens=1
        )
        result = await run_inference(request)

        # Get logit lens trajectory
        inputs = model_manager.tokenize(prompt)
        with torch.no_grad():
            with model_manager.hook_manager.capture_context(capture_hidden_states=True):
                _ = model_manager.model(**inputs)

        captured = model_manager.hook_manager.get_captured_activations()
        trajectory = model_manager.logit_lens.get_prediction_trajectory(
            captured.hidden_states,
            token_idx=-1
        )

        export_data = {
            "name": name,
            "model": model_manager.model_name,
            "model_info": model_manager.get_model_info(),
            "prompt": prompt,
            "input_tokens": result.input_tokens,
            "generated_tokens": result.generated_tokens,
            "top_predictions": [[p.model_dump() for p in pos] for pos in result.top_predictions] if result.top_predictions else None,
            "attention_maps": [a.model_dump() for a in result.attention_maps] if result.attention_maps else None,
            "logit_lens_trajectory": trajectory,
            "logit_lens_results": {str(k): v.model_dump() for k, v in result.logit_lens_results.items()} if result.logit_lens_results else None
        }

        return export_data

    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
