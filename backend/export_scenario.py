#!/usr/bin/env python3
"""
Export Script for GitHub Pages Static Demo

Generates JSON files with pre-computed analysis for static hosting.
Usage: python export_scenario.py --prompt "The capital of France is" --output scenarios/
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from app.model_manager import ModelManager
from app.hook_manager import HookManager


def export_scenario(
    prompt: str,
    model_name: str = "gpt2",
    output_dir: str = "scenarios",
    scenario_name: str = "default",
    device: str = None,
    top_k: int = 20
):
    """Export a complete analysis scenario as JSON."""

    print(f"Loading model: {model_name}")
    manager = ModelManager()
    manager.load_model(model_name=model_name, device=device)

    print(f"Running inference on: {prompt}")

    # Tokenize
    inputs = manager.tokenize(prompt)
    input_ids = inputs["input_ids"]
    input_tokens = manager.decode_tokens(input_ids)

    # Run with hooks
    with torch.no_grad():
        with manager.hook_manager.capture_context(
            capture_hidden_states=True,
            capture_attention=True
        ):
            outputs = manager.model(**inputs, output_attentions=True)

    captured = manager.hook_manager.get_captured_activations()

    # Process logits
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

    top_predictions = []
    for pos in range(logits.shape[1]):
        pos_predictions = []
        for k in range(top_k):
            token_id = top_indices[0, pos, k].item()
            prob = top_probs[0, pos, k].item()
            logit = logits[0, pos, token_id].item()
            token = manager.tokenizer.decode([token_id])
            pos_predictions.append({
                "token": token,
                "probability": float(prob),
                "logit": float(logit)
            })
        top_predictions.append(pos_predictions)

    # Process attention
    attention_maps = []
    if outputs.attentions:
        for layer_idx, attn in enumerate(outputs.attentions):
            weights = attn[0].cpu().float().numpy().tolist()
            attention_maps.append({
                "layer_idx": layer_idx,
                "weights": weights
            })

    # Process logit lens
    logit_lens_results = {}
    trajectory = {
        "layers": [],
        "top_predictions": [],
        "confidences": [],
        "entropies": []
    }

    if captured.hidden_states:
        lens_results = manager.logit_lens.analyze_all_layers(
            captured.hidden_states,
            token_idx=-1,
            top_k=top_k
        )

        for layer_idx, result in lens_results.items():
            logit_lens_results[str(layer_idx)] = {
                "layer_idx": result.layer_idx,
                "token_idx": result.token_idx,
                "top_tokens": result.top_tokens,
                "top_probs": [float(p) for p in result.top_probs],
                "top_logits": [float(l) for l in result.top_logits],
                "entropy": float(result.entropy)
            }
            trajectory["layers"].append(layer_idx)
            trajectory["top_predictions"].append(result.top_tokens[0])
            trajectory["confidences"].append(float(result.top_probs[0]))
            trajectory["entropies"].append(float(result.entropy))

    # Generate next token
    with torch.no_grad():
        gen_outputs = manager.model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=manager.tokenizer.pad_token_id
        )
        generated_tokens = manager.decode_tokens(gen_outputs[0, input_ids.shape[1]:])

    # Build export data
    export_data = {
        "name": scenario_name,
        "model": model_name,
        "model_info": manager.get_model_info(),
        "prompt": prompt,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "top_predictions": top_predictions,
        "attention_maps": attention_maps,
        "logit_lens_trajectory": trajectory,
        "logit_lens_results": logit_lens_results
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scenario_name}.json")

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Exported to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export scenario for static demo")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--output", type=str, default="scenarios", help="Output directory")
    parser.add_argument("--name", type=str, default="scenario", help="Scenario name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k predictions")

    args = parser.parse_args()

    export_scenario(
        prompt=args.prompt,
        model_name=args.model,
        output_dir=args.output,
        scenario_name=args.name,
        device=args.device,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
