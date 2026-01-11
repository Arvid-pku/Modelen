import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import type {
  InferenceResponse,
  ModelInfo,
  InterventionConfig,
  ComparisonResponse,
} from '../types';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export function useModelInfo() {
  return useQuery<ModelInfo>({
    queryKey: ['modelInfo'],
    queryFn: async () => {
      const { data } = await api.get('/model/info');
      return data;
    },
  });
}

export function useHealthCheck() {
  return useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const { data } = await api.get('/health');
      return data;
    },
    refetchInterval: 5000,
  });
}

interface MemoryDevice {
  index: number;
  name: string;
  total_memory_gb: number;
  allocated_memory_gb: number;
  reserved_memory_gb: number;
  free_memory_gb: number;
  utilization_percent: number;
}

interface MemoryInfo {
  gpu_available: boolean;
  gpu_count: number;
  devices: MemoryDevice[];
}

export function useMemoryInfo() {
  return useQuery<MemoryInfo>({
    queryKey: ['memory'],
    queryFn: async () => {
      const { data } = await api.get('/memory');
      return data;
    },
    refetchInterval: 3000,
  });
}

export function useClearMemory() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      const { data } = await api.post('/memory/clear');
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memory'] });
    },
  });
}

interface InferenceParams {
  prompt: string;
  interventions?: InterventionConfig;
  includeLogits?: boolean;
  includeAttentions?: boolean;
  includeHiddenStates?: boolean;
  includeLogitLens?: boolean;
  topK?: number;
  maxNewTokens?: number;
  analyzeTokenIdx?: number;
}

export function useInference() {
  return useMutation<InferenceResponse, Error, InferenceParams>({
    mutationFn: async (params) => {
      const { data } = await api.post('/inference', {
        request_type: params.interventions ? 'inference_with_intervention' : 'inference',
        prompt: params.prompt,
        interventions: params.interventions,
        response_format: {
          include_logits: params.includeLogits ?? true,
          include_attentions: params.includeAttentions ?? true,
          include_hidden_states: params.includeHiddenStates ?? false,
          include_logit_lens: params.includeLogitLens ?? true,
          top_k: params.topK ?? 10,
          analyze_token_idx: params.analyzeTokenIdx ?? -1,
        },
        max_new_tokens: params.maxNewTokens ?? 1,
      });
      return data;
    },
  });
}

interface ComparisonParams {
  prompt: string;
  interventions: InterventionConfig;
  topK?: number;
}

export function useComparison() {
  return useMutation<ComparisonResponse, Error, ComparisonParams>({
    mutationFn: async (params) => {
      const { data } = await api.post('/compare', {
        prompt: params.prompt,
        interventions: params.interventions,
        response_format: {
          include_logits: true,
          include_attentions: true,
          include_hidden_states: true,
          include_logit_lens: true,
          top_k: params.topK ?? 10,
        },
      });
      return data;
    },
  });
}

interface LogitLensParams {
  prompt: string;
  tokenIdx?: number;
  topK?: number;
}

export function useLogitLens() {
  return useMutation({
    mutationFn: async (params: LogitLensParams) => {
      const { data } = await api.post('/logit_lens', null, {
        params: {
          prompt: params.prompt,
          token_idx: params.tokenIdx ?? -1,
          top_k: params.topK ?? 10,
        },
      });
      return data;
    },
  });
}

interface ExportParams {
  prompt: string;
  name?: string;
}

export function useExport() {
  return useMutation({
    mutationFn: async (params: ExportParams) => {
      const { data } = await api.post('/export', null, {
        params: {
          prompt: params.prompt,
          name: params.name ?? 'scenario',
        },
      });
      return data;
    },
  });
}

interface LoadModelParams {
  modelName: string;
  quantization?: '4bit' | '8bit' | null;
}

export function useLoadModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (params: LoadModelParams | string) => {
      const modelName = typeof params === 'string' ? params : params.modelName;
      const quantization = typeof params === 'string' ? null : params.quantization;
      const { data } = await api.post('/model/load', null, {
        params: {
          model_name: modelName,
          quantization: quantization,
        },
      });
      return data;
    },
    onSuccess: () => {
      // Invalidate model info query to refetch with new model
      queryClient.invalidateQueries({ queryKey: ['modelInfo'] });
    },
  });
}
