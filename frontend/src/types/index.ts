/** TypeScript type definitions for the Molecular AI frontend. */

export interface DatasetInfo {
  name: string;
  description: string;
  task_type: string;
  metric: string;
  train_size: number;
  val_size: number;
  test_size: number;
}

export interface DatasetResponse {
  datasets: Record<string, DatasetInfo>;
  count: number;
}

export interface PredictRequest {
  smiles: string;
  model: string;
  dataset: string;
}

export interface PredictResponse {
  smiles: string;
  model: string;
  dataset: string;
  prediction: number;
  uncertainty: number;
  ci_lower: number;
  ci_upper: number;
  task_type: string;
  molecule_info: Record<string, number | string>;
  demo_mode?: boolean;
  message?: string;
}

export interface TrainingJob {
  job_id: string;
  model_name: string;
  dataset_name: string;
  status: string;
  message: string;
  progress: number;
  created_at?: number;
  result?: {
    history: {
      train_loss: number[];
      val_loss: number[];
      val_metric: number[];
      learning_rate: number[];
    };
    test_metrics: Record<string, number>;
    epochs_trained: number;
  };
}

export interface TrainingRequest {
  model_name: string;
  dataset_name: string;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
}

export interface BenchmarkResult {
  dataset: string;
  model: string;
  task_type: string;
  epochs_trained?: number;
  best_val_loss?: number;
  status?: string;
  metrics: Record<string, number>;
  error?: string;
}

export interface BenchmarkResponse {
  results: BenchmarkResult[];
  datasets: string[];
  models: string[];
  count: number;
}

export interface ExplainResponse {
  smiles: string;
  model: string;
  dataset: string;
  demo_mode?: boolean;
  message?: string;
  attention: {
    atom_scores: Record<number, number>;
    num_atoms?: number;
    method?: string;
    num_layers?: number;
    token_scores?: Record<number, { token: string; score: number }>;
    tokens?: string[];
    num_tokens?: number;
    error?: string;
  };
  integrated_gradients?: {
    atom_scores: Record<number, number>;
    num_atoms: number;
    method: string;
  };
}

export interface HealthResponse {
  status: string;
  device: string;
  cuda_available: boolean;
  cuda_devices: number;
  version: string;
}

export interface ApiInfo {
  name: string;
  version: string;
  models: string[];
  datasets: string[];
  endpoints: Record<string, string>;
}

export interface NavItem {
  label: string;
  path: string;
  icon: string;
}
