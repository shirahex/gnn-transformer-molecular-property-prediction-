/** API client for the Molecular Property Prediction backend. */

import axios, { AxiosError } from "axios";
import type {
  DatasetResponse,
  PredictRequest,
  PredictResponse,
  TrainingJob,
  BenchmarkResponse,
  ExplainResponse,
  HealthResponse,
  ApiInfo,
} from "@/types";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Error handling interceptor
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    const message =
      (error.response?.data as { detail?: string })?.detail ||
      error.message ||
      "An unknown error occurred";
    console.error("API Error:", message);
    return Promise.reject(new Error(message));
  }
);

export const molecularApi = {
  // Health
  getHealth: (): Promise<HealthResponse> =>
    api.get("/health").then((r) => r.data),

  getInfo: (): Promise<ApiInfo> =>
    api.get("/").then((r) => r.data),

  // Datasets
  getDatasets: (): Promise<DatasetResponse> =>
    api.get("/datasets").then((r) => r.data),

  // Prediction
  predict: (request: PredictRequest): Promise<PredictResponse> =>
    api.post("/predict", request).then((r) => r.data),

  predictBatch: (requests: PredictRequest[]): Promise<PredictResponse[]> =>
    api.post("/predict/batch", requests).then((r) => r.data),

  // Training
  startTraining: (
    modelName: string,
    datasetName: string,
    epochs?: number,
    batchSize?: number,
    learningRate?: number
  ): Promise<{ job_id: string; status: string; message: string }> =>
    api
      .post(
        `/train/${modelName}/${datasetName}?epochs=${epochs || 50}&batch_size=${batchSize || 32}&learning_rate=${learningRate || 0.001}`
      )
      .then((r) => r.data),

  getTrainingStatus: (jobId: string): Promise<TrainingJob> =>
    api.get(`/train/status/${jobId}`).then((r) => r.data),

  listTrainingJobs: (): Promise<{ jobs: TrainingJob[]; count: number }> =>
    api.get("/train/jobs").then((r) => r.data),

  // Results
  getResults: (): Promise<BenchmarkResponse> =>
    api.get("/results").then((r) => r.data),

  getDatasetResults: (datasetName: string): Promise<BenchmarkResponse> =>
    api.get(`/results/${datasetName}`).then((r) => r.data),

  // Explainability
  explain: (
    smiles: string,
    model: string,
    dataset: string
  ): Promise<ExplainResponse> =>
    api
      .get(`/explain/${encodeURIComponent(smiles)}?model=${model}&dataset=${dataset}`)
      .then((r) => r.data),

  renderMolecule: (smiles: string, width?: number, height?: number): Promise<Blob> =>
    api
      .get(
        `/explain/${encodeURIComponent(smiles)}/render?width=${width || 400}&height=${height || 400}`,
        { responseType: "blob" }
      )
      .then((r) => r.data),

  getHeatmap: (
    smiles: string,
    model: string,
    dataset: string,
    width?: number,
    height?: number
  ): Promise<Blob> =>
    api
      .get(
        `/explain/${encodeURIComponent(smiles)}/heatmap?model=${model}&dataset=${dataset}&width=${width || 500}&height=${height || 500}`,
        { responseType: "blob" }
      )
      .then((r) => r.data),
};

export default api;
