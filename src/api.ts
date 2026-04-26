/**
 * API Client for YOLO Vision Dashboard Backend
 * Handles all communication with the FastAPI backend.
 */

const API_BASE = '/api';

// ============================================
// Token Management
// ============================================
let authToken: string | null = sessionStorage.getItem('yolo_token');

function setToken(token: string | null) {
  authToken = token;
  if (token) {
    sessionStorage.setItem('yolo_token', token);
  } else {
    sessionStorage.removeItem('yolo_token');
  }
}

function getToken(): string | null {
  return authToken;
}

// ============================================
// HTTP Helper
// ============================================
async function request<T = any>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> || {}),
  };

  // Add auth token if available
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }

  // Add Content-Type for JSON bodies (but not for FormData)
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorDetail = response.statusText;
    try {
      const errorBody = await response.json();
      errorDetail = errorBody.detail || errorBody.message || errorDetail;
    } catch {}
    throw new Error(errorDetail);
  }

  return response.json();
}

// ============================================
// Auth API
// ============================================
export const authApi = {
  async login(username: string, password: string) {
    const result = await request<{
      success: boolean;
      message: string;
      token: string | null;
      username: string | null;
      user_id: string | null;
    }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });

    if (result.token) {
      setToken(result.token);
    }
    return result;
  },

  async register(username: string, password: string) {
    return request<{
      success: boolean;
      message: string;
      username: string | null;
      user_id: string | null;
    }>('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
  },

  async logout() {
    try {
      await request('/auth/logout', { method: 'POST' });
    } finally {
      setToken(null);
    }
  },

  isLoggedIn() {
    return !!authToken;
  },

  clearToken() {
    setToken(null);
  },
};

// ============================================
// Detection Records API
// ============================================
export interface Detection {
  classId: number;
  className: string;
  score: number;
  box: [number, number, number, number];
}

export interface RecordData {
  time: string;
  image?: string;
  total_detections: number;
  fps: number;
  avg_confidence: number;
  detections: Detection[];
  video_clips?: { index: number; start_sec: number; end_sec: number; data: string }[];
}

export interface RecordResponse {
  id: string;
  user_id: string;
  time: string;
  image: string | null;
  total_detections: number;
  fps: number;
  avg_confidence: number;
  detections: Detection[];
  video_clips?: { index: number; start_sec: number; end_sec: number; data: string }[];
  created_at: string;
}

export const recordsApi = {
  async list() {
    const result = await request<{ success: boolean; records: RecordResponse[] }>('/records');
    return result.records;
  },

  async create(record: RecordData) {
    const result = await request<{ success: boolean; record: RecordResponse }>('/records', {
      method: 'POST',
      body: JSON.stringify(record),
    });
    return result.record;
  },

  async delete(id: string) {
    return request<{ success: boolean; message: string }>(`/records/${id}`, {
      method: 'DELETE',
    });
  },
};

// ============================================
// Users API
// ============================================
export interface UserProfile {
  id: string;
  username: string;
  role: string;
  created_at: string;
}

export const usersApi = {
  async list() {
    const result = await request<{ success: boolean; users: UserProfile[] }>('/users');
    return result.users;
  },

  async delete(userId: string) {
    return request<{ success: boolean; message: string }>(`/users/${userId}`, {
      method: 'DELETE',
    });
  },
};

// ============================================
// Inference API (CUDA Backend)
// ============================================
export interface InferenceResult {
  success: boolean;
  detections: Detection[];
  inference_time_ms: number;
  message: string;
}

export interface ModelStatus {
  loaded: boolean;
  model_name: string;
  provider: string;
}

export const inferenceApi = {
  async getStatus() {
    return request<ModelStatus>('/inference/status');
  },

  async listModels() {
    return request<{
      success: boolean;
      models: { name: string; size_mb: number; is_active: boolean }[];
      active_model: string;
    }>('/inference/models');
  },

  async loadModel(modelName: string) {
    return request<{
      success: boolean;
      message: string;
      model_name: string;
      provider: string;
    }>(`/inference/load/${encodeURIComponent(modelName)}`, {
      method: 'POST',
    });
  },

  async uploadModel(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    return request<{
      success: boolean;
      message: string;
      model_name: string;
      provider: string;
      input_names: string[];
    }>('/inference/upload-model', {
      method: 'POST',
      body: formData,
    });
  },

  async detect(imageBase64: string, confThreshold: number = 0.45, iouThreshold: number = 0.50) {
    return request<InferenceResult>('/inference/detect', {
      method: 'POST',
      body: JSON.stringify({
        image: imageBase64,
        conf_threshold: confThreshold,
        iou_threshold: iouThreshold,
      }),
    });
  },

  async submitVideoFrame(sessionId: string, frameBase64: string, timestampMs: number, width: number, height: number) {
    return request<{ success: boolean; buffered: number }>('/inference/video-frame', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        frame: frameBase64,
        timestamp_ms: timestampMs,
        width,
        height,
      }),
    });
  },

  async finalizeVideo(sessionId: string, fps: number = 1, clipDurationSec: number = 10) {
    return request<{
      success: boolean;
      clips: { index: number; start_sec: number; end_sec: number; data: string }[];
      total_frames: number;
    }>('/inference/video-finalize', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        fps,
        clip_duration_sec: clipDurationSec,
      }),
    });
  },
};
