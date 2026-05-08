import axios, { AxiosError } from "axios";
import type {
  Activity,
  AthleteProfile,
  CoachRecommendation,
  FitnessProgression,
  PaginatedResponse,
  UploadResult,
} from "@/types";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  headers: { "Content-Type": "application/json" },
});

// Attach JWT from localStorage on every request
api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("access_token");
    if (token) config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Auto-refresh on 401
api.interceptors.response.use(
  (res) => res,
  async (err: AxiosError) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("access_token");
      window.location.href = "/login";
    }
    return Promise.reject(err);
  }
);

// ─── Auth ────────────────────────────────────────────────────────────────────

export const authAPI = {
  signup: (email: string, password: string, name: string) =>
    api.post("/api/v1/auth/signup", { email, password, name }),

  login: async (email: string, password: string) => {
    const { data } = await api.post<{ access_token: string; token_type: string }>(
      "/api/v1/auth/login",
      new URLSearchParams({ username: email, password }),
      { headers: { "Content-Type": "application/x-www-form-urlencoded" } }
    );
    localStorage.setItem("access_token", data.access_token);
    return data;
  },

  logout: () => {
    localStorage.removeItem("access_token");
  },

  me: () => api.get("/api/v1/auth/me"),
};

// ─── Strava ──────────────────────────────────────────────────────────────────

export const stravaAPI = {
  getAuthURL: () =>
    api.get<{ url: string }>("/api/v1/strava/auth-url").then((r) => r.data.url),

  exchangeCode: (code: string) =>
    api.post("/api/v1/strava/exchange", { code }),

  syncHistory: () =>
    api.post<{ task_id: string }>("/api/v1/strava/sync-history").then((r) => r.data),

  getSyncStatus: (taskId: string) =>
    api.get<{ status: string; progress: number; total: number }>(
      `/api/v1/strava/sync-status/${taskId}`
    ).then((r) => r.data),

  disconnect: () => api.delete("/api/v1/strava/disconnect"),
};

// ─── Activities ──────────────────────────────────────────────────────────────

export const activitiesAPI = {
  list: (page = 1, size = 20) =>
    api.get<PaginatedResponse<Activity>>("/api/v1/activities", {
      params: { page, size },
    }).then((r) => r.data),

  get: (id: string) =>
    api.get<Activity>(`/api/v1/activities/${id}`).then((r) => r.data),

  uploadFile: (file: File, onProgress?: (pct: number) => void) => {
    const form = new FormData();
    form.append("file", file);
    return api.post<UploadResult>("/api/v1/activities/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (e) => {
        if (onProgress && e.total) onProgress(Math.round((e.loaded * 100) / e.total));
      },
    }).then((r) => r.data);
  },

  delete: (id: string) => api.delete(`/api/v1/activities/${id}`),
};

// ─── Fitness ─────────────────────────────────────────────────────────────────

export const fitnessAPI = {
  getProgression: (weeks = 52) =>
    api.get<FitnessProgression>("/api/v1/fitness/progression", {
      params: { weeks },
    }).then((r) => r.data),

  recalculate: () =>
    api.post<{ task_id: string }>("/api/v1/fitness/recalculate").then((r) => r.data),
};

// ─── AI Coach ────────────────────────────────────────────────────────────────

export const coachAPI = {
  getRecommendation: () =>
    api.get<CoachRecommendation>("/api/v1/coach/recommendation").then((r) => r.data),

  refreshRecommendation: () =>
    api.post<CoachRecommendation>("/api/v1/coach/recommendation/refresh").then((r) => r.data),

  analyzeActivity: (activityId: string) =>
    api.get<{ analysis: string; insights: string[] }>(
      `/api/v1/coach/analyze/${activityId}`
    ).then((r) => r.data),
};

// ─── Profile ─────────────────────────────────────────────────────────────────

export const profileAPI = {
  get: () =>
    api.get<AthleteProfile>("/api/v1/profile").then((r) => r.data),

  update: (data: Partial<AthleteProfile>) =>
    api.put<AthleteProfile>("/api/v1/profile", data).then((r) => r.data),
};

export default api;
