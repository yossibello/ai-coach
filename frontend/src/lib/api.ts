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

// ─── Nutrition / Supplements ─────────────────────────────────────────────────

export interface BloodMarkerValue {
  value: number;
  unit: string;
  ref_low: number;
  ref_high: number;
  status:
    | "critical_low" | "low" | "suboptimal" | "optimal" | "high" | "critical_high" | "unknown";
  label: string;
  category: string;
  _confidence?: number;
}

export interface BloodTest {
  id: string;
  test_date: string;
  lab_name: string | null;
  source: "pdf" | "manual" | "api";
  parser_version: string | null;
  parser_confidence: number | null;
  markers: Record<string, BloodMarkerValue>;
  notes: string | null;
}

export interface MarkerTimeSeries {
  marker_key: string;
  label: string;
  unit: string;
  points: Array<{
    test_date: string;
    value: number;
    status: string;
    ref_low: number;
    ref_high: number;
  }>;
}

export interface SupplementItem {
  supplement_key: string;
  label: string;
  category: string;
  evidence_grade: "A" | "B" | "C" | "D";
  dose: number;
  dose_unit: string;
  frequency: string;
  timing: string;
  duration: string;
  rationale: string;
  citations: string[];
  warnings: string[];
  contraindications: string[];
  score: number;
  triggered_by: string[];
}

export interface SupplementWarning {
  warning_key: string;
  applies_to: string[];
  message: string;
  citations: string[];
}

export interface SupplementStackPayload {
  stack: SupplementItem[];
  depletion_signals: Record<string, number>;
  warnings: SupplementWarning[];
  based_on_blood_test_id: string | null;
  has_blood_test: boolean;
  engine_version: string;
  is_cold_start: boolean;
  disclaimer: string;
}

export interface SupplementStack {
  id: string;
  generated_at: string;
  engine_version: string;
  is_cold_start: boolean;
  based_on_blood_test_id: string | null;
  payload: SupplementStackPayload;
}

export const nutritionAPI = {
  uploadBloodTestPDF: (file: File, notes?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (notes) form.append("notes", notes);
    return api.post<BloodTest>("/api/v1/nutrition/blood-tests", form, {
      headers: { "Content-Type": "multipart/form-data" },
    }).then((r) => r.data);
  },

  uploadBloodTestManual: (body: {
    test_date: string;
    lab_name?: string;
    notes?: string;
    markers: Array<{ marker_key: string; value: number; unit?: string }>;
  }) =>
    api.post<BloodTest>("/api/v1/nutrition/blood-tests/manual", body).then((r) => r.data),

  listBloodTests: () =>
    api.get<BloodTest[]>("/api/v1/nutrition/blood-tests").then((r) => r.data),

  getBloodTest: (id: string) =>
    api.get<BloodTest>(`/api/v1/nutrition/blood-tests/${id}`).then((r) => r.data),

  deleteBloodTest: (id: string) =>
    api.delete(`/api/v1/nutrition/blood-tests/${id}`).then((r) => r.data),

  getMarkerSeries: (key: string) =>
    api.get<MarkerTimeSeries>(`/api/v1/nutrition/markers/${key}`).then((r) => r.data),

  getSupplements: () =>
    api.get<SupplementStack>("/api/v1/nutrition/supplements").then((r) => r.data),

  refreshSupplements: () =>
    api.post<SupplementStack>("/api/v1/nutrition/supplements/refresh").then((r) => r.data),
};

// ─── Tracking (intake + performance tests) ───────────────────────────────────
export interface SupplementIntakeRecord {
  id: string;
  supplement_key: string;
  label: string;
  dose: number | null;
  dose_unit: string | null;
  frequency: string | null;
  timing: string | null;
  started_at: string;
  stopped_at: string | null;
  adherence_pct: number | null;
  source: string;
  notes: string | null;
  is_active: boolean;
}

export interface PerformanceTestRecord {
  id: string;
  test_date: string;
  test_type: string;
  value: number;
  unit: string;
  source: string;
  notes: string | null;
}

export const trackingAPI = {
  // Supplement intake
  listIntakes: (active?: boolean) =>
    api
      .get<SupplementIntakeRecord[]>("/api/v1/tracking/intake", {
        params: active === undefined ? {} : { active },
      })
      .then((r) => r.data),

  createIntake: (body: {
    supplement_key: string;
    label?: string;
    dose?: number;
    dose_unit?: string;
    frequency?: string;
    timing?: string;
    started_at?: string;
    notes?: string;
  }) =>
    api.post<SupplementIntakeRecord>("/api/v1/tracking/intake", body).then((r) => r.data),

  createIntakeFromRecommendation: (supplement_key: string, started_at?: string) =>
    api
      .post<SupplementIntakeRecord>("/api/v1/tracking/intake/from-recommendation", {
        supplement_key,
        started_at,
      })
      .then((r) => r.data),

  updateIntake: (
    id: string,
    body: {
      dose?: number;
      dose_unit?: string;
      frequency?: string;
      timing?: string;
      stopped_at?: string;
      adherence_pct?: number;
      notes?: string;
    }
  ) =>
    api
      .patch<SupplementIntakeRecord>(`/api/v1/tracking/intake/${id}`, body)
      .then((r) => r.data),

  deleteIntake: (id: string) =>
    api.delete(`/api/v1/tracking/intake/${id}`).then((r) => r.data),

  // Performance tests
  listPerformanceTests: (test_type?: string) =>
    api
      .get<PerformanceTestRecord[]>("/api/v1/tracking/performance-tests", {
        params: test_type ? { test_type } : {},
      })
      .then((r) => r.data),

  createPerformanceTest: (body: {
    test_date: string;
    test_type: string;
    value: number;
    unit: string;
    notes?: string;
  }) =>
    api
      .post<PerformanceTestRecord>("/api/v1/tracking/performance-tests", body)
      .then((r) => r.data),

  deletePerformanceTest: (id: string) =>
    api.delete(`/api/v1/tracking/performance-tests/${id}`).then((r) => r.data),
};

export default api;
