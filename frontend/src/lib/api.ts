import axios, { AxiosError } from "axios";
import type {
  Activity,
  AthleteProfile,
  CoachRecommendation,
  Macrocycle,
  MultiHorizonRecommendation,
  FitnessProgression,
  PaginatedResponse,
  UploadResult,
} from "@/types";

// All requests use relative paths so they go through the Next.js proxy
// (/api/v1/* → backend). Never call the backend directly from the browser —
// that breaks when accessing from a phone or any non-localhost device.
const api = axios.create({
  baseURL: "",
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
  status: () =>
    api.get<{ connected: boolean; athlete_id: string | null }>("/api/v1/strava/status").then((r) => r.data),

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

  estimateFTP: () =>
    api.post<{
      estimated_ftp: number | null;
      previous_ftp: number | null;
      updated: boolean;
      message: string;
      confidence: number;
      confidence_low: number | null;
      confidence_high: number | null;
      method: string;
      best_ride_age_days: number | null;
      last_test_age_days: number | null;
      tsb_correction: number;
      sample_count: number;
      trend: string;
    }>("/api/v1/strava/estimate-ftp").then((r) => r.data),

  rebuildPMC: () =>
    api.post("/api/v1/strava/rebuild-pmc").then((r) => r.data),
};

// ─── Activities ──────────────────────────────────────────────────────────────

export const activitiesAPI = {
  list: (params: { page?: number; limit?: number; search?: string } = {}) =>
    api.get<PaginatedResponse<Activity>>("/api/v1/activities", {
      params: { page: params.page ?? 1, size: params.limit ?? 20, search: params.search },
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

  // Multi-horizon: short / medium / event suggestions side-by-side. Always fresh.
  getMultiHorizon: () =>
    api.get<MultiHorizonRecommendation>("/api/v1/coach/recommendation/multi-horizon")
      .then((r) => r.data),

  // Macrocycle: week-by-week reverse-periodization plan from today to event date.
  // Pass days to get a horizon-scoped view (28 for medium, 7 for short, omit for event).
  getMacrocycle: (days?: number) =>
    api.get<Macrocycle>("/api/v1/coach/macrocycle", { params: days != null ? { days } : undefined }).then((r) => r.data),

  analyzeActivity: (activityId: string) =>
    api.get<{ analysis: string; insights: string[] }>(
      `/api/v1/coach/analyze/${activityId}`
    ).then((r) => r.data),

  postFeedback: (recId: string, action: "accepted" | "modified" | "rejected" | "skipped", opts?: {
    post_ride_rpe?: number;
    modified_workout_type?: string;
    comment?: string;
  }) =>
    api.post<{ id: string; ok: boolean }>(
      `/api/v1/coach/recommendation/${recId}/feedback`,
      { action, ...opts }
    ).then((r) => r.data),
};

// ─── Profile ─────────────────────────────────────────────────────────────────

export const profileAPI = {
  get: () =>
    api.get<AthleteProfile>("/api/v1/profile").then((r) => r.data),

  update: (data: Partial<AthleteProfile>) =>
    api.put<AthleteProfile>("/api/v1/profile", data).then((r) => r.data),
};

// ─── Garmin ──────────────────────────────────────────────────────────────────

export const garminAPI = {
  status: () =>
    api.get<{ connected: boolean; username: string | null; method: string }>(
      "/api/v1/garmin/status"
    ).then((r) => r.data),

  connectCredentials: (username: string, password: string) =>
    api.post<{ status: string; username: string }>(
      "/api/v1/garmin/connect",
      { username, password }
    ).then((r) => r.data),

  disconnect: () => api.delete("/api/v1/garmin/disconnect").then((r) => r.data),

  sync: (days = 60) =>
    api.post<{ task_id: string }>(`/api/v1/garmin/sync?days=${days}`).then((r) => r.data),

  syncStatus: (taskId: string) =>
    api.get<{ status: string; progress: number; total: number; stats: Record<string, number> | null; error: string | null }>(
      `/api/v1/garmin/sync-status/${taskId}`
    ).then((r) => r.data),
};

// ─── Oura Ring ───────────────────────────────────────────────────────────────

export const ouraAPI = {
  status: () => api.get<{ connected: boolean }>("/api/v1/oura/status").then(r => r.data),
  connect: (token: string) => api.post("/api/v1/oura/connect", { token }).then(r => r.data),
  disconnect: () => api.delete("/api/v1/oura/disconnect").then(r => r.data),
  sync: (days = 60) => api.post<{ task_id: string }>(`/api/v1/oura/sync?days=${days}`).then(r => r.data),
  syncStatus: (taskId: string) => api.get<{ status: string; stats: Record<string, number> | null; error: string | null }>(`/api/v1/oura/sync-status/${taskId}`).then(r => r.data),
};

// ─── Fitbit ──────────────────────────────────────────────────────────────────

export const fitbitAPI = {
  status: () => api.get<{ connected: boolean; user_id: string | null }>("/api/v1/fitbit/status").then(r => r.data),
  authUrl: () => api.get<{ url: string }>("/api/v1/fitbit/auth-url").then(r => r.data),
  disconnect: () => api.delete("/api/v1/fitbit/disconnect").then(r => r.data),
  sync: (days = 60) => api.post<{ task_id: string }>(`/api/v1/fitbit/sync?days=${days}`).then(r => r.data),
  syncStatus: (taskId: string) => api.get<{ status: string; stats: Record<string, number> | null; error: string | null }>(`/api/v1/fitbit/sync-status/${taskId}`).then(r => r.data),
};

// ─── Health (HRV / RHR / Sleep / Body Battery / Readiness) ──────────────────

export interface HealthDay {
  date: string;
  source: string;
  sleep_total_seconds: number | null;
  sleep_score: number | null;
  hrv_overnight_avg_ms: number | null;
  hrv_7d_avg_ms: number | null;
  hrv_status: string | null;
  resting_hr: number | null;
  body_battery_high: number | null;
  body_battery_low: number | null;
  stress_avg: number | null;
}

export interface Readiness {
  score: number;
  status: "green" | "amber" | "red";
  hrv_z: number | null;
  rhr_delta: number | null;
  sleep_score: number | null;
  body_battery: number | null;
  hrv_score: number | null;
  rhr_score: number | null;
  drivers: string[];
  advice: string;
}

export interface DriftStatus {
  state: "stable" | "decoupled" | "stressed" | "unknown";
  drift_pct: number | null;
  trend: "improving" | "worsening" | "stable" | "unknown";
  overtraining_risk: boolean;
  action: string;
  note: string;
}

export interface ManualHealthLog {
  hrv_ms?: number | null;
  resting_hr?: number | null;
  sleep_hours?: number | null;
  sleep_score?: number | null;
  energy_level?: number | null;
  stress_level?: number | null;
}

export const healthAPI = {
  recent: (days = 30) =>
    api.get<{ days: HealthDay[]; readiness: Readiness; drift: DriftStatus | null; has_manual_today: boolean }>(
      `/api/v1/health/recent?days=${days}`
    ).then((r) => r.data),

  readiness: () =>
    api.get<Readiness>("/api/v1/health/readiness").then((r) => r.data),

  logManual: (body: ManualHealthLog) =>
    api.post<HealthDay>("/api/v1/health/log", body).then((r) => r.data),
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
  // Real-time overlay (from dose-log table, computed on GET)
  taken_today: boolean;
  today_total_dose: number | null;
  dose_exceeded: boolean;
  already_enrolled: boolean;
  dose_warning: string | null;
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

export interface DoseLogRecord {
  id: string;
  supplement_key: string;
  label: string;
  dose_taken: number;
  dose_unit: string | null;
  taken_at: string;
  notes: string | null;
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

export const doseLogAPI = {
  create: (body: {
    supplement_key: string;
    label: string;
    dose_taken: number;
    dose_unit?: string;
    taken_at?: string;
    notes?: string;
  }) =>
    api.post<DoseLogRecord>("/api/v1/tracking/dose-log", body).then((r) => r.data),

  list: (params?: { since?: string; supplement_key?: string }) =>
    api
      .get<DoseLogRecord[]>("/api/v1/tracking/dose-log", { params })
      .then((r) => r.data),

  update: (id: string, body: { dose_taken?: number; notes?: string }) =>
    api
      .patch<DoseLogRecord>(`/api/v1/tracking/dose-log/${id}`, body)
      .then((r) => r.data),

  delete: (id: string) =>
    api.delete(`/api/v1/tracking/dose-log/${id}`).then((r) => r.data),
};

export default api;
