// ─── Athlete / User ─────────────────────────────────────────────────────────

export interface User {
  id: string;
  email: string;
  name: string;
  avatar_url?: string;
  strava_connected: boolean;
  garmin_connected: boolean;
  created_at: string;
}

export interface AthleteProfile {
  id: string;
  user_id: string;
  age: number;
  weight_kg: number;
  height_cm: number;
  sex: "male" | "female" | "other";
  ftp: number;             // Functional Threshold Power (watts)
  max_hr: number;          // bpm
  resting_hr: number;      // bpm
  vo2max_estimate?: number;
  cycling_experience_years: number;
  primary_goal: GoalType;
  goal_event_date?: string;
  goal_event_name?: string;
  training_days_per_week: number;
  diet?: "omnivore" | "vegetarian" | "vegan" | "pescatarian" | "keto";
  climate?: "temperate" | "hot_humid" | "hot_dry" | "cold" | "northern_winter" | "indoor_only";
  event_type?:
    | "long_road"
    | "crit"
    | "tt"
    | "stage_race"
    | "gran_fondo"
    | "climbing_camp"
    | "mtb_marathon"
    | "ultra_endurance"
    | "triathlon_70_3"
    | "triathlon_140_6";
  recent_illness_count_3m?: number;
}

export type GoalType =
  | "general_fitness"
  | "weight_loss"
  | "ftp_improvement"
  | "event_specific"
  | "gran_fondo"
  | "criterium"
  | "climbing"
  | "triathlon";

// ─── Activity ────────────────────────────────────────────────────────────────

export interface Activity {
  id: string;
  user_id: string;
  external_id?: string;     // Strava/Garmin ID
  source: "strava" | "garmin" | "gpx" | "fit" | "manual";
  name: string;
  date: string;             // ISO
  duration_seconds: number;
  distance_meters: number;
  elevation_gain_meters: number;

  // Power
  avg_power?: number;
  max_power?: number;
  normalized_power?: number;
  intensity_factor?: number;
  tss?: number;             // Training Stress Score

  // Heart Rate
  avg_hr?: number;
  max_hr?: number;
  hr_drift?: number;        // Aerobic decoupling %

  // Cadence
  avg_cadence?: number;

  // Environment
  temperature_c?: number;
  humidity_pct?: number;
  wind_speed_kmh?: number;

  // Derived metrics
  aerobic_efficiency?: number;  // Pw:HR ratio
  variability_index?: number;   // NP/AP ratio

  // Zones (% time in each zone)
  time_in_zones?: TimeInZones;

  workout_type?: WorkoutType;
  perceived_exertion?: number;  // 1-10
  notes?: string;
}

export interface TimeInZones {
  z1_recovery: number;
  z2_endurance: number;
  z3_tempo: number;
  z4_threshold: number;
  z5_vo2max: number;
  z6_anaerobic: number;
  z7_neuromuscular: number;
}

export type WorkoutType =
  | "easy"
  | "endurance"
  | "tempo"
  | "sweetspot"
  | "threshold"
  | "vo2max"
  | "sprint"
  | "race"
  | "recovery"
  | "long_ride";

// ─── Fitness Metrics (PMC - Performance Management Chart) ───────────────────

export interface FitnessSnapshot {
  date: string;
  ctl: number;   // Chronic Training Load (fitness) — 42-day EMA
  atl: number;   // Acute Training Load (fatigue)   —  7-day EMA
  tsb: number;   // Training Stress Balance (form)  = CTL - ATL
  tss: number;   // Daily TSS
  ftp: number;
  ftp_method?: string | null;
  ftp_meta?: {
    method: string;
    confidence: number;
    sample_count: number;
    trend: string;
    best_ride_age_days: number | null;
    last_test_age_days: number | null;
    confidence_low: number | null;
    confidence_high: number | null;
  } | null;
}

export interface FitnessProgression {
  current: FitnessSnapshot;
  history: FitnessSnapshot[];
  ftp_history: { date: string; ftp: number }[];
  personal_records: PersonalRecord[];
}

export interface PersonalRecord {
  duration_seconds: number;   // e.g. 5, 60, 300, 1200, 3600
  power_watts: number;
  date: string;
  activity_id: string;
}

// ─── AI Coach Recommendation ─────────────────────────────────────────────────

export interface CoachRecommendation {
  id: string;
  generated_at: string;
  confidence: number;            // 0-1
  model_version: string;
  is_cold_start: boolean;        // true = rule-based fallback

  // Next workout
  next_workout: WorkoutPlan;

  // Weekly plan
  weekly_plan: WorkoutPlan[];

  // Insights
  insights: CoachInsight[];

  // Fitness forecast
  forecast: FitnessForecast;

  // Risk flags
  risks: TrainingRisk[];
}

export interface WorkoutPlan {
  day_offset: number;            // 0 = today, 1 = tomorrow…
  workout_type: WorkoutType;
  duration_minutes: number;
  description: string;
  structure: WorkoutInterval[];
  target_tss: number;
  rationale: string;             // Why this workout
  key_metric: string;            // e.g. "Keep HR below 145 bpm"
}

export interface WorkoutInterval {
  phase: "warmup" | "main" | "cooldown" | "rest";
  duration_minutes: number;
  power_target_pct_ftp?: number; // % of FTP
  hr_target_bpm?: number;
  cadence_rpm?: number;
  description: string;
}

export interface CoachInsight {
  type: "progress" | "warning" | "tip" | "achievement";
  title: string;
  body: string;
  metric?: string;
  value?: number;
  unit?: string;
}

export interface FitnessForecast {
  weeks: number;    // horizon (e.g. 8)
  predicted_ftp_change_watts: number;
  predicted_ctl_peak: number;
  event_readiness_pct?: number;
  confidence_interval: [number, number];
}

export interface TrainingRisk {
  type: "overtraining" | "undertraining" | "injury" | "illness" | "staleness";
  severity: "low" | "medium" | "high";
  message: string;
}

// ─── Multi-Horizon Recommendation ────────────────────────────────────────────
// One model — three horizons (short / medium / event). Each horizon's payload
// mirrors the standard CoachRecommendation shape (next_workout, weekly_plan…)
// plus a `horizon_label` for the UI.
export type HorizonKey = "short" | "medium" | "event";

export interface HorizonPayload {
  horizon: HorizonKey;
  horizon_days: number;
  horizon_label: string;
  next_workout: WorkoutPlan;
  weekly_plan: WorkoutPlan[];
  insights: CoachInsight[];
  forecast: FitnessForecast;
  risks: TrainingRisk[];
  confidence: number;
  model_version: string;
  readiness?: {
    score: number;
    status: string;
    advice?: string;
  };
  safety_notes?: string[];
}

export interface MultiHorizonRecommendation {
  is_cold_start: boolean;
  model_version: string;
  active_horizon: HorizonKey;
  horizons: Partial<Record<HorizonKey, HorizonPayload>>;
  supplements?: Record<string, unknown> | null;
}

// ─── Macrocycle (reverse-periodization plan to event date) ──────────────────
export interface MacrocycleWeek {
  week_index: number;
  week_start: string;          // ISO date of the Monday
  weeks_to_event: number;
  phase: "base" | "build" | "peak" | "taper" | "event_week";
  target_weekly_tss: number;
  target_ctl_end: number;
  workout_focus: string[];
  is_recovery_week: boolean;
  notes: string;
}

export interface Macrocycle {
  event_date: string;
  event_name?: string | null;
  event_type?: string | null;
  current_ctl: number;
  current_atl: number;
  peak_ctl_target: number;
  planned_tsb_event: number;
  weeks_to_event: number;
  days_to_event: number;
  feasibility: "comfortable" | "balanced" | "ambitious" | "unrealistic";
  confidence: number;
  weeks: MacrocycleWeek[];
  summary: string[];
  method: string;
  error?: string;
}

// ─── Upload ──────────────────────────────────────────────────────────────────

export interface UploadResult {
  activity_id: string;
  status: "success" | "duplicate" | "error";
  message: string;
  activity?: Partial<Activity>;
}

// ─── API responses ───────────────────────────────────────────────────────────

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

export interface APIError {
  detail: string;
  code?: string;
}
