/**
 * analytics.ts
 * Thin wrapper around Firebase Analytics logEvent.
 * All events are no-ops when analytics is unavailable (e.g., SSR, ad-blockers, dev mode).
 *
 * Usage:
 *   import { track } from "@/lib/analytics";
 *   track("question_asked", { subject_id: "physics", confidence_tier: "HIGH" });
 */

import { logEvent, type Analytics } from "firebase/analytics";
import { analyticsPromise } from "../firebase";

let _analytics: Analytics | null = null;
analyticsPromise.then((a) => { _analytics = a; });

/** Fire a Firebase Analytics event. Safe to call even if analytics is null. */
export function track(eventName: string, params?: Record<string, string | number | boolean>) {
  if (!_analytics) return;
  logEvent(_analytics, eventName, params);
}

// ─── Typed event helpers ──────────────────────────────────────────────────────

/** Fired when a user successfully logs in. */
export const trackLogin = (method: "email" | "google" = "email") =>
  track("login", { method });

/** Fired when a new account is created. */
export const trackSignUp = (method: "email" | "google" = "email") =>
  track("sign_up", { method });

/** Fired when a file is uploaded for RAG indexing. */
export const trackDocumentUpload = (subjectId: string, fileType: string) =>
  track("document_upload", { subject_id: subjectId, file_type: fileType });

/** Fired when a URL is scraped and indexed. */
export const trackUrlUpload = (subjectId: string) =>
  track("url_upload", { subject_id: subjectId });

/** Fired when the user submits a question in the chat interface. */
export const trackQuestionAsked = (subjectId: string) =>
  track("question_asked", { subject_id: subjectId });

/** Fired when an answer is received. Logs the confidence tier. */
export const trackAnswerReceived = (
  subjectId: string,
  confidenceTier: "HIGH" | "MEDIUM" | "LOW" | "NOT_FOUND"
) => track("answer_received", { subject_id: subjectId, confidence_tier: confidenceTier });

/** Fired when study mode is opened. */
export const trackStudyModeStart = (subjectId: string) =>
  track("study_mode_start", { subject_id: subjectId });

/** Fired when a quiz (MCQ/SAQ) is completed. */
export const trackQuizCompleted = (subjectId: string, score: number, total: number) =>
  track("quiz_completed", { subject_id: subjectId, score, total });

/** Fired on an uncaught React error (from ErrorBoundary). */
export const trackAppError = (errorMessage: string, componentStack?: string) =>
  track("app_error", {
    error_message: errorMessage.slice(0, 100),
    component_stack: (componentStack ?? "").slice(0, 150),
  });

