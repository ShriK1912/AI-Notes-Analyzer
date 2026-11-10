import apiClient from "./axios";
import { auth } from "../firebase";
import type { AnswerPayload } from "../types/document";
import type { Subject } from "../types/subject";
import { trackQuestionAsked, trackAnswerReceived } from "../lib/analytics";

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

const strictNotFound = (subjectName: string): AnswerPayload => ({
  answer: `Not found in your notes for ${subjectName}`,
  confidenceTier: "NOT_FOUND",
  confidenceScore: 0,
  citations: [],
  evidenceSnippets: []
});

const normalizeResponse = (payload: Partial<AnswerPayload>, subjectName: string): AnswerPayload => {
  const answer = payload.answer?.trim() ?? "";
  const confidenceTier = payload.confidenceTier ?? "NOT_FOUND";
  const confidenceScore = Number(payload.confidenceScore ?? 0);
  const citations = payload.citations ?? [];
  const evidenceSnippets = payload.evidenceSnippets ?? [];

  if (
    confidenceTier === "NOT_FOUND" ||
    answer.toLowerCase().startsWith("not found in your notes") ||
    citations.length === 0
  ) {
    return {
      ...strictNotFound(subjectName),
      evidenceSnippets
    };
  }

  return {
    answer,
    confidenceTier,
    confidenceScore,
    citations,
    evidenceSnippets,
    topChunkIds: payload.topChunkIds ?? []
  };
};

export const qaApi = {
  async askQuestion(question: string, subject: Subject, history: ChatTurn[] = []): Promise<AnswerPayload> {
    const user = auth.currentUser;
    if (!user) throw new Error("Not authenticated");
    const token = await user.getIdToken();

    try {
      trackQuestionAsked(subject.id);
      const res = await apiClient.post("/qa/ask", {
        subjectId: subject.id,
        question,
        history
      }, {
        headers: { "Authorization": `Bearer ${token}` }
      });
      const result = normalizeResponse(res.data as Partial<AnswerPayload>, subject.name);
      trackAnswerReceived(subject.id, result.confidenceTier as "HIGH" | "MEDIUM" | "LOW" | "NOT_FOUND");
      return result;
    } catch (error) {
      console.error("QA API Error", error);
      throw error;
    }
  }
};
