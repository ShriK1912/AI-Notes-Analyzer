import apiClient, { uploadAxios } from "./axios";
import { auth } from "../firebase";
import { trackDocumentUpload, trackUrlUpload } from "../lib/analytics";

export const uploadApi = {
  async uploadNotes(file: File, subjectId: string): Promise<{ job_id: string }> {
    const user = auth.currentUser;
    if (!user) throw new Error("Not authenticated");
    const token = await user.getIdToken();

    const formData = new FormData();
    formData.append("file", file);
    formData.append("subjectId", subjectId);

    const res = await uploadAxios.post("/upload/", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
        "Authorization": `Bearer ${token}`
      }
    });
    trackDocumentUpload(subjectId, file.name.split(".").pop() ?? "unknown");
    return res.data as { job_id: string };
  },

  async uploadFromUrl(url: string, subjectId: string): Promise<{ message: string; documentId: string; chunkCount: number; sourceUrl: string; title: string }> {
    const user = auth.currentUser;
    if (!user) throw new Error("Not authenticated");
    const token = await user.getIdToken();

    const formData = new FormData();
    formData.append("url", url);
    formData.append("subjectId", subjectId);

    const res = await apiClient.post("/upload/url", formData, {
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": `Bearer ${token}`
      }
    });
    trackUrlUpload(subjectId);
    return res.data as any;
  },
};
