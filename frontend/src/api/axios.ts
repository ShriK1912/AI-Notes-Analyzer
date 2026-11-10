import axios from "axios";
import { auth, appCheck } from "../firebase";
import { getToken } from "firebase/app-check";

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1",
  timeout: 300000, // 5 minutes to accommodate API retries
});

// Always attach a fresh Firebase token before every request
export const uploadAxios = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1",
  timeout: 300000, // 300 seconds for initial file transfer
});

// Reuse the same interceptors logic for both
const attachToken = async (config: any) => {
  try {
    const user = auth.currentUser;
    if (user) {
      const token = await user.getIdToken(false);
      config.headers.Authorization = `Bearer ${token}`;
      config.headers["x-user-id"] = user.uid;
    }
    // Attach Firebase App Check token if configured
    if (appCheck) {
      try {
        const appCheckToken = await getToken(appCheck, /* forceRefresh= */ false);
        config.headers["X-Firebase-AppCheck"] = appCheckToken.token;
      } catch (appCheckErr) {
        console.warn("App Check token retrieval skipped:", appCheckErr);
      }
    }
  } catch (err) {
    console.warn("Failed to attach Firebase token:", err);
  }
  return config;
};

apiClient.interceptors.request.use(attachToken);
uploadAxios.interceptors.request.use(attachToken);

export default apiClient;
