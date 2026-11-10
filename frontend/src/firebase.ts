import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";
import { getFunctions } from "firebase/functions";
import { getAnalytics, isSupported } from "firebase/analytics";
import { initializeAppCheck, ReCaptchaV3Provider } from "firebase/app-check";

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
  measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID,
};

const app = initializeApp(firebaseConfig);

// ─── App Check (reCAPTCHA v3) ─────────────────────────────────────────────────
// Protects Firestore & Storage from unauthorized API access.
// In development, set VITE_APPCHECK_DEBUG_TOKEN=true in frontend/.env to bypass.
if (import.meta.env.DEV && import.meta.env.VITE_APPCHECK_DEBUG_TOKEN === "true") {
  // @ts-expect-error – Firebase debug token for local dev
  self.FIREBASE_APPCHECK_DEBUG_TOKEN = true;
}

if (import.meta.env.VITE_RECAPTCHA_V3_SITE_KEY) {
  initializeAppCheck(app, {
    provider: new ReCaptchaV3Provider(import.meta.env.VITE_RECAPTCHA_V3_SITE_KEY),
    isTokenAutoRefreshEnabled: true,
  });
}

// ─── Core services ────────────────────────────────────────────────────────────
export const auth = getAuth(app);
export const db = getFirestore(app, "neww"); // 'neww' database — where Firestore rules are deployed
export const storage = getStorage(app);
export const functions = getFunctions(app);

// ─── Analytics (async – only in environments that support it) ─────────────────
export const analyticsPromise = isSupported().then((yes) =>
  yes ? getAnalytics(app) : null
);
