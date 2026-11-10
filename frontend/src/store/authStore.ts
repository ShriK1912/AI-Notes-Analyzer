import { useSyncExternalStore } from "react";
import { trackLogin, trackSignUp } from "../lib/analytics";

interface User {
  id: string;
  name: string;
  email: string;
}

interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
}

const TOKEN_KEY = "amn_token";
const USER_KEY = "amn_user";

const loadPersistedState = (): AuthState => {
  try {
    const token = localStorage.getItem(TOKEN_KEY);
    const userJson = localStorage.getItem(USER_KEY);
    if (token && userJson) {
      return { isAuthenticated: true, token, user: JSON.parse(userJson) };
    }
  } catch { /* ignore */ }
  return { isAuthenticated: false, token: null, user: null };
};

let state: AuthState = loadPersistedState();

const listeners = new Set<() => void>();
const notify = () => listeners.forEach((l) => l());

export const authStore = {
  getState: () => state,
  subscribe: (listener: () => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  login(name: string, token: string, user: User) {
    state = { isAuthenticated: true, token, user: { ...user, name } };
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, JSON.stringify(state.user));
    trackLogin("email");
    notify();
  },
  loginNew(name: string, token: string, user: User) {
    // Called after register — fires sign_up event instead of login
    state = { isAuthenticated: true, token, user: { ...user, name } };
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, JSON.stringify(state.user));
    trackSignUp("email");
    notify();
  },
  logout() {
    state = { isAuthenticated: false, token: null, user: null };
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    notify();
  },
};

export const useAuthStore = () =>
  useSyncExternalStore(authStore.subscribe, authStore.getState, authStore.getState);
