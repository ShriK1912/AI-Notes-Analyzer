import React from "react";
import { trackAppError } from "../lib/analytics";

interface Props {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * ErrorBoundary — catches unhandled React render errors and:
 *   1. Reports them to Firebase Analytics (replaces Crashlytics for web)
 *   2. Shows a friendly fallback UI instead of a blank screen
 *
 * Usage:
 *   <ErrorBoundary>
 *     <YourComponent />
 *   </ErrorBoundary>
 */
class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Report to Firebase Analytics (Crashlytics replacement for web)
    trackAppError(error.message, info.componentStack ?? "");
    console.error("[ErrorBoundary] Caught error:", error, info);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;

      return (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            minHeight: "40vh",
            gap: "1rem",
            padding: "2rem",
            textAlign: "center",
            color: "var(--text-primary, #e0e0e0)",
          }}
        >
          <h2 style={{ fontSize: "1.4rem", color: "var(--arc-blue, #60a5fa)" }}>
            ⚡ System Anomaly Detected
          </h2>
          <p style={{ opacity: 0.7, maxWidth: 400 }}>
            An unexpected error occurred. It has been reported automatically.
          </p>
          {this.state.error && (
            <code
              style={{
                fontSize: "0.75rem",
                opacity: 0.5,
                maxWidth: 480,
                wordBreak: "break-word",
              }}
            >
              {this.state.error.message}
            </code>
          )}
          <button
            onClick={this.handleReset}
            style={{
              marginTop: "0.5rem",
              padding: "0.5rem 1.5rem",
              background: "var(--arc-blue, #60a5fa)",
              color: "#000",
              border: "none",
              borderRadius: "6px",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

