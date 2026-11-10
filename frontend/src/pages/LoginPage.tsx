import { useState, FormEvent } from "react";
import { Zap } from "lucide-react";
import { authApi } from "../api/authApi";
import { authStore } from "../store/authStore";

const LoginPage = () => {
    const [name, setName] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [isRegister, setIsRegister] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if (!email.trim() || !password.trim()) return;
        if (isRegister && !name.trim()) return;

        setError("");
        setLoading(true);
        try {
            if (isRegister) {
                await authApi.register({ name: name.trim(), email: email.trim(), password });
                const res = await authApi.login({ email: email.trim(), password });
                authStore.loginNew(res.user.name, res.token, res.user);
            } else {
                const res = await authApi.login({ email: email.trim(), password });
                authStore.login(res.user.name, res.token, res.user);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : "Authentication failed. Try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <div className="grid-bg" />
            <div className="login-page">
                <form className="login-card" onSubmit={handleSubmit}>
                    <div className="login-card__arc">
                        <Zap size={32} color="var(--arc-blue)" />
                    </div>
                    <h1 className="login-card__title">AskMyNotes</h1>
                    <p className="login-card__subtitle">S.T.U.D.Y. Initiative</p>

                    {isRegister && (
                        <div className="login-field">
                            <input
                                type="text"
                                placeholder="Your Name"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                autoComplete="name"
                            />
                        </div>
                    )}

                    <div className="login-field">
                        <input
                            type="email"
                            placeholder="Email Address"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            autoComplete="email"
                            required
                        />
                    </div>
                    <div className="login-field">
                        <input
                            type="password"
                            placeholder="Password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            autoComplete={isRegister ? "new-password" : "current-password"}
                            required
                        />
                    </div>

                    <button className="login-btn" type="submit" disabled={loading}>
                        {loading ? "Authenticating..." : isRegister ? "CREATE ACCOUNT" : "ACCESS INITIATIVE"}
                    </button>

                    {error && <p className="login-error">{error}</p>}

                    <p className="login-register">
                        {isRegister ? "Already have access? " : "New recruit? "}
                        <button type="button" onClick={() => { setIsRegister(!isRegister); setError(""); }}>
                            {isRegister ? "Sign In" : "Register"}
                        </button>
                    </p>
                </form>
            </div>
        </>
    );
};

export default LoginPage;
