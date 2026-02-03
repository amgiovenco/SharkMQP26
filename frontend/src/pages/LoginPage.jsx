import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { apiFetch } from "../utility/ApiFetch";
import { useAuthStore } from "../stores/authStore";
import { useCasesStore } from "../stores/casesStore";
import AuthBackground from "../components/AuthBackground";

const LoginPage = () => {
    const navigate = useNavigate();
    const [form, setForm] = useState({ username: "", password: "" });
    
    // Zustand stores
    const { setAuth } = useAuthStore();
    const { setCases, setIsLoading: setCasesLoading, setError: setCasesError } = useCasesStore();

    // Mutation for login
    const loginMutation = useMutation({
        mutationFn: async () => {
            return apiFetch("/auth/login", {
                method: "POST",
                body: JSON.stringify(form),
            });
        },
        onSuccess: async (data) => {
            const token = data?.access_token;
            const user = data?.user;

            if (!token) {
                console.error("Login succeeded but no token returned:", data);
                return;
            }

            // Store auth in Zustand
            setAuth(
                token,
                user.id,
                user.username,
                user.role,
                user.first_name,
                user.last_name,
                user.job_title,
                user.is_system_admin || false,
                user.organizations || []
            );

            // Fetch cases after login
            try {
                setCasesLoading(true);
                const casesData = await apiFetch("/cases");
                setCases(casesData.cases || []);
                setCasesLoading(false);
            } catch (err) {
                console.error("Failed to fetch cases:", err);
                setCasesError(err.message);
                setCasesLoading(false);
            }

            // Navigate to home page if login is successful
            navigate("/home", { replace: true });
        },
        onError: (error) => {
            console.error("Login failed:", error);
        },
    });

    const handleSubmit = (e) => {
        e.preventDefault();
        loginMutation.mutate();
    };

    return (
        <div className="min-h-screen bg-white flex relative overflow-hidden">
            <AuthBackground />

            {/* Main Content */}
            <div className="relative z-10 w-full flex items-center justify-center px-8">
                <div className="w-full max-w-[575px]">
                    <div className="text-center mb-6">
                        <h1 className="text-[75px] font-black text-pelagia-darknavy leading-[28px] tracking-[-1.5px] mb-7">
                            Login
                        </h1>
                        <div className="w-[279px] h-1 bg-gradient-login mx-auto mb-8"/>
                        <p className="text-gray-600 text-lg mb-2">
                            Welcome back! Login to access Pelagia.
                        </p>
                        <p className="text-gray-500 text-sm">
                            New here?{" "}
                            <a href="/register" className="text-pelagia-blue hover:underline font-medium">
                                Create an Account
                            </a>
                        </p>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-6 flex flex-col items-center">
                        <input
                            type="text"
                            name="username"
                            placeholder="Email"
                            value={form.username}
                            onChange={(e) => setForm((prev) => ({ ...prev, username: e.target.value }))}
                            required
                            className="w-[575px] h-[47px] px-4 border-b-2 border-pelagia-inputborder focus:border-pelagia-blue outline-none transition bg-transparent text-gray-700 placeholder-gray-400"
                        />

                        <input
                            type="password"
                            name="password"
                            placeholder="Password"
                            value={form.password}
                            onChange={(e) => setForm((prev) => ({ ...prev, password: e.target.value }))}
                            required
                            className="w-[575px] h-[47px] px-4 border-b-2 border-pelagia-inputborder focus:border-pelagia-blue outline-none transition bg-transparent text-gray-700 placeholder-gray-400"
                        />

                        {loginMutation.isError && (
                            <p className="text-red-500 text-sm text-center">
                                {loginMutation.error?.message || "Invalid credentials. Please try again."}
                            </p>
                        )}

                        <button
                            type="submit"
                            disabled={loginMutation.isPending}
                            className="w-[575px] h-[60px] bg-pelagia-deepblue hover:bg-opacity-90 text-white font-semibold rounded-lg transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-md"
                        >
                            {loginMutation.isPending ? "Signing in..." : "Login"}
                        </button>

                        <p className="text-center text-sm text-gray-600">
                            Did you{" "}
                            <a href="/forgot-password" className="text-pelagia-blue hover:underline font-medium">
                                forget your password?
                            </a>
                        </p>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default LoginPage;