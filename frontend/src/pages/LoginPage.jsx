import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { useAuthStore } from "../stores/authStore";
import AuthBackground from "../components/AuthBackground";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

const LoginPage = () => {
    const navigate = useNavigate();
    const [key, setKey] = useState("");
    const { setAuth } = useAuthStore();

    const verifyMutation = useMutation({
        mutationFn: async () => {
            const response = await fetch(`${API_BASE_URL}/auth/verify-key`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ key }),
            });

            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.detail || "Invalid access key");
            }

            return response.json();
        },
        onSuccess: (data) => {
            const token = data?.access_token;
            if (!token) {
                throw new Error("No token returned");
            }
            setAuth(token);
            navigate("/analysis", { replace: true });
        },
    });

    const handleSubmit = (e) => {
        e.preventDefault();
        verifyMutation.mutate();
    };

    return (
        <div className="min-h-screen bg-white flex relative overflow-hidden">
            <AuthBackground />

            <div className="relative z-10 w-full flex items-center justify-center px-8">
                <div className="w-full max-w-[575px]">
                    <div className="text-center mb-6">
                        <h1 className="text-[75px] font-black text-pelagia-darknavy leading-[28px] tracking-[-1.5px] mb-7">
                            Pelagia
                        </h1>
                        <div className="w-[279px] h-1 bg-gradient-login mx-auto mb-8" />
                        <p className="text-gray-600 text-lg mb-2">
                            Enter your access key to begin analysis.
                        </p>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-6 flex flex-col items-center">
                        <input
                            type="password"
                            name="accessKey"
                            placeholder="Access Key"
                            value={key}
                            onChange={(e) => setKey(e.target.value)}
                            required
                            autoFocus
                            className="w-[575px] h-[47px] px-4 border-b-2 border-pelagia-inputborder focus:border-pelagia-blue outline-none transition bg-transparent text-gray-700 placeholder-gray-400"
                        />

                        {verifyMutation.isError && (
                            <p className="text-red-500 text-sm text-center">
                                {verifyMutation.error?.message || "Invalid access key. Please try again."}
                            </p>
                        )}

                        <button
                            type="submit"
                            disabled={verifyMutation.isPending}
                            className="w-[575px] h-[60px] bg-pelagia-deepblue hover:bg-opacity-90 text-white font-semibold rounded-lg transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-md"
                        >
                            {verifyMutation.isPending ? "Verifying..." : "Enter"}
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default LoginPage;
