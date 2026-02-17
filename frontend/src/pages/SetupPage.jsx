import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import AuthBackground from "../components/AuthBackground";
import { useAuthStore } from "../stores/authStore";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

const SetupPage = () => {
    const navigate = useNavigate();
    const { setAuth } = useAuthStore();

    const [step, setStep] = useState(1);
    const [orgForm, setOrgForm] = useState({ org_name: "", org_description: "" });
    const [adminForm, setAdminForm] = useState({
        admin_first_name: "",
        admin_last_name: "",
        admin_email: "",
        admin_password: "",
        admin_confirm_password: "",
    });
    const [error, setError] = useState(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Step 3: auto-redirect to /home after 2500ms
    useEffect(() => {
        if (step !== 3) return;
        const timer = setTimeout(() => navigate("/home", { replace: true }), 2500);
        return () => clearTimeout(timer);
    }, [step, navigate]);

    const handleStep1Next = () => {
        if (!orgForm.org_name.trim()) {
            setError("Organization name is required.");
            return;
        }
        setError(null);
        setStep(2);
    };

    const handleStep2Submit = async () => {
        const { admin_first_name, admin_last_name, admin_email, admin_password, admin_confirm_password } = adminForm;

        if (!admin_first_name.trim() || !admin_last_name.trim() || !admin_email.trim() || !admin_password) {
            setError("All fields are required.");
            return;
        }
        if (admin_password.length < 8) {
            setError("Password must be at least 8 characters.");
            return;
        }
        if (admin_password !== admin_confirm_password) {
            setError("Passwords do not match.");
            return;
        }

        setError(null);
        setIsSubmitting(true);

        try {
            const res = await fetch(`${API_BASE_URL}/setup/complete`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    org_name: orgForm.org_name.trim(),
                    org_description: orgForm.org_description.trim() || null,
                    admin_email: admin_email.trim(),
                    admin_password,
                    admin_first_name: admin_first_name.trim(),
                    admin_last_name: admin_last_name.trim(),
                }),
            });

            if (!res.ok) {
                let message = `HTTP ${res.status}`;
                try {
                    const data = await res.json();
                    if (data?.detail) message = data.detail;
                } catch {
                    // ignore
                }
                throw new Error(message);
            }

            const data = await res.json();
            const { access_token, user } = data;

            setAuth(
                access_token,
                user.id,
                user.email,
                user.role,
                user.first_name,
                user.last_name,
                user.job_title,
                user.is_system_admin || false,
                user.organizations || []
            );

            setStep(3);
        } catch (err) {
            setError(err.message || "Setup failed. Please try again.");
        } finally {
            setIsSubmitting(false);
        }
    };

    const inputClass =
        "w-full h-[47px] px-4 border-b-2 border-pelagia-inputborder focus:border-pelagia-blue outline-none transition bg-transparent text-gray-700 placeholder-gray-400";

    return (
        <div className="min-h-screen bg-white flex relative overflow-hidden">
            <AuthBackground />

            <div className="relative z-10 w-full flex items-center justify-center px-8">
                <div className="w-full max-w-[575px]">
                    {/* Heading */}
                    <div className="text-center mb-6">
                        <h1 className="text-[75px] font-black text-pelagia-darknavy leading-[28px] tracking-[-1.5px] mb-7">
                            Setup
                        </h1>
                        <div className="w-[279px] h-1 bg-gradient-login mx-auto mb-8" />
                        <p className="text-gray-600 text-lg mb-4">
                            Welcome! Let's get your organization set up.
                        </p>

                        {/* Step indicator dots */}
                        {step < 3 && (
                            <div className="flex justify-center gap-2 mb-6">
                                {[1, 2].map((s) => (
                                    <div
                                        key={s}
                                        className={`w-3 h-3 rounded-full transition-colors ${
                                            step >= s ? "bg-pelagia-deepblue" : "bg-gray-300"
                                        }`}
                                    />
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Step 1: Organization */}
                    {step === 1 && (
                        <div className="space-y-6 flex flex-col items-center">
                            <p className="text-gray-500 text-sm self-start">Step 1 of 2 — Organization Details</p>

                            <input
                                type="text"
                                placeholder="Organization Name *"
                                value={orgForm.org_name}
                                onChange={(e) => setOrgForm((prev) => ({ ...prev, org_name: e.target.value }))}
                                className={inputClass}
                            />

                            <textarea
                                placeholder="Organization Description (optional)"
                                value={orgForm.org_description}
                                onChange={(e) => setOrgForm((prev) => ({ ...prev, org_description: e.target.value }))}
                                rows={3}
                                className="w-full px-4 py-3 border-b-2 border-pelagia-inputborder focus:border-pelagia-blue outline-none transition bg-transparent text-gray-700 placeholder-gray-400 resize-none"
                            />

                            {error && <p className="text-red-500 text-sm text-center">{error}</p>}

                            <button
                                onClick={handleStep1Next}
                                className="w-full h-[60px] bg-pelagia-deepblue hover:bg-opacity-90 text-white font-semibold rounded-lg transition duration-200 shadow-md"
                            >
                                Next →
                            </button>
                        </div>
                    )}

                    {/* Step 2: Admin Account */}
                    {step === 2 && (
                        <div className="space-y-6 flex flex-col items-center">
                            <p className="text-gray-500 text-sm self-start">Step 2 of 2 — Admin Account</p>

                            <div className="flex gap-4 w-full">
                                <input
                                    type="text"
                                    placeholder="First Name *"
                                    value={adminForm.admin_first_name}
                                    onChange={(e) =>
                                        setAdminForm((prev) => ({ ...prev, admin_first_name: e.target.value }))
                                    }
                                    className="flex-1 h-[47px] px-4 border-b-2 border-pelagia-inputborder focus:border-pelagia-blue outline-none transition bg-transparent text-gray-700 placeholder-gray-400"
                                />
                                <input
                                    type="text"
                                    placeholder="Last Name *"
                                    value={adminForm.admin_last_name}
                                    onChange={(e) =>
                                        setAdminForm((prev) => ({ ...prev, admin_last_name: e.target.value }))
                                    }
                                    className="flex-1 h-[47px] px-4 border-b-2 border-pelagia-inputborder focus:border-pelagia-blue outline-none transition bg-transparent text-gray-700 placeholder-gray-400"
                                />
                            </div>

                            <input
                                type="email"
                                placeholder="Email *"
                                value={adminForm.admin_email}
                                onChange={(e) =>
                                    setAdminForm((prev) => ({ ...prev, admin_email: e.target.value }))
                                }
                                className={inputClass}
                            />

                            <input
                                type="password"
                                placeholder="Password *"
                                value={adminForm.admin_password}
                                onChange={(e) =>
                                    setAdminForm((prev) => ({ ...prev, admin_password: e.target.value }))
                                }
                                className={inputClass}
                            />

                            <input
                                type="password"
                                placeholder="Confirm Password *"
                                value={adminForm.admin_confirm_password}
                                onChange={(e) =>
                                    setAdminForm((prev) => ({ ...prev, admin_confirm_password: e.target.value }))
                                }
                                className={inputClass}
                            />

                            {error && <p className="text-red-500 text-sm text-center">{error}</p>}

                            <div className="flex items-center justify-between w-full">
                                <button
                                    onClick={() => { setError(null); setStep(1); }}
                                    className="text-pelagia-blue hover:underline font-medium text-sm"
                                >
                                    ← Back
                                </button>

                                <button
                                    onClick={handleStep2Submit}
                                    disabled={isSubmitting}
                                    className="h-[60px] px-8 bg-pelagia-deepblue hover:bg-opacity-90 text-white font-semibold rounded-lg transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-md"
                                >
                                    {isSubmitting ? "Setting up..." : "Complete Setup"}
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Step 3: Success */}
                    {step === 3 && (
                        <div className="flex flex-col items-center text-center space-y-6">
                            <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center">
                                <svg className="w-8 h-8 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                </svg>
                            </div>

                            <h2 className="text-3xl font-black text-pelagia-darknavy">You're all set!</h2>
                            <p className="text-gray-600">
                                Your organization and admin account have been created. Redirecting to dashboard...
                            </p>

                            <button
                                onClick={() => navigate("/home", { replace: true })}
                                className="h-[60px] px-8 bg-pelagia-deepblue hover:bg-opacity-90 text-white font-semibold rounded-lg transition duration-200 shadow-md"
                            >
                                Go to Dashboard →
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default SetupPage;
