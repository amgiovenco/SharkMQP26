import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { apiFetch } from '../utility/ApiFetch';
import AuthBackground from '../components/AuthBackground';

const RegisterPage = () => {
    const navigate = useNavigate();
    const setAuth = useAuthStore((state) => state.setAuth);

    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [firstName, setFirstName] = useState('');
    const [lastName, setLastName] = useState('');
    const [registrationCode, setRegistrationCode] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleRegister = async (e) => {
        e.preventDefault();
        setError('');

        // Validation
        if (!username || !password || !registrationCode) {
            setError('Username, password, and registration code are required');
            return;
        }

        if (password.length < 8) {
            setError('Password must be at least 8 characters');
            return;
        }

        if (password !== confirmPassword) {
            setError('Passwords do not match');
            return;
        }

        setIsLoading(true);

        try {
            await apiFetch('/auth/signup', {
                method: 'POST',
                body: JSON.stringify({
                    username,
                    password,
                    registration_code: registrationCode.toUpperCase().trim(),
                    first_name: firstName || null,
                    last_name: lastName || null,
                }),
            });

            // After successful registration, login
            const loginData = await apiFetch('/auth/login', {
                method: 'POST',
                body: JSON.stringify({ username, password }),
            });

            setAuth(
                loginData.access_token,
                loginData.user.id,
                loginData.user.username,
                loginData.user.role,
                loginData.user.first_name,
                loginData.user.last_name,
                loginData.user.job_title,
                loginData.user.is_system_admin,
                loginData.user.organizations
            );

            navigate('/');
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-white flex relative overflow-hidden">
            <AuthBackground />

            {/* Main Content */}
            <div className="relative z-10 w-full flex items-center justify-center px-8">
                <div className="w-full max-w-[575px]">
                    <div className="text-center mb-6">
                        <h1 className="text-[75px] font-black text-pelagia-darknavy leading-[28px] tracking-[-1.5px] mb-7">
                            Sign Up
                        </h1>
                        <div className="w-[279px] h-1 bg-gradient-login mx-auto mb-8"/>
                        <p className="text-gray-600 text-lg mb-2">
                            Create your account to access Pelagia.
                        </p>
                        <p className="text-gray-500 text-sm">
                            Already have an account?{" "}
                            <a href="/login" className="text-pelagia-blue hover:underline font-medium">
                                Sign in
                            </a>
                        </p>
                    </div>

                    <form onSubmit={handleRegister} className="space-y-6 flex flex-col items-center">
                        <input
                            type="text"
                            placeholder="Registration Code"
                            value={registrationCode}
                            onChange={(e) => setRegistrationCode(e.target.value)}
                            required
                            className="w-[575px] h-[47px] px-4 border-b-2 border-pelagia-inputborder focus:border-pelagia-blue outline-none transition bg-transparent text-gray-700 placeholder-gray-400"
                        />

                        <input
                            type="text"
                            placeholder="Email"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                            className="w-[575px] h-[47px] px-4 border-b-2 border-pelagia-inputborder focus:border-pelagia-blue outline-none transition bg-transparent text-gray-700 placeholder-gray-400"
                        />

                        <input
                            type="text"
                            placeholder="First Name (Optional)"
                            value={firstName}
                            onChange={(e) => setFirstName(e.target.value)}
                            className="w-[575px] h-[47px] px-4 border-b-2 border-pelagia-inputborder focus:border-pelagia-blue outline-none transition bg-transparent text-gray-700 placeholder-gray-400"
                        />

                        <input
                            type="text"
                            placeholder="Last Name (Optional)"
                            value={lastName}
                            onChange={(e) => setLastName(e.target.value)}
                            className="w-[575px] h-[47px] px-4 border-b-2 border-pelagia-inputborder focus:border-pelagia-blue outline-none transition bg-transparent text-gray-700 placeholder-gray-400"
                        />

                        <input
                            type="password"
                            placeholder="Password (At least 8 characters)"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            className="w-[575px] h-[47px] px-4 border-b-2 border-pelagia-inputborder focus:border-pelagia-blue outline-none transition bg-transparent text-gray-700 placeholder-gray-400"
                        />

                        <input
                            type="password"
                            placeholder="Confirm Password"
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            required
                            className="w-[575px] h-[47px] px-4 border-b-2 border-pelagia-inputborder focus:border-pelagia-blue outline-none transition bg-transparent text-gray-700 placeholder-gray-400"
                        />

                        {error && (
                            <p className="text-red-500 text-sm text-center">
                                {error}
                            </p>
                        )}

                        <button
                            type="submit"
                            disabled={isLoading}
                            className="w-[575px] h-[60px] bg-pelagia-deepblue hover:bg-opacity-90 text-white font-semibold rounded-lg transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-md"
                        >
                            {isLoading ? 'Creating Account...' : 'Create Account'}
                        </button>

                        <p className="text-center text-sm text-gray-600">
                            Need help?{" "}
                            <span className="text-gray-500">
                                Ask your organization admin for a registration code
                            </span>
                        </p>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default RegisterPage;