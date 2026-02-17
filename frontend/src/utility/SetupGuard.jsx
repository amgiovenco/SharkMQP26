import { useState, useEffect } from "react";
import { useLocation, Navigate } from "react-router-dom";
import { useAuthStore } from "../stores/authStore";
import { is } from "express/lib/request";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

const SetupGuard = ({ children }) => {
    const [needsSetup, setNeedsSetup] = useState(null);
    const location = useLocation();
    const { isAuthenticated } = useAuthStore();

    useEffect(() => {
        if (isAuthenticated) return;
        setNeedsSetup(null);
        fetch(`${API_BASE_URL}/setup/status`)
            .then((res) => res.json())
            .then((data) => setNeedsSetup(data.needs_setup === true))
            .catch(() => setNeedsSetup(false));
    }, [isAuthenticated]);

    if (needsSetup === null) {
        return (
            <div className="min-h-screen bg-white flex items-center justify-center">
                <div className="w-8 h-8 border-4 border-pelagia-deepblue border-t-transparent rounded-full animate-spin" />
            </div>
        );
    }

    // Only redirect to setup if not already authenticated (e.g. just completed setup)
    if (needsSetup && !isAuthenticated && location.pathname !== "/setup") {
        return <Navigate to="/setup" replace />;
    }

    if (!needsSetup && location.pathname === "/setup") {
        return <Navigate to="/login" replace />;
    }

    return children;
};

export default SetupGuard;
