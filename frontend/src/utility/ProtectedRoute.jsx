import { Navigate } from "react-router-dom";
import { useAuthStore } from "../stores/authStore";
import { validateJwt } from "./JWTUtil";

const ProtectedRoute = ({ children }) => {
    const { isAuthenticated, jwt, clearAuth } = useAuthStore();

    // Check if user is authenticated and has a valid JWT
    if (!isAuthenticated || !jwt) {
        return <Navigate to="/login" replace />;
    }

    // Validate JWT - check if it's expired
    if (!validateJwt(jwt)) {
        clearAuth();
        return <Navigate to="/login" replace />;
    }

    // Allow authenticated users with valid tokens to proceed
    return children;
};

export default ProtectedRoute;