import { Navigate } from "react-router-dom";
import { useAuthStore } from "../stores/authStore";

// Gatekeep most pages from non-logged in users, need to add more (JWT decoding, expiration, etc.)
const ProtectedRoute = ({ children }) => {
    const { isAuthenticated, jwt } = useAuthStore();
    
    if (!isAuthenticated || !jwt) {
        return <Navigate to="/login" replace />;
    }

    return children;
};

export default ProtectedRoute;