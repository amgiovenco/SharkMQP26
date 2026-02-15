import { useEffect } from "react";
import { Outlet, useNavigate } from "react-router-dom";
import NavBar from '../components/Navbar';
import { SocketProvider } from '../contexts/SocketContext';
import { useAuthStore } from "../stores/authStore";
import { validateJwt } from "./JWTUtil";

const PageLayout = () => {
    const navigate = useNavigate();
    const { jwt, clearAuth } = useAuthStore();

    // Periodically check if the JWT is still valid
    useEffect(() => {
        const checkTokenValidity = () => {
            if (jwt && !validateJwt(jwt)) {
                clearAuth();
                navigate('/login', { replace: true });
            }
        };

        // Check immediately on mount
        checkTokenValidity();

        // Check every 30 seconds
        const interval = setInterval(checkTokenValidity, 30000);

        return () => clearInterval(interval);
    }, [jwt, clearAuth, navigate]);

    return (
        <div className="flex flex-col h-full">
            <NavBar />

            <div className="flex-grow overflow-y-auto">
                {/* Renders a child component (i.e. different pages after login) */}
                <Outlet />
            </div>
        </div>
    );
};

// General format for page layout after login
const MainLayout = () => {
    return (
        <>
            <SocketProvider>
                <PageLayout />
            </SocketProvider>
        </>
    );
}

export default MainLayout;