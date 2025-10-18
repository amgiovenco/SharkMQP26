import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { validateJwt } from "./JWTUtil";
import { logoutUser } from "./LogoutUtil";

// Gatekeep most pages from non-logged in users, need to add more (JWT decoding, expiration, etc.)
const ProtectedRoute = ({ children }) => {
    const navigate = useNavigate();

    // Perform the validation synchronously before any rendering
    const isJwtValid = validateJwt();

    // Ensure the JWT is still valid upon page switching
    useEffect(() => {
        if (!isJwtValid) {
            logoutUser(navigate);
        }
    }, [isJwtValid, navigate]);

    // If it is, render the child component (protected page)
    return isJwtValid ? children : null;
;
};

export default ProtectedRoute;