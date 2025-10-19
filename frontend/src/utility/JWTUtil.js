import { jwtDecode } from 'jwt-decode' 

// Function to validate JWT
export const validateJwt = () => {
    const jwt = localStorage.getItem("access_token");

    // If no JWT is found, it is not valid
    if (!jwt) {
        console.log("No JWT found in local storage.");
        return false;
    }

    // Deocde the JWT and check the expiration
    try {
        const decodedToken = jwtDecode(jwt);
        const currentTime = Date.now() / 1000; // convert to seconds

        // If the JWT is expired, it is not valid
        if (decodedToken.exp && decodedToken.exp < currentTime) {
            console.warn("JWT has expired.");
            return false;
        }

        return true;
        
    } catch (error) {
        console.error("Failed to decode JWT:", error);
        return false;
    }
};