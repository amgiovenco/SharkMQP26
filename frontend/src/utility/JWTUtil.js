import { jwtDecode } from 'jwt-decode'

// Function to validate JWT
export const validateJwt = (jwt) => {
    // If no JWT is found, it is not valid
    if (!jwt) {
        return false;
    }

    // Decode the JWT and check the expiration
    try {
        const decodedToken = jwtDecode(jwt);
        const currentTime = Date.now() / 1000; // convert to seconds

        // If the JWT is expired, it is not valid
        if (decodedToken.exp && decodedToken.exp < currentTime) {
            return false;
        }

        return true;

    } catch (error) {
        return false;
    }
};

// Check if JWT is about to expire (within 5 minutes)
export const isJwtExpiringSoon = (jwt) => {
    if (!jwt) return false;

    try {
        const decodedToken = jwtDecode(jwt);
        const currentTime = Date.now() / 1000;
        const fiveMinutes = 5 * 60;

        return decodedToken.exp && (decodedToken.exp - currentTime) < fiveMinutes;
    } catch {
        return false;
    }
};