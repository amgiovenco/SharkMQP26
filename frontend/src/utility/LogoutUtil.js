const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

export const logoutUser = async (navigate) => {
    const token = localStorage.getItem("access_token");

    if (token) {
        try {
            await fetch(`${API_BASE_URL}/logout`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${token}`,
                },
            });
        } catch (_) {
        }
    }

    localStorage.removeItem("access_token");

    if (navigate) navigate("/login");
};