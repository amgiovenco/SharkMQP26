import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuthStore } from "../stores/authStore";
import PalegiaLogo from "../assets/logos/pelagiaLogo.svg";

const NAV_LINKS = [
    { name: "Home", path: "/home" },
    { name: "Analysis", path: "/analysis" },
    { name: "History", path: "/history" },
    { name: "Account", path: "/account" },
];

const Navbar = ({ logoSize = "60px" }) => {
    const location = useLocation();
    const navigate = useNavigate();
    const { clearAuth } = useAuthStore();

    const isActive = (path) => location.pathname === path;

    const handleLogout = () => {
        clearAuth();
        navigate("/login", { replace: true });
    };

    return (
        <nav className="sticky top-0 z-20 flex items-center justify-between bg-white px-6 py-3 shadow-sm border-b border-gray-200">
            {/* Left: Logo */}
            <Link to="/home" className="flex items-center space-x-2">
                <img
                    src={PalegiaLogo}
                    alt="Logo"
                    style={{ width: logoSize, height: logoSize }}
                    className="rounded-full"
                />
                <span className="text-lg font-semibold text-pelagia-navy"></span>
            </Link>

            {/* Middle: Nav links */}
            <div className="hidden md:flex items-left space-x-6">
                {NAV_LINKS.map((link) => (
                <Link
                    key={link.path}
                    to={link.path}
                    className={`text-sm font-medium transition-colors ${
                        isActive(link.path)
                            ? "text-pelagia-deepblue"
                            : "text-gray-600 hover:text-pelagia-navy"
                    }`}
                >
                    {link.name}
                </Link>
                ))}
            </div>

            {/* Right: Logout button */}
            <div className="flex items-end space-x-3">
                <button
                    onClick={handleLogout}
                    className="rounded-md bg-pelagia-navy text-white px-3 py-1 text-sm hover:bg-pelagia-deepblue"
                >
                    Logout
                </button>
            </div>
        </nav>
    );
};

export default Navbar;