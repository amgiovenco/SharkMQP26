import { Link, useLocation } from "react-router-dom";
import { HiHome, HiOutlineChartBar, HiOutlineClock, HiOutlineUser, HiOutlineUsers } from "react-icons/hi";
import PalegiaLogo from "../assets/logos/pelagiaLogo.svg";
import { OrganizationSwitcher } from "./OrganizationSwitcher";
import { usePermissions } from "../hooks/usePermissions";

const Navbar = ({ logoSize = "36px" }) => {
  const location = useLocation();
  const { canManageTeam } = usePermissions();

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="sticky top-0 z-20 flex items-center justify-between bg-white px-6 py-2 border-b border-gray-200 shadow-sm">
      {/* Left: Logo and Organization Switcher */}
      <div className="flex items-center space-x-4">
        <Link to="/home" className="flex items-center">
          <img
            src={PalegiaLogo}
            alt="Pelagia Logo"
            style={{ width: logoSize, height: logoSize }}
            className="object-contain"
          />
        </Link>
        <OrganizationSwitcher />
      </div>

      {/* Right: Icons */}
      <div className="flex items-center space-x-5 text-indigo-600">
        {/* Home */}
        <Link
          to="/home"
          className={`hover:text-indigo-800 transition ${
            isActive("/home") ? "text-indigo-700" : ""
          }`}
        >
          <HiHome className="h-6 w-6" />
        </Link>

        {/* Analysis */}
        <Link
          to="/analysis"
          className={`hover:text-indigo-800 transition ${
            isActive("/analysis") ? "text-indigo-700" : ""
          }`}
        >
          <HiOutlineChartBar className="h-6 w-6" />
        </Link>

        {/* History */}
        <Link
          to="/history"
          className={`hover:text-indigo-800 transition ${
            isActive("/history") ? "text-indigo-700" : ""
          }`}
        >
          <HiOutlineClock className="h-6 w-6" />
        </Link>

        {/* Divider */}
        <span className="h-6 w-px bg-gray-300" />

        {/* Team (if user has permission) */}
        {canManageTeam && (
          <Link
            to="/team"
            className={`hover:text-indigo-800 transition ${
              isActive("/team") ? "text-indigo-700" : ""
            }`}
          >
            <HiOutlineUsers className="h-6 w-6" />
          </Link>
        )}

        {/* Account */}
        <Link
          to="/account"
          className={`hover:text-indigo-800 transition ${
            isActive("/account") ? "text-indigo-700" : ""
          }`}
        >
          <HiOutlineUser className="h-6 w-6" />
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;
