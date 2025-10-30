import { Outlet } from "react-router-dom";
import NavBar from '../components/Navbar';
import { SocketProvider } from '../contexts/SocketContext';

const PageLayout = () => {
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