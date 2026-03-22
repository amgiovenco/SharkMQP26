import { Routes, Route, Navigate } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import ProtectedRoute from "./utility/ProtectedRoute";
import { SocketProvider } from "./contexts/SocketContext";

import LoginPage from "./pages/LoginPage";
import AnalysisPage from "./pages/AnalysisPage";

const App = () => {
    return (
        <div className="App h-full">
            <Toaster
                position="top-right"
                toastOptions={{
                    duration: 4000,
                    style: {
                        background: '#363636',
                        color: '#fff',
                    },
                    success: {
                        duration: 3000,
                        iconTheme: {
                            primary: '#10b981',
                            secondary: '#fff',
                        },
                    },
                    error: {
                        duration: 5000,
                        iconTheme: {
                            primary: '#ef4444',
                            secondary: '#fff',
                        },
                    },
                }}
            />
            <Routes>
                <Route path="/" element={<LoginPage />} />
                <Route
                    path="/analysis"
                    element={
                        <ProtectedRoute>
                            <SocketProvider>
                                <AnalysisPage />
                            </SocketProvider>
                        </ProtectedRoute>
                    }
                />
                <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
        </div>
    );
};

export default App;
