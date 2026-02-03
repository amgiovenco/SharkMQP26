import { Routes, Route, Navigate } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import MainLayout from "./utility/MainLayout";
import ProtectedRoute from "./utility/ProtectedRoute";
import { useInitializeApp } from "./hooks/useInitializeApp";

// Public pages
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";

// Protected pages
import HomePage from "./pages/HomePage";
import AnalysisPage from "./pages/AnalysisPage";
import AnalysisHistoryPage from "./pages/AnalysisHistoryPage";
import CaseDetailPage from "./pages/CaseDetailPage";
import AccountPage from "./pages/AccountPage";
import { TeamManagementPage } from "./pages/TeamManagementPage";

const App = () => {
    // Initialize app on mount - fetches fresh user data and cases
    useInitializeApp();

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
                {/* Public (no navbar) */}
                <Route path="/" element={<LoginPage />} />
                <Route path="/login" element={<LoginPage />} />
                <Route path="/register" element={<RegisterPage />} />

                {/* Protected area (with navbar via MainLayout) */}
                <Route element={<MainLayout />}>
                {/* Default protected index -> /home */}
                <Route
                    index
                    element={<Navigate to="/home" replace />}
                />

                <Route
                    path="/home"
                    element={
                    <ProtectedRoute>
                        <HomePage />
                    </ProtectedRoute>
                    }
                />

                <Route
                    path="/analysis"
                    element={
                    <ProtectedRoute>
                        <AnalysisPage />
                    </ProtectedRoute>
                    }
                />

                <Route
                    path="/history"
                    element={
                    <ProtectedRoute>
                        <AnalysisHistoryPage />
                    </ProtectedRoute>
                    }
                />

                {/* Detail page (e.g., a specific case/run) */}
                <Route
                    path="/case/:caseId"
                    element={
                    <ProtectedRoute>
                        <CaseDetailPage />
                    </ProtectedRoute>
                    }
                />

                <Route
                    path="/account"
                    element={
                    <ProtectedRoute>
                        <AccountPage />
                    </ProtectedRoute>
                    }
                />

                <Route
                    path="/team"
                    element={
                    <ProtectedRoute>
                        <TeamManagementPage />
                    </ProtectedRoute>
                    }
                />
                </Route>

                {/* Catch-all */}
                <Route path="*" element={<Navigate to="/home" replace />} />
            </Routes>
        </div>
    );
};

export default App;