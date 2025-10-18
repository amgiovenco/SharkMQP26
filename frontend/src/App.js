import { Routes, Route, Navigate } from "react-router-dom";
import MainLayout from "./utility/MainLayout";
import ProtectedRoute from "./utility/ProtectedRoute";

// Public pages
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";

// Protected pages
import HomePage from "./pages/HomePage";
import AnalysisPage from "./pages/AnalysisPage";
import AnalysisHistoryPage from "./pages/AnalysisHistoryPage";
import CasePage from "./pages/CasePage";
import AccountPage from "./pages/AccountPage";

const App = () => {
  return (
    <div className="App h-full">
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
                <CasePage />
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
        </Route>

        {/* Catch-all */}
        <Route path="*" element={<Navigate to="/home" replace />} />
      </Routes>
    </div>
  );
};

export default App;