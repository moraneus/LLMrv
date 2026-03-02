import { Navigate, Outlet, Route, Routes } from "react-router-dom";

import ChatView from "@/components/chat/ChatView";
import RulesView from "@/components/rules/RulesView";
import SettingsView from "@/components/settings/SettingsView";
import ErrorBoundary from "@/components/shared/ErrorBoundary";
import Sidebar from "@/components/shared/Sidebar";

function Layout() {
  return (
    <div className="flex h-screen" data-testid="app-layout">
      <Sidebar />
      <main className="flex-1 overflow-auto" data-testid="main-content">
        <ErrorBoundary>
          <Outlet />
        </ErrorBoundary>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/chat" element={<ChatView />} />
        <Route path="/rules" element={<RulesView />} />
        <Route path="/settings" element={<SettingsView />} />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Route>
    </Routes>
  );
}
