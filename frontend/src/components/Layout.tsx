/** Main application layout with navigation sidebar. */

import { Link, useLocation } from "react-router-dom";
import {
  FlaskConical,
  LayoutDashboard,
  Microscope,
  TrendingUp,
  BrainCircuit,
  FileText,
  Menu,
  X,
  Atom,
} from "lucide-react";
import { useState } from "react";

const navItems = [
  { label: "Dashboard", path: "/", icon: LayoutDashboard },
  { label: "Predictor", path: "/predictor", icon: FlaskConical },
  { label: "Training", path: "/training", icon: TrendingUp },
  { label: "Results", path: "/results", icon: FileText },
  { label: "Explain", path: "/explain", icon: Microscope },
  { label: "Poster", path: "/poster", icon: BrainCircuit },
];

export default function Layout({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100 flex">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-50
          w-64 bg-slate-950/80 backdrop-blur-xl border-r border-slate-800/60
          flex flex-col transition-transform duration-300 ease-in-out
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
        `}
      >
        {/* Logo */}
        <div className="px-6 py-6 border-b border-slate-800/60">
          <Link
            to="/"
            className="flex items-center gap-3 hover:opacity-80 transition-opacity"
            onClick={() => setSidebarOpen(false)}
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Atom className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                Molecular AI
              </h1>
              <p className="text-[10px] text-slate-500 font-medium tracking-wider uppercase">
                Drug Discovery
              </p>
            </div>
          </Link>
        </div>

        {/* Nav Links */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => setSidebarOpen(false)}
                className={`
                  flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium
                  transition-all duration-200
                  ${
                    isActive
                      ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20"
                      : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/50"
                  }
                `}
              >
                <Icon className={`w-4 h-4 ${isActive ? "text-cyan-400" : ""}`} />
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-800/60">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            Backend: {import.meta.env.VITE_API_BASE || "localhost:8000"}
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 min-w-0 flex flex-col">
        {/* Mobile header */}
        <header className="lg:hidden flex items-center justify-between px-4 py-3 bg-slate-950/50 backdrop-blur border-b border-slate-800/60 sticky top-0 z-30">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
          <span className="text-sm font-semibold text-slate-200">Molecular AI</span>
          <div className="w-9" />
        </header>

        {/* Page content */}
        <div className="flex-1 p-4 lg:p-8 overflow-auto">{children}</div>
      </main>
    </div>
  );
}
