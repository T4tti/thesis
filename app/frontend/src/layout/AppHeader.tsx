"use client";

import React from "react";
import { Menu, Globe } from "lucide-react";
import { useSidebar } from "@/context/SidebarContext";
import { ThemeToggleButton } from "@/components/common/ThemeToggleButton";
import { useLanguage } from "@/context/LanguageContext";

const AppHeader: React.FC = () => {
  const { toggleSidebar, toggleMobileSidebar, isMobileOpen } = useSidebar();
  const { t, toggle, lang } = useLanguage();

  const handleToggle = () => {
    if (window.innerWidth >= 1024) {
      toggleSidebar();
    } else {
      toggleMobileSidebar();
    }
  };

  return (
    <header className="sticky top-0 z-50 flex w-full border-b border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
      <div className="flex w-full items-center justify-between gap-3 px-3 py-3 sm:px-4 lg:px-6 lg:py-4">
        <button
          type="button"
          className="flex h-10 w-10 items-center justify-center rounded-lg border border-gray-200 text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-700 dark:border-gray-800 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-white"
          onClick={handleToggle}
          aria-label={t.nav.toggleSidebar}
          aria-controls="app-sidebar"
          aria-expanded={isMobileOpen}
        >
          <Menu className="h-4 w-4" />
        </button>

        <div className="flex items-center gap-2">
          <ThemeToggleButton />
          <button
            type="button"
            onClick={toggle}
            aria-label={t.nav.toggleLanguage}
            className="inline-flex h-10 items-center gap-2 rounded-lg border border-gray-200 px-2.5 text-sm font-medium text-gray-700 hover:bg-gray-100 dark:border-gray-800 dark:text-gray-300 dark:hover:bg-gray-800 sm:px-3"
          >
            <Globe className="h-4 w-4" />
            <span className="hidden sm:inline">{lang === "en" ? "VI" : "EN"}</span>
          </button>
        </div>
      </div>
    </header>
  );

};

export default AppHeader;
