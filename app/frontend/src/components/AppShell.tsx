"use client";

import React from "react";
import AppHeader from "@/layout/AppHeader";
import AppSidebar from "@/layout/AppSidebar";
import Backdrop from "@/layout/Backdrop";
import { useSidebar } from "@/context/SidebarContext";
import Footer from "@/components/Footer";

const AppShell: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isExpanded, isMobileOpen } = useSidebar();

  const mainContentMargin = isMobileOpen ? "ml-0" : isExpanded ? "lg:ml-[280px]" : "lg:ml-[84px]";

  return (
    <div className="min-h-screen xl:flex">
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
      <AppSidebar />
      <Backdrop />
      <div className={`flex-1 transition-all duration-300 ease-in-out ${mainContentMargin}`}>
        <AppHeader />
        <main id="main-content" tabIndex={-1} className="mx-auto w-full max-w-screen-2xl px-3 py-4 sm:px-4 md:px-6 md:py-6">
          {children}
        </main>
        <Footer />
      </div>
    </div>
  );
};

export default AppShell;
