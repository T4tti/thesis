"use client";

import React from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { BarChart3, BookOpen, Database, Home, X } from "lucide-react";
import { useSidebar } from "@/context/SidebarContext";
import { useLanguage } from "@/context/LanguageContext";

const AppSidebar: React.FC = () => {
  const pathname = usePathname();
  const { isExpanded, isMobileOpen, isHovered, setIsHovered, toggleMobileSidebar, closeMobileSidebar } = useSidebar();
  const { t } = useLanguage();

  const isActive = (path: string) => (path === "/" ? pathname === "/" : pathname.startsWith(path));

  const navItems = [
    { name: t.nav.home, icon: <Home className="h-5 w-5" />, path: "/" },
    { name: t.nav.ratingTool, icon: <BarChart3 className="h-5 w-5" />, path: "/rating-tool" },
    { name: t.nav.reports, icon: <Database className="h-5 w-5" />, path: "/reports" },
    { name: t.nav.methodology, icon: <BookOpen className="h-5 w-5" />, path: "/methodology" },
  ];

  const expandedClass = isExpanded || isHovered || isMobileOpen ? "w-[280px]" : "w-[84px]";

  return (
    <aside
      id="app-sidebar"
      className={`fixed left-0 top-0 z-40 mt-16 flex h-screen flex-col border-r border-gray-200 bg-white px-4 text-gray-900 transition-all duration-300 ease-in-out dark:border-gray-800 dark:bg-gray-900 lg:mt-0 lg:px-5 ${expandedClass} ${isMobileOpen ? "translate-x-0" : "-translate-x-full"} lg:translate-x-0`}
      onMouseEnter={() => !isExpanded && setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      aria-label="Primary navigation"
    >
      <div className="flex items-center justify-end py-3 lg:hidden">
        <button
          type="button"
          onClick={closeMobileSidebar}
          aria-label="Close menu"
          className="rounded-lg p-2 text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800 dark:hover:text-white"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className={`flex py-6 ${!isExpanded && !isHovered ? "lg:justify-center" : "justify-start"}`}>
        <Link href="/" className="flex items-center gap-3" aria-label="VN-Rating home">
          {(isExpanded || isHovered || isMobileOpen) ? (
            <>
              <Image
                className="dark:hidden"
                src="/images/logo/logo.svg"
                alt="Logo"
                width={140}
                height={36}
              />
              <Image
                className="hidden dark:block"
                src="/images/logo/logo-dark.svg"
                alt="Logo"
                width={140}
                height={36}
              />
            </>
          ) : (
            <Image src="/images/logo/logo-icon.svg" alt="Logo" width={32} height={32} />
          )}
        </Link>
      </div>

      <div className="no-scrollbar flex flex-col overflow-y-auto pb-8">
        <nav className="mb-6" aria-label="Main">
          <span
            className={`mb-4 flex text-xs uppercase leading-[20px] text-gray-400 ${
              !isExpanded && !isHovered ? "lg:justify-center" : "justify-start"
            }`}
          >
            {isExpanded || isHovered || isMobileOpen ? "Menu" : "\u2026"}
          </span>
          <ul className="flex flex-col gap-2">
            {navItems.map((item) => {
              const active = isActive(item.path);
              return (
                <li key={item.path}>
                  <Link
                    href={item.path}
                    onClick={() => {
                      if (isMobileOpen) {
                        toggleMobileSidebar();
                      }
                    }}
                    aria-current={active ? "page" : undefined}
                    className={`group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                      active
                        ? "bg-brand-50 text-brand-500 dark:bg-brand-500/15 dark:text-brand-400"
                        : "text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-white/5"
                    } ${!isExpanded && !isHovered ? "lg:justify-center" : "lg:justify-start"}`}
                  >
                    <span className={active ? "text-brand-500 dark:text-brand-400" : "text-gray-500 dark:text-gray-400"}>
                      {item.icon}
                    </span>
                    {(isExpanded || isHovered || isMobileOpen) && <span>{item.name}</span>}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
      </div>
    </aside>
  );
};

export default AppSidebar;
