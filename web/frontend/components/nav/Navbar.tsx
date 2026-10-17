'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import NeuroActionLogo from '@/components/icons/NeuroActionLogo';

interface NavItem {
  name: string;
  href: string;
}

const navItems: NavItem[] = [
  { name: 'Overview', href: '/' },
  { name: 'Science', href: '/science' },
  { name: 'Model Lab', href: '/model-lab' },
  { name: '3D Twin', href: '/digital-twin' },
  { name: 'Explainability', href: '/explainability' },
  { name: 'Evidence', href: '/evidence' },
];

export default function Navbar() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const isActive = (href: string) => {
    if (href === '/') return pathname === '/';
    return pathname.startsWith(href);
  };

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-[#D9D8D0] bg-[#F5F3EE]/96 backdrop-blur-md">
      <div className="max-w-6xl mx-auto px-6 lg:px-10">
        <div className="flex items-center justify-between h-16">

          {/* Brand Logo & Wordmark */}
          <Link
            href="/"
            className="flex items-center gap-2.5 focus:outline-none group"
            aria-label="NeuroAction Home"
          >
            <NeuroActionLogo size={34} className="shrink-0 transition-transform duration-200 group-hover:scale-105 shadow-xs rounded-xl" />
            <span className="text-xl font-bold tracking-tight text-[#1C1C1A] group-hover:text-[#3A5244] transition-colors duration-200">
              NeuroAction
            </span>
          </Link>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-0.5">
            {navItems.map((item) => {
              const active = isActive(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'relative px-3.5 py-1.5 text-sm tracking-wide transition-colors duration-200',
                    active
                      ? 'text-[#1C1C1A] font-semibold'
                      : 'text-[#6B6B66] hover:text-[#1C1C1A] font-normal'
                  )}
                >
                  {item.name}
                  {active && (
                    <span className="absolute bottom-[-1px] left-2 right-2 h-px bg-[#3A5244]" />
                  )}
                </Link>
              );
            })}
          </div>

          {/* Mobile menu toggle */}
          <div className="flex items-center md:hidden">
            <button
              type="button"
              className="p-2 rounded text-[#6B6B66] hover:text-[#1C1C1A] hover:bg-[#EDECE6] transition-colors"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-controls="mobile-menu"
              aria-expanded={mobileMenuOpen}
            >
              <span className="sr-only">Menu</span>
              {mobileMenuOpen
                ? <X className="w-5 h-5" aria-hidden="true" />
                : <Menu className="w-5 h-5" aria-hidden="true" />
              }
            </button>
          </div>
        </div>
      </div>

      {/* Mobile dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-[#F5F3EE] border-b border-[#D9D8D0] px-6 pt-1 pb-4 space-y-0.5">
          {navItems.map((item) => {
            const active = isActive(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className={cn(
                  'block px-3 py-2.5 rounded-md text-sm tracking-wide transition-colors',
                  active
                    ? 'bg-[#EDECE6] text-[#3A5244] font-semibold'
                    : 'text-[#6B6B66] hover:text-[#1C1C1A] hover:bg-[#EDECE6]'
                )}
              >
                {item.name}
              </Link>
            );
          })}
        </div>
      )}
    </nav>
  );
}
