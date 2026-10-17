import type { Metadata } from 'next';
import { Poppins, Open_Sans } from 'next/font/google';
import Navbar from '@/components/nav/Navbar';
import { BackendProvider } from '@/lib/backend-context';
import BackendNoticeModal from '@/components/modals/BackendNoticeModal';
import './globals.css';

/* Headings / display: Poppins Bold */
const poppins = Poppins({
  variable: '--font-poppins',
  subsets: ['latin'],
  display: 'swap',
  weight: ['400', '500', '600', '700', '800', '900'],
});

/* Body / descriptions: Open Sans */
const openSans = Open_Sans({
  variable: '--font-opensans',
  subsets: ['latin'],
  display: 'swap',
  weight: ['300', '400', '500', '600', '700'],
});

export const metadata: Metadata = {
  title: 'NeuroAction — Neural Intention–Action Research Project',
  description:
    'An offline multimodal research prototype investigating pre-movement EEG goal decoding and 3D limb trajectory forecasting.',
  icons: {
    icon: [
      { url: '/icon.svg', type: 'image/svg+xml' },
      { url: '/favicon.ico', sizes: 'any' },
    ],
    apple: '/apple-touch-icon.png',
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${poppins.variable} ${openSans.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-[#F5F3EE] text-[#1C1C1A] font-body selection:bg-[#3A5244]/20">
        <BackendProvider>
          <Navbar />
          <main className="flex-1 w-full flex flex-col">{children}</main>
          <BackendNoticeModal />
        </BackendProvider>
      </body>
    </html>
  );
}
