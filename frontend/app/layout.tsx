import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "FastH3 queue console",
  description: "Operate or monitor an Infinite Livestream FastH3 session.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
