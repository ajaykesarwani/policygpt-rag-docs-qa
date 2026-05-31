import "./globals.css";
import type { ReactNode } from "react";

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="app-shell">
          <div className="app-shell__inner">{children}</div>
        </div>
      </body>
    </html>
  );
}