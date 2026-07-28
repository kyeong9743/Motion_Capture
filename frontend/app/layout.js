import "./globals.css";

export const metadata = {
  title: "MediaMotion — 3D Motion Capture",
  description: "Single-video 3D motion capture, locally on your GPU.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
