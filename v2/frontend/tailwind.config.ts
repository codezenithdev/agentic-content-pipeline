import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0D0D0F",
        panel: "#16161A",
        elevated: "#1C1C22",
        line: "#26262C",
        paper: "#F0EDE6",
        muted: "#9A968C",
        teal: "#2DD4BF",
        amber: "#F59E0B",
        coral: "#F87171",
        ok: "#34D399",
      },
      fontFamily: {
        sans: ["var(--font-dm-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-plex-mono)", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
