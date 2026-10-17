import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        /* Display / Headings → Poppins Bold */
        display: ["var(--font-poppins)", "Poppins", "system-ui", "sans-serif"],
        heading: ["var(--font-poppins)", "Poppins", "system-ui", "sans-serif"],
        poppins: ["var(--font-poppins)", "Poppins", "system-ui", "sans-serif"],
        /* Body / Labels → Open Sans */
        body: ["var(--font-opensans)", "Open Sans", "system-ui", "sans-serif"],
        sans: ["var(--font-opensans)", "Open Sans", "system-ui", "sans-serif"],
        /* Legacy aliases — both point to their Poppins/OS counterpart */
        serif: ["var(--font-poppins)", "Poppins", "system-ui", "sans-serif"],
        mono: ["var(--font-opensans)", "Open Sans", "system-ui", "sans-serif"],
      },
      colors: {
        ivory: "#F5F3EE",
        "ivory-subtle": "#FAF9F5",
        surface: "#FFFFFF",
        "surface-inset": "#EDECE6",
        charcoal: "#1C1C1A",
        muted: "#6B6B66",
        forest: {
          DEFAULT: "#3A5244",
          hover: "#2D4036",
          subtle: "#E4EDE6",
        },
        border: {
          DEFAULT: "#D9D8D0",
          strong: "#AEADA4",
        },
        eegCyan: "#2563EB",
        goalBlue: "#1D4ED8",
        trajGreen: "#2E6B47",
        forecastViolet: "#6B46C1",
        errorOrange: "#B45309",
        grayNeutral: "#6B6B66",
        navyBg: "#F5F3EE",
      },
    },
  },
  plugins: [],
};

export default config;
