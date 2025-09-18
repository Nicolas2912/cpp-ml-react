// client/tailwind.config.js

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}", // Ensure this covers your components
    "./public/index.html",
  ],
  theme: {
    extend: {}, // You can add theme extensions here if needed
  },
  // Add DaisyUI to the plugins array
  plugins: [require("daisyui")],

  // DaisyUI configuration (optional - themes are defined here)
  daisyui: {
    themes: ["light", "dark", "cupcake", "synthwave"], // Your chosen themes
    // other daisyui config options...
    // styled: true,
    // base: true,
    // utils: true,
    // logs: true,
    // rtl: false,
    // prefix: "",
    // darkTheme: "dark",
  },
}