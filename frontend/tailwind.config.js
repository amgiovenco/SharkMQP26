/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Nunito', 'ui-sans-serif', 'system-ui'],
      },
      colors: {
        pelagia: {
          white: "#F9F7F7",
          mist: "#9FB6C4",
          aqua: "#44B3D3",
          slate: "#7F848D",
          navy: "#35557B",
          pure: "#FFFEFE",
          lilac: "#9397B6",
          lightblue: "#A2C4D7",
          periwinkle: "#555EAC",
          deepblue: "#333C80",
          darknavy: "#183150",
          blue: "#3B4FFF",
          cyan: "#0CB6FF",
          inputborder: "#9D9D9D",
        },
      },
      backgroundImage: {
        'gradient-login': 'linear-gradient(90deg, #3B4FFF 0%, #0CB6FF 35.58%, #44B3D3 69.71%, #9FB6C4 98.08%)',
      },
    },
  },
  plugins: [],
};