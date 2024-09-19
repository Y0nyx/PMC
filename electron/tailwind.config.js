/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  corePlugins: {
       preflight: false,
      },
  theme: {
    extend: {
      fontFamily:{
       "normal": ['Chakra Petch', 'sans-serif']
      },
    },
  },
  plugins: [],
}

