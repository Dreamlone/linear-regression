import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/linear-regression/interactive/",
  plugins: [react()],
});