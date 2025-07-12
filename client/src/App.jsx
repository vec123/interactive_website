import React, { useState } from "react";
import GPLVMMethodDescription from "./GPLVM";
import WVAEMethodDescription from "./Wasserstein_VAE";
import "katex/dist/katex.min.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("GPLVM");

  return (
    <div
      style={{
        // Make the WHOLE page content centered in a 900px box
        maxWidth: "900px",
        margin: "0 auto",
        fontFamily: "Arial, sans-serif",
      }}
    >
      {/* Navigation Bar */}
      <nav
        style={{
          display: "flex",
          justifyContent: "center",
          gap: "2rem",
          padding: "1rem",
          background: "#f0f0f0",
          borderBottom: "1px solid #ddd",
        }}
      >
        <button
          onClick={() => setActiveTab("GPLVM")}
          style={{
            padding: "0.5rem 1rem",
            border: "none",
            background: activeTab === "GPLVM" ? "#ccc" : "transparent",
            cursor: "pointer",
            fontWeight: activeTab === "GPLVM" ? "bold" : "normal",
          }}
        >
          GPLVM
        </button>
        <button
          onClick={() => setActiveTab("WassersteinVAE")}
          style={{
            padding: "0.5rem 1rem",
            border: "none",
            background: activeTab === "WassersteinVAE" ? "#ccc" : "transparent",
            cursor: "pointer",
            fontWeight: activeTab === "WassersteinVAE" ? "bold" : "normal",
          }}
        >
          Wasserstein VAE
        </button>
      </nav>

      {/* Content */}
      <div>
        {activeTab === "GPLVM" && (
          <GPLVMMethodDescription />
        )}
        {activeTab === "WassersteinVAE" && (
          <WVAEMethodDescription />
        )}
      </div>
    </div>
  );
}
