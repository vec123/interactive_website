import React, { useState } from "react";
import GPLVMLatentExplorer from "./GPLVM_latent_explorer";
import GPLVMMethodDescription from "./GPLVM";
import WVAEMMethodDescription from "./Wasserstein_VAE";
import "katex/dist/katex.min.css";

function App() {
  const [activeTab, setActiveTab] = useState("GPLVM");

  return (
    <div>
      {/* Navigation Bar */}
      <nav style={{
        display: "flex",
        justifyContent: "center",
        gap: "2rem",
        padding: "1rem",
        background: "#f0f0f0",
        borderBottom: "1px solid #ddd",
        fontFamily: "Arial, sans-serif"
      }}>
        <button
          onClick={() => setActiveTab("GPLVM")}
          style={{
            padding: "0.5rem 1rem",
            border: "none",
            background: activeTab === "GPLVM" ? "#ccc" : "transparent",
            cursor: "pointer",
            fontWeight: activeTab === "GPLVM" ? "bold" : "normal"
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
            fontWeight: activeTab === "WassersteinVAE" ? "bold" : "normal"
          }}
        >
          Wasserstein VAE
        </button>
      </nav>

      {/* Content */}
      <div>
        {activeTab === "GPLVM" && (
          <div>
            <GPLVMMethodDescription />
            <GPLVMLatentExplorer />
          </div>
        )}
        {activeTab === "WassersteinVAE" && (
          <WVAEMMethodDescription />
        )}
      </div>
    </div>
  );
}

export default App;
