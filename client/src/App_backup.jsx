import React, { useState } from "react";
import GPLVMMethodDescription from "./GPLVM";
import WVAEMethodDescription from "./WassersteinVAE";
import GPLVMLatentExplorer from "./GPLVMLatentExplorer";
import "katex/dist/katex.min.css";
import "./styles/ContentStyles.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("GPLVM");

  return (
    <main>
      {/* Navigation Bar */}
      <nav
        style={{
          width: "100%",
          background: "#f0f0f0",
          borderBottom: "1px solid #ddd",
        }}
      >
        <div
          style={{
            maxWidth: "900px",
            margin: "0 auto",
            display: "flex",
            justifyContent: "center",
            gap: "2rem",
            padding: "1rem",
            fontFamily: "Arial, sans-serif",
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
        </div>
      </nav>

      {/* Main Content */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          padding: "2rem 1rem",
        }}
      >
      <div
        style={{
          width: "100%",
          fontFamily: "Arial, sans-serif",
        }}
      >
          {activeTab === "GPLVM" && (
            <>
              <GPLVMMethodDescription />
              <GPLVMLatentExplorer />
            </>
          )}
          {activeTab === "WassersteinVAE" && <WVAEMethodDescription />}
        </div>
      </div>
    </main>
  );
}
