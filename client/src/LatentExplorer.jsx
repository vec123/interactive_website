import { useEffect, useState } from "react";
import Plot from "react-plotly.js";

function LatentExplorer({ modelId, modelType, description }) {
  const [latentPoints, setLatentPoints] = useState([]);
  const [output, setOutput] = useState(null);
  const [axisRange, setAxisRange] = useState({
    xMin: -3,
    xMax: 3,
    yMin: -3,
    yMax: 3,
  });
  const [densityGrid, setDensityGrid] = useState(null);

  // Density helper (same as before)
  function computeDensityGrid(latentPoints, axisRange, step = 0.1, sigma = 0.5) {
    const xs = [];
    for (let x = axisRange.xMin; x <= axisRange.xMax; x += step) xs.push(x);
    const ys = [];
    for (let y = axisRange.yMin; y <= axisRange.yMax; y += step) ys.push(y);

    const z = ys.map((y) =>
      xs.map((x) => {
        let sum = 0;
        for (const [px, py] of latentPoints) {
          const dx = x - px;
          const dy = y - py;
          const r2 = dx * dx + dy * dy;
          sum += Math.exp(-r2 / (2 * sigma * sigma));
        }
        return sum;
      })
    );

    return { xs, ys, z };
  }

  // Load latent points dynamically
  useEffect(() => {
    setOutput(null);
    fetch(`/encodings/${modelId}.json`)
      .then((res) => res.json())
      .then((data) => {
        setLatentPoints(data);

        const xs = data.map((p) => p[0]);
        const ys = data.map((p) => p[1]);
        const margin = 3.0;
        const newRange = {
          xMin: Math.min(...xs) - margin,
          xMax: Math.max(...xs) + margin,
          yMin: Math.min(...ys) - margin,
          yMax: Math.max(...ys) + margin,
        };
        setAxisRange(newRange);

        const density = computeDensityGrid(data, newRange, 0.05, 0.4);
        setDensityGrid(density);
      });
  }, [modelId]);

  // Request generation
  const sendRequest = async (x, y) => {
    try {
      const response = await fetch(`/api/models/${modelId}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ latent: [x, y] }),
      });
      const data = await response.json();
      setOutput(data);
    } catch (error) {
      console.error("Error fetching output:", error);
    }
  };

  const handleHover = (event) => {
    const point = event.points[0];
    sendRequest(point.x, point.y);
  };

  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        gap: "2rem",
        padding: "1rem",
      }}
    >
      {/* Latent Space */}
      <div style={{ width: "600px", maxWidth: "100%", textAlign: "center" }}>
        <h3>Latent Space (hover anywhere)</h3>
        {densityGrid && (
          <Plot
            data={[
              {
                z: densityGrid.z,
                x: densityGrid.xs,
                y: densityGrid.ys,
                type: "heatmap",
                colorscale: [
                  [0, "black"],
                  [1, "white"],
                ],
                reversescale: true,
                showscale: false,
                hoverinfo: "x+y",
              },
              {
                x: latentPoints.map((p) => p[0]),
                y: latentPoints.map((p) => p[1]),
                mode: "markers",
                marker: {
                  color: "white",
                  size: 6,
                  line: { width: 1, color: "black" },
                },
                name: "Latents",
              },
            ]}
            layout={{
              width: 600,
              height: 600,
              dragmode: false,
              hovermode: "closest",
              xaxis: { range: [axisRange.xMin, axisRange.xMax], fixedrange: true },
              yaxis: { range: [axisRange.yMin, axisRange.yMax], fixedrange: true },
              margin: { t: 20, b: 40, l: 40, r: 20 },
            }}
            config={{
              displayModeBar: false,
              staticPlot: false,
            }}
            onHover={handleHover}
          />
        )}
        <p style={{ fontSize: "0.9em", marginTop: "0.5em" }}>{description}</p>
      </div>

      {/* Output */}
      <div style={{ width: "600px", maxWidth: "100%", textAlign: "center" }}>
        <h3>Model Output</h3>
        {output ? (
          modelType === "skeleton" ? (
            <svg width="600" height="600" style={{ border: "1px solid #ccc" }}>
              {output.edges.map(([i, j], idx) => {
                if (!output.joints[i] || !output.joints[j]) return null;

                const [xi_raw, yi_raw, zi_raw] = output.joints[i];
                const [xj_raw, yj_raw, zj_raw] = output.joints[j];

                const xi = xi_raw + zi_raw * 0.5;
                const yi = yi_raw - zi_raw * 0.2;
                const xj = xj_raw + zj_raw * 0.5;
                const yj = yj_raw - zj_raw * 0.2;

                const scale = 5;
                const offsetX = 300;
                const offsetY = 300;

                return (
                  <line
                    key={idx}
                    x1={xi * scale + offsetX}
                    y1={-yi * scale + offsetY}
                    x2={xj * scale + offsetX}
                    y2={-yj * scale + offsetY}
                    stroke="black"
                    strokeWidth="2"
                  />
                );
              })}
            </svg>
          ) : modelType === "image" ? (
            <img
      src={`data:image/png;base64,${output.image}`}
      alt="Generated"
      style={{
        width: "600px",
        height: "600px",
        border: "1px solid #ccc",
        objectFit: "contain"
  }}
/>
          ) : (
            <p>Unknown model type.</p>
          )
        ) : (
          <p>Hover over the latent space to see output.</p>
        )}
      </div>
    </div>
  );
}

export default LatentExplorer;
