const express = require("express");
const path = require("path");
const app = express();

// API endpoint
app.get("/api/message", (req, res) => {
  res.json({ message: "Hello from the server!" });
});

// Serve frontend build
app.use(express.static(path.join(__dirname, "../client/dist")));
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "../client/dist/index.html"));
});

// Start server
const port = process.env.PORT || 5000;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
