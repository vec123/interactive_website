# Interactive GPLVM Website

This project is an interactive website that:
- Visualizes a Gaussian Process Latent Variable Model (GPLVM)
- Has a **React frontend**
- Has a **FastAPI backend** serving model predictions

## 🚀 Deployment on Fly.io

This guide walks you through deploying the site to Fly.io so it runs online without requiring your machine.

---

## 📝 Prerequisites

✅ You need:

- A [Fly.io account](https://fly.io/)
- Docker installed and working (`docker build` works locally)
- `flyctl` installed and in your PATH  
  Install with PowerShell (Windows):

  ```powershell
  iwr https://fly.io/install.ps1 -useb | iex