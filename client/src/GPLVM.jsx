import React from "react";
import { BlockMath, InlineMath } from "react-katex";

export default function GPLVMMethodDescription() {
  return (
    <div style={{ textAlign: "center", maxWidth: "900px", margin: "0 auto" }}>
      <h2>About the Method: Gaussian Process Latent Variable Models</h2>
      <p>
        Here, we explore the fascinating world of <strong>Gaussian Processes (GPs)</strong>, a powerful framework for modeling functions.
        Our exploration focuses on <strong>Gaussian Process Latent Variable Models (GPLVMs)</strong>, 
        which extend GPs to learn low-dimensional representations of high-dimensional data.
        Our data will be a sequence of motion frames, where each frame is a 36 dimensional vector (i.e. 12 joints, 3 coordinates).
        Gaussian Processes are cool because they allow building non-parametric model and doing principled inference on functions directly from observations. 
        This makes them flexible models with many applications, among which, data-imputation, dimensionality reduction, GP state space models and more.
        A downside is their computational complexity, which scales cubically with the number of data points.
        Methods which scale better include sparse approximations, inducing points, and variational inference techniques.
        The interactive visualization demonstrates a{" "}
        <strong>Bayesian Gaussian Process Latent Variable Model (bGPLVM)</strong>.
        A GPLVM is a generative probabilistic model that relates a low-dimensional latent space to high-dimensional observed data
        through a Gaussian Process (GP) prior over functions. 
      </p>

      <p>
        Formally, if we denote latent variables as <InlineMath math="Z"/> and observed data as <InlineMath math="X"/>, the model is:
      </p>

      <BlockMath math={`X = f(Z) + \\varepsilon`} />
      <BlockMath math={`\\varepsilon \\sim \\mathcal{N}(0, \\sigma^2 I)`} />
      <BlockMath math={`f \\sim \\mathcal{GP}(0, k(\\cdot, \\cdot))`} />

      <p>
        This can be interpreted as a latent variable model where the marginal likelihood integrates out the functions:
      </p>

      <BlockMath math={`p(X | Z) = \\int p(X | f, Z) \\; p(f) \\; df`} />

      <p>
        Since this integral is intractable in general, variational inference techniques are applied.
        In variational GPLVMs, an approximate posterior over the latent variables and function values is optimized by maximizing the
        <strong> Evidence Lower Bound (ELBO)</strong>, which balances data fit and complexity:
      </p>

      <BlockMath math={`\\text{ELBO} = \\mathbb{E}_q \\big[ \\log p(X | f, Z) \\big] - \\mathrm{KL}\\big( q(f, Z) \\;\\|\\; p(f, Z) \\big)`} />

      <p>
        The ELBO allows learning both the latent coordinates <InlineMath math="Z"/> and kernel hyperparameters jointly. 
        However, it is only a lower bound for the log-likelihood and thus only a surrogate objective. 
      </p>

      <p>
        The kernel function <InlineMath math="k(\cdot, \cdot)"/> is central: it defines a covariance structure encoding assumptions about correlations
        and smoothness, implicitely defining the function space over which optimization occurs.
        This important perspective connecting function spaces and kernel parameterizations 
        comes from the theory of <strong>Reproducing Kernel Hilbert Spaces (RKHS)</strong>.
        In this view, the kernel defines an inner product between functions:
      </p>

      <BlockMath math={`k(z, z') = \\langle \\varphi(z), \\varphi(z') \\rangle`} />

      <p>
        The inner producs is a projection from one function to another. This appears also in Fourier transforms (where one projects to periodic basis functions) 
        and many other formalisms. The frequency truncated Fourier transform has the reproducing property. 
        This is related to convolution with the sinc kernel (which is equivalent to applying the truncated Fourier transform) and Paley-Wiener spaces
        (ignore this, it is not important).
        This reproducing property implies that any function in the space can be expressed as a linear combination of kernels evaluated at training points:
      </p>

      <BlockMath math={`f(z) = \\sum_i \\alpha_i \\; k(z_i, z)`} />

      <p>
        This property connects Gaussian Processes to kernel methods and provides a rich geometric interpretation: the model learns a distribution
        over functions in an infinite-dimensional feature space. 
      
      </p>

      <p>
        In the visualizations on this page, every point you sample in the latent 2D space corresponds to evaluating the learned GP mapping, giving you
        a reconstructed motion frame.
      </p>

      <p>
        <strong>Why is this interesting?</strong> For students, GPLVMs illustrate how Gaussian Processes can serve as flexible decoders,
        generalizing PCA to nonlinear manifolds, with deep similarities to variational auto-encoders.  
        This makes the model interesting as a non-linear dimensionality reduction technique that captures complex data structures
        while providing uncertainty estimates and allowing for interpretations through its generative capabilities.
        This last part is often missing in other non-linear dimensionality reduction methods like t-SNE or UMAP.
        For researchers, they provide a probabilistic generative model with uncertainty quantification
        in both latent representations and reconstructions.
      </p>

      <p>
        <strong>About the implementation: </strong>
        This demo uses <code>gpytorch</code>, a modern PyTorch-based library for Gaussian Processes.
      </p>
      <ul style={{ textAlign: "left", display: "inline-block" }}>
        <li>
          It integrates tightly with PyTorch, enabling automatic differentiation, GPU acceleration, and batching.
        </li>
        <li>
          It provides scalable variational inference methods, such as stochastic variational Gaussian Processes (SVGP) and variational GPLVMs.
        </li>
        <li>
          Its modular design makes it straightforward to define custom kernels, likelihoods, and models.
        </li>
      </ul>
      <p>
        In research and applications, <code>gpytorch</code> has become a popular choice for combining GPs with deep learning workflows and large datasets.
      </p>

      <p><strong>To learn more:</strong></p>
      <ul style={{ textAlign: "left", display: "inline-block" }}>
        <li>Rasmussen & Williams: <em>Gaussian Processes for Machine Learning</em>.</li>
        <li>Titsias & Lawrence (2010): <em>Bayesian Gaussian Process Latent Variable Model</em>.</li>
        <li>Lawrence (2005): <em>Probabilistic Non-linear PCA with Gaussian Process Latent Variable Models</em>.</li>
        <li>Aronszajn (1950): <em>The Theory of Reproducing Kernels</em>.</li>
        <li>
          <a
            href="https://www.youtube.com/watch?v=DS853uA0u4I&t=2418s"
            target="_blank"
            rel="noopener noreferrer"
          >
            📺 Youtube Video: Probabilistic Dimensional Reduction with Gaussian Process Latent Variable Model
          </a>
        </li>
        <li>
          <a
            href="https://www.youtube.com/watch?v=pAZxfwo6efg&t=4871s"
            target="_blank"
            rel="noopener noreferrer"
          >
            📺 Youtube Video: Neil Lawrence: Latent Variable Models with Gaussian Processes
          </a>
        </li>
      </ul>
    </div>
  );
}
