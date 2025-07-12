import React from "react";
import { BlockMath, InlineMath } from "react-katex";
import "./styles/ContentStyles.css";

export default function WVAEMethodDescription() {
  return (
    <div className="method-container">
      <h2>
        About the Method: Variational Autoencoder with Optimal Transport Reconstruction Loss
      </h2>

      <p>
        A <strong>Variational Autoencoder (VAE)</strong> is a probabilistic generative model
        that learns to encode data into a latent space and reconstruct it back to the observed space.
        Formally, a VAE models the data likelihood as:
      </p>
      <BlockMath math={"p(x) = \\int p(x|z) \\, p(z) \\, dz"} />
      <p>
        where <InlineMath math={"z"} /> is a latent variable sampled from a prior, typically <InlineMath math={"\\mathcal{N}(0, I)"} />, and <InlineMath math={"p(x|z)"} /> 
        is the decoder likelihood.
      </p>

      <p>
        To train the model, the Evidence Lower Bound (ELBO) is maximized:
      </p>
      <BlockMath math={"\\text{ELBO} = \\mathbb{E}_{q(z|x)} \\big[ \\log p(x|z) \\big] - \\mathrm{KL}\\big( q(z|x) \\| p(z) \\big)"} />
      <p>
        The <strong>reconstruction loss</strong> in many VAEs is the Mean Squared Error (MSE):
      </p>
      <BlockMath math={"\\mathcal{L}_{\\text{rec}} = \\| x - \\hat{x} \\|_2^2"} />
      <p>
        This loss penalizes pointwise Euclidean deviations between the input and reconstruction.
        As a result, when interpolating between latent encodings of two training samples, the decoder will produce outputs that
        are linear blends in Euclidean space.
      </p>
      <p>
        For example, interpolating between a Gaussian blob and a non-Gaussian lumpy shape will yield reconstructions that look
        like blurred mixtures rather than natural morphings.
      </p>

      <h3>Wasserstein Distance: A Geometric Measure</h3>
      <p>
        The <strong>Wasserstein distance</strong>, in contrast, measures the minimal cost to transport mass from one distribution to another.
        This makes it sensitive to the geometry and support of the distributions.
      </p>
      <p>
        Given two probability measures <InlineMath math={"\\mu"} /> and <InlineMath math={"\\nu"} />, the p-Wasserstein distance is defined as:
      </p>
      <BlockMath math={"W_p(\\mu, \\nu) = \\left( \\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\int \\|x - y\\|^p \\, d\\gamma(x,y) \\right)^{1/p}"} />
      <p>
        where <InlineMath math={"\\Pi(\\mu, \\nu)"} /> denotes all couplings (transport plans) between <InlineMath math={"\\mu"} /> and <InlineMath math={"\\nu"} />.
      </p>
      <p>
        Unlike Euclidean MSE, interpolating distributions along the Wasserstein geodesic leads to natural displacement of mass.
        For example, the interpolation between a Gaussian and a non-Gaussian bump would yield intermediate distributions where the mass
        gradually shifts, rather than simply blending pixel intensities.
      </p>

      <h3>The Sinkhorn Algorithm</h3>
      <p>
        Computing the exact Wasserstein distance requires solving a linear program, which is computationally expensive.
        The <strong>Sinkhorn algorithm</strong> provides an efficient approximation by introducing entropy regularization, 
        which turns the optimal transport problem from a linear to a convex optimization problem:
      </p>
      <BlockMath math={"W_{\\varepsilon}(\\mu, \\nu) = \\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\int \\|x - y\\| \\, d\\gamma(x,y) - \\varepsilon \\, H(\\gamma)"} />
      <p>
        where <InlineMath math={"H(\\gamma) = - \\int \\log \\gamma(x,y) \\, d\\gamma(x,y)"} /> is the entropy of the transport plan.
      </p>
      <p>
        In this formulation, the optimal plan <InlineMath math={"\\gamma^*"} /> is expressed implicitly through two potential functions which arise from the dual formulation: 
        <InlineMath math={"u(x)"} /> and <InlineMath math={"v(y)"} />:
      </p>
      <BlockMath math={"\\gamma^*(x,y) = \\exp\\left( \\frac{u(x) + v(y) - c(x,y)}{\\varepsilon} \\right)"} />
      <p>
        where <InlineMath math={"c(x,y) = \\|x - y\\|"} /> is the cost function.
      </p>
      <p>
        These potentials are iteratively updated by a scaling procedure that alternates normalizing over rows and columns of
        the transport matrix. Other approaches directly learn the potentials between two distributions as neural networks, using the Sinkhorn algorithm to guide training.
      </p>

      <h3>Variational Autoencoder with Wasserstein (Sinkhorn) Reconstruction Loss</h3>
      <p>
        In our method, we replace the standard MSE reconstruction loss by a <strong>Sinkhorn Wasserstein distance</strong> between the
        input data and the reconstructed output:
      </p>
      <BlockMath math={"\\mathcal{L}_{\\text{rec}} = W_{\\varepsilon}(x, \\hat{x})"} />
      <p>
        This changes the behavior of the learned latent space.
        Whereas an MSE VAE tends to learn latents whose interpolations correspond to Euclidean averages of pixel intensities,
        a Wasserstein VAE learns latents whose interpolations correspond to <em>McCann interpolations</em>:
      </p>
      <BlockMath math={"P_t = \\bigl( (1 - t)\\, \\text{id} + t \\, T \\bigr)_{\\#} \\, \\mu"} />
      <p>
        Here, <InlineMath math={"T"} /> is the optimal transport map from the input distribution to the reconstruction, and
        <InlineMath math={"P_t"} /> defines a geodesic in Wasserstein space.
      </p>
      <p>
        Consequently, latent interpolations between two samples decode to realistic intermediate shapes, so called Wasserstein barycenters, obtained by progressively
        transporting mass rather than blending intensities.
      </p>
      <p>
        This provides a more natural and semantically meaningful notion of reconstruction and interpolation, especially in
        image or distributional data.
      </p>

      <p>
        <strong>Why is this interesting?</strong>
        This approach makes VAEs aware of the geometry of the data distribution, yielding smoother and more interpretable latent spaces.
        It also connects optimal transport theory with generative modeling, opening research opportunities in
        distribution alignment and shape-based losses.
      </p>

      <p><strong>References and Further Reading:</strong></p>
      <ul>
        <li>Cuturi (2013): <em>Sinkhorn Distances: Lightspeed Computation of Optimal Transport</em>.</li>
        <li>Villani (2009): <em>Optimal Transport: Old and New</em>.</li>
        <li>Arjovsky et al. (2017): <em>Wasserstein GAN</em>.</li>
        <li>Genevay et al. (2018): <em>Learning Generative Models with Sinkhorn Divergences</em>.</li>
        <li>
          <a
            href="https://arxiv.org/abs/1711.01558"
            target="_blank"
            rel="noopener noreferrer"
          >
            📄 Wasserstein Auto-Encoders
          </a>
        </li>
      </ul>
    </div>
  );
}
