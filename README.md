# Autoencoders vs PCA — A Modern Replication of Hinton & Salakhutdinov (2006)

This project replicates the core empirical claim of  **Hinton & Salakhutdinov (2006), _Reducing the Dimensionality of Data with Neural Networks_**,  using modern tools and training practices.

The original paper showed that nonlinear autoencoders can achieve lower reconstruction error than PCA when compressing high-dimensional data. In this project, we compare a standard PCA baseline with a multilayer perceptron autoencoder under matched latent dimensionality constraints.

📄 **Technical Report:**  
[Revisiting Dimensionality Reduction with Autoencoders — A Modern Empirical Replication (PDF)](report/Revisiting_Dimensionality_Reduction_with_Autoencoders__A_Modern_Empirical_Replication.pdf)

## Dataset
- MNIST (28×28 grayscale images)

## Experiments
- Methods: PCA, MLP Autoencoder  
- Latent dimensions: 2, 16, 32, 64  
- Metric: mean squared reconstruction error  
- Qualitative comparison via reconstruction visualizations

## Structure
- `src/` — core implementations (data loading, models, training, baselines)
- `scripts/` — experiment runners
- `notebooks/` — analysis and figure generation
- `results/` — metrics and visual outputs
- `report/` — LaTeX write-up of results

## Running the code
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m scripts.sweep_latents
python -m scripts.make_figures

