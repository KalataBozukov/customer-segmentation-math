# Customer Segmentation & Geometric Data Analysis

## Project Overview
In the modern business landscape, understanding customer behavior is crucial for sustainable growth. This project implements a full-stack Data Science pipeline to solve the challenge of **Customer Segmentation** – partitioning a company’s client base into distinct, statistically significant groups based on financial behavior.

## Objectives
* **Mathematical Framework:** Translate raw financial transactions into a high-dimensional vector space.
* **Geometric Clustering:** Implement and analyze the **K-Means** algorithm through the lens of Euclidean distance and inertia minimization.
* **Latent Space Mapping:** Utilize **PCA (Principal Component Analysis)** to reduce dimensionality while preserving global variance.
* **Statistical Validation:** Use **ANOVA** and **Eta-squared ($\eta^2$)** to prove the distinctness of the identified segments.

## Tech Stack & Methodology
This project follows a rigorous scientific approach across five specialized modules:

1.  **Theory & Simulation:** Validating K-Means logic using synthetic "Ground Truth" datasets (`make_blobs`).
2.  **Dimensionality Reduction:** Handling 17+ financial features using **Log-Transformation** and **PCA** to capture latent structures.
3.  **Model Optimization:** Selecting the optimal $k$ via **Elbow Method (WCSS)** and **Silhouette Analysis**.
4.  **Persona Decoding:** Mapping cluster centroids back to real-world financial metrics (Balance, Purchases, Cash Advance) using **Lift Analysis**.
5.  **Business Intelligence:** Translating statistical signatures into actionable marketing strategies and policy recommendations.

## Business Significance
By transitioning from raw data to geometric segments, the project enables:
1.  **Precision Marketing:** Tailored campaigns based on "Persona Lifts" (e.g., distinguishing "High-Spending Transactors" from "Cash Advance Dependent" users).
2.  **Resource Allocation:** Identifying high-value segments to optimize credit limit increases and loyalty rewards.
3.  **Risk Mitigation:** Early detection of at-risk behavior through cluster-specific churn indicators and payment patterns.

## Project Structure
* `01_simulation_and_theory.ipynb` – The geometry of K-Means and Euclidean distance.
* `02_real_world_analysis.ipynb` – Data preprocessing, Scaling, and PCA projection.
* `03_comperative_results.ipynb` – Quantitative model selection and cluster validation.
* `04_cluster_profiling_and_FPD.ipynb` – Statistical profiling via ANOVA and Persona decoding.
* `05_bussiness_strategy_and_conclusions.ipynb` – Strategic decision-making and final ROI-focused insights.
  
