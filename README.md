This project is an end-to-end financial security platform that models realistic transaction profiles to combat extreme class imbalance (0.17% fraud rate). Using SMOTE, adaptive thresholding, and concurrent champion-challenger modeling, it stops revenue drainage while generating sub-100ms game-theoretic SHAP explanations to give fraud investigators absolute clarity on live transaction risk flags.

### Core Platform Features

* **Behavioral Simulation Engine:** Replicates 100K+ transactional cardholder profiles tracking complex fraudulent velocity and location anomalies.
* **Imbalanced Data Mitigation:** Uses SMOTE and class-weight scaling to eliminate algorithm bias in extreme minority target spaces.
* **Asymmetric Risk Optimization:** Shifts the decision threshold to 0.35 to maximize fraud capture (Recall) while reducing false-alarm customer friction.
* **Concurrent A/B Experimentation:** Processes real-time traffic across a production Champion model and a shadow Challenger model simultaneously to track data drift.
* **Local SHAP Explainability:** Translates complex mathematical tree decisions into distinct, feature-level attribution graphs for security analysts.
* **Self-Healing Cloud Deployment:** Built as a lightweight, zero-bloat pipeline that dynamically provisions, builds, and caches the ML state instantly on initialization.
Access this application :- https://financial-fraud-detection-p9e2smuaichwtrpzxskpha.streamlit.app/

<img width="1919" height="846" alt="image" src="https://github.com/user-attachments/assets/49f3fadc-1d59-4068-aa4e-e985775b6446" />

<img width="1919" height="884" alt="image" src="https://github.com/user-attachments/assets/4d09d95e-0790-47a7-8617-5724a060f6db" />
