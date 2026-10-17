# Contributing to NeuroAction

Thank you for your interest in contributing to the **NeuroAction** neural intention–action digital twin research project.

---

## Scientific and Epistemic Standards

NeuroAction adheres to strict epistemic standards to ensure that computational and clinical claims remain honest and empirically grounded:

1. **Strict Held-Out Evaluation**: Models must be evaluated under 5-fold Leave-One-Subject-Out (LOSO) cross-validation. In-sample optimization or test-subject normalization is prohibited.
2. **Kinematic Alignment**: Kinematic prefixes must anchor to **physical kinematic onset** ($v_{\text{hand}} > 50$ mm/s), not the visual go-cue, to account for visuomotor latency.
3. **No Target Fabrication**: OpenNeuro `ds006840` lacks independent physical target survey coordinates; any targets used must be explicitly labeled as empirical proxy centroids.
4. **Distinguish Conditions from Slips**: Claims of detecting involuntary biological motor slips must not be conflated with experimental condition discrimination.

---

## Code Quality and Testing Guidelines

Before submitting contributions:

### 1. Python Scientific Pipeline
- Adhere to PEP 8 standards with type hints where applicable.
- Ensure all 108 automated pytest suites pass:
  ```bash
  PYTHONPATH=. .venv/bin/pytest tests/
  ```

### 2. Next.js Frontend
- Run ESLint and TypeScript compilation:
  ```bash
  cd web/frontend
  npm run lint
  npm run build
  ```
- Ensure the offline demo data (`lib/sub01DemoData.ts`) remains intact and functional for frontend-only deployments.
