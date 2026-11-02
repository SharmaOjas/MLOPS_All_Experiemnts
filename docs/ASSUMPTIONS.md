# Assumptions, limitations and design decisions

## Data and labels
- **Subset:** FD001 (one operating condition, one fault mode: HPC degradation). `src/data.py` loads FD002-FD004
  the same way, and the Dataset Explorer can browse them.
- **RUL cap = 125 cycles.** Early in life, degradation can't be seen in the sensors, so the training target is
  capped (a common choice for C-MAPSS). The model therefore never predicts more than 125. Evaluation uses the
  **uncapped** true RUL from `RUL_FD001.txt`, so engines with a true RUL > 125 show as under-predicted. This
  is conservative, and it is visible in the *Model Evaluation* page.
- **Test labels:** `RUL_FD001.txt` gives the RUL after the last observed cycle of each test engine. Earlier
  cycles get `RUL + (last_cycle − cycle)`.
- **Dropped sensors:** sensors with training std ≤ 1e-4 are dropped (FD001: s1, s5, s10, s16, s18, s19). The
  list is computed from data, not hard-coded, and is stored in the model bundle.
- **Operational settings** are near-constant in FD001 and are not used as features.

## Features
- Window = 10 cycles. Per sensor: raw value, rolling mean, rolling std, least-squares slope, and delta from
  the engine's own early-life baseline (mean of its first 10 cycles, computed causally). Plus `cycle`.
  That gives 76 features.
- Short histories work: rolling statistics use `min_periods=1`, and std/slope fall back to 0.
- **Uploaded histories should start at cycle 1.** The baseline feature assumes the first rows are early life.
  A history that starts mid-life still gets a prediction, but the delta features are less informative.

## Model selection
- GroupKFold(5) by engine on the 100 training engines. The selection metric is `cv_rmse`. The official test
  set is used **only** for reporting.
- Classic GradientBoosting was reduced to 150 shallow trees so CI finishes in reasonable time. It scores a
  slightly lower test RMSE than the selected HistGradientBoosting, but it was not selected, because selection
  is by CV.
- Deep-learning models (LSTM/CNN) are deliberately left out in favour of reproducible scikit-learn regressors.

## Serving and monitoring
- **Operational monitoring:** request counts, status codes, errors by type, latency histograms, and
  model-loaded status. This only says whether the service works, not whether its predictions are good.
- **Input drift:** PSI of the **latest cycle of each request** against the training distribution
  (10 quantile bins per sensor, rolling buffer of the last 500 requests, computed once ≥ 50 are buffered).
  Measured baseline: genuine FD001 test traffic already gives a median PSI of ~0.15-0.3, because test engines
  are younger than the run-to-failure training fleet. The alert is calibrated to a median PSI > 0.5 (fleet-wide)
  or a single sensor PSI > 1. `simulate_traffic.py --drift 3` shifts sensors by 3 σ and triggers both.
- **Model-performance monitoring** needs ground truth, which for RUL arrives only after maintenance or failure.
  Offline, it is the test-set metrics. Online, clients POST the true RUL to `/feedback`, and the API exports a
  rolling MAE against `min(true RUL, 125)`. The traffic simulator uses the real NASA labels for this.
- The PSI buffer and feedback MAE are kept **per process**, so each Kubernetes replica reports its own values.
  A shared store (Redis) or batch job would be needed for fleet-wide numbers.

## Infrastructure
- Ports are separated so all deployments can run at once: Compose API **8000**, Terraform API **8001**,
  Kubernetes **30080**, Streamlit **8501**, Prometheus **9090**, Grafana **3000**, MLflow UI **5000**.
- Kubernetes uses the locally built image (`imagePullPolicy: IfNotPresent`), which works with Docker Desktop.
  For Minikube, run `minikube image load rul-api:1.0.0` first. The Service is `LoadBalancer` with
  NodePort 30080, because Docker Desktop's kind-based cluster does not forward plain NodePorts to Windows.
- Terraform manages the container, and the image comes from `docker build`. The provider's in-Terraform
  `build` block hung against Docker Engine 29 during testing. `keep_locally = true` means `terraform destroy`
  never deletes the image that Compose and Kubernetes share.
- On Windows, URLs use `127.0.0.1` rather than `localhost`. Python's `requests` added ~2 s per call through
  the IPv6 fallback.

## Known limitations
- FD002/FD004 (six operating conditions) would need per-condition normalisation before training. This is
  backlog item US15.
- No automatic retraining. Drift and feedback alerts only notify.
- MLflow uses a local SQLite store, so it isn't shared between machines or CI runs.
- Grafana uses the default `admin/admin` credentials. That is fine for a local lab and not for production.
