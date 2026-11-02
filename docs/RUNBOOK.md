# Runbook: every command, per experiment (Windows PowerShell)

Run everything from the project root `aircraft-rul-mlops\`, with the venv activated (`.\.venv\Scripts\Activate.ps1`).
Docker Desktop must be running for sections 5-7. Enable *Settings → Kubernetes* for section 5B.

## 0. Setup, data and model (Experiment 4: ETL, Feature Store, MLflow)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.data --download            # downloads + extracts NASA C-MAPSS, prints dataset summary
python -m src.train                      # ETL -> feature_store/ -> 4 MLflow runs -> models/rul_model.joblib
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000    # http://127.0.0.1:5000
```
In the MLflow UI, open experiment **aircraft-rul-fd001**, tick all runs, then **Compare**. The best model is
under **Models → aircraft-rul-model**.

## 1. Run the application locally

```powershell
uvicorn api.main:app --port 8000          # http://127.0.0.1:8000/docs
streamlit run streamlit_app/app.py        # http://localhost:8501  (deep links: ?page=predict, ?page=monitoring, ...)
python scripts/simulate_traffic.py --n 100                          # real test engines + ground-truth feedback
python scripts/simulate_traffic.py --n 60 --error-rate 0.3 --drift 3  # errors + drifted inputs
```
Single request from PowerShell:
```powershell
$rows = Import-Csv engine_history_example.csv | Select-Object -Property * -ExcludeProperty unit
$body = @{engine_id="demo"; records=$rows} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body $body
```
(Download `engine_history_example.csv` from the dashboard's *RUL Prediction → Upload* tab.)

## 2. Tests (used by CI)

```powershell
pytest -v                                # 25 tests: data, features, metrics, model, API
pytest -v -m "not model"                 # only tests that need no trained model
```

## 3. Experiment 1: Git & GitHub

See [GIT_WORKFLOW.md](GIT_WORKFLOW.md). Key commands:
```powershell
git log --oneline --graph --all
git branch -a
git checkout -b feature/new-idea ; git commit -am "..." ; git checkout main ; git merge --no-ff feature/new-idea
git push origin main --tags
```

## 4. Experiment 3: CI/CD with GitHub Actions

The workflow `.github/workflows/ci.yml` runs on **every push** (any branch) and on **pull requests to main**:
1. unit tests (no data needed), 2. download data + train, 3. full pytest (model + API), 4. build the Docker
image and smoke-test `/health`, `/predict` and `/metrics` in a running container, 5. `terraform validate` +
Kubernetes manifest check.

Setup: push the repo to GitHub. Actions are enabled by default. Open the **Actions** tab.

Demonstrate a failed and then fixed run:
```powershell
git checkout -b demo/ci-failure
# introduce a bug, e.g. in src/data.py add_train_rul change  - df["cycle"]  to  + df["cycle"]
git commit -am "Introduce RUL label bug (CI should fail)"
git push -u origin demo/ci-failure       # Actions: test job FAILS on test_train_rul_counts_down_to_zero_and_caps
git revert --no-edit HEAD
git push                                 # Actions: green again
gh pr create --base main --head demo/ci-failure --title "CI demo" --body "Fail then fix"   # PR trigger
```

## 5A. Experiment 5: Docker

```powershell
docker build -f docker/Dockerfile -t rul-api:1.0.0 .
docker run -d --name rul-api-test -p 8002:8000 rul-api:1.0.0
docker ps
curl.exe http://127.0.0.1:8002/health
docker stop rul-api-test ; docker rm rul-api-test

# Full stack: API + dashboard + Prometheus + Grafana
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs -f api
docker compose -f docker/docker-compose.yml down
```

## 5B. Experiment 5: Kubernetes (Docker Desktop Kubernetes or Minikube)

```powershell
kubectl cluster-info
kubectl get nodes
docker build -f docker/Dockerfile -t rul-api:1.0.0 .     # (Minikube: minikube image load rul-api:1.0.0)
kubectl apply -f kubernetes/
kubectl get deployment,pods,svc -n rul-mlops              # 3 replicas
curl.exe http://127.0.0.1:30080/health
kubectl scale deployment rul-api -n rul-mlops --replicas=5
kubectl get pods -n rul-mlops                             # 5 replicas
kubectl get all -n rul-mlops
kubectl describe deployment rul-api -n rul-mlops
kubectl delete -f kubernetes/                             # clean up
```

## 6. Experiment 6: Terraform (Docker provider)

```powershell
docker build -f docker/Dockerfile -t rul-api:1.0.0 .     # image Terraform will run
cd terraform
terraform init
terraform validate
terraform fmt -check
terraform plan
terraform apply            # type yes  -> container rul-api-terraform on http://localhost:8001
curl.exe http://127.0.0.1:8001/health
terraform output
terraform state list
terraform apply -var="external_port=8003"   # change demo (like changing 8082 -> 8083)
terraform destroy          # type yes
cd ..
```
Terraform has its own container name and port, so it can run next to Compose (8000) and Kubernetes (30080).

## 7. Experiment 7: Prometheus & Grafana

```powershell
docker compose -f docker/docker-compose.yml up -d --build
python scripts/simulate_traffic.py --n 100
python scripts/simulate_traffic.py --n 60 --error-rate 0.3 --drift 3
```
- Raw metrics: <http://127.0.0.1:8000/metrics>
- Prometheus: <http://localhost:9090>. Check **Status → Targets** (compose / terraform / kubernetes jobs) and **Alerts**.
  Queries to try:
  - `up`
  - `sum(rul_predictions_total)`
  - `sum by (type) (rul_prediction_errors_total)`
  - `sum by (status) (rate(rul_http_requests_total{endpoint="/predict"}[1m]))`
  - `histogram_quantile(0.95, sum by (le) (rate(rul_prediction_latency_seconds_bucket[5m])))`
  - `rul_input_psi`
  - `rul_feedback_rolling_mae`
- Grafana: <http://localhost:3000> (opens read-only without login; admin / admin to edit). Open **Dashboards → Aircraft RUL → Aircraft Engine RUL - Prediction Service**.
  The data source and dashboard are provisioned automatically.
- Availability exercise: `docker stop rul-api-compose` → `up{job="rul-api-compose"}` = 0 and the
  `RulApiDown` alert fires. Then `docker start rul-api-compose`.

---

## Screenshot checklist for the practical submission

**Exp 1: Git & GitHub**
1. `git log --oneline --graph --all` showing feature branches merged into main
2. `git branch -a` and `git tag`
3. GitHub repository home page with the README rendered
4. GitHub *Insights → Network* graph (branches/merges)

**Exp 2: Agile & Scrum**
5. `docs/SCRUM.md` rendered on GitHub: roles, Product Backlog, Sprint Backlogs
6. Daily Scrum, Sprint Review and Retrospective tables

**Exp 3: CI/CD**
7. `.github/workflows/ci.yml` in the editor
8. Actions tab: a **failed** run (red) with the failing test highlighted in the log
9. Actions tab: the **successful** run (green) after the fix, with all 3 jobs
10. Docker smoke-test job log showing the `/predict` JSON response
11. A pull request page with the CI check result

**Exp 4: ETL, Feature Store, MLflow**
12. Terminal output of `python -m src.train` (metrics table + "BEST:")
13. `feature_store/` folder (parquet + json) and one metadata JSON
14. MLflow UI: experiment run list
15. MLflow UI: compare view of the 4 runs (metrics chart)
16. MLflow UI: one run's artifacts (model + evaluation CSV) and the Model Registry entry
17. Dashboard pages: *ETL & Feature Store*, *Model Experiments*, *Model Evaluation*

**Exp 5: Docker & Kubernetes**
18. `docker build` finishing + `docker images rul-api`
19. `docker compose ps` with 4 healthy containers
20. Browser: `http://127.0.0.1:8000/docs` (Swagger) with a successful `/predict` call
21. `kubectl get pods -n rul-mlops` with **3** pods
22. `kubectl scale ... --replicas=5` + `kubectl get pods` with **5** pods
23. `kubectl get svc -n rul-mlops` + `curl` to port 30080

**Exp 6: Terraform**
24. `terraform init`, 25. `terraform validate`, 26. `terraform plan` (2 to add)
27. `terraform apply` with outputs, 28. `docker ps` showing `rul-api-terraform`, browser on `:8001/health`
29. `terraform destroy` (2 destroyed)

**Exp 7: Prometheus & Grafana**
30. `/metrics` raw output
31. Prometheus *Targets* page (all rul-api jobs UP)
32. Prometheus graph for the p95 latency query and for `rul_prediction_errors_total`
33. Prometheus *Alerts* page after running the drift simulation
34. Grafana dashboard: operational row, drift row, model-performance row
35. `up` = 0 after `docker stop rul-api-compose`

**Streamlit dashboard (final deliverable)**
36. All 9 pages, especially *Project Overview*, *RUL Prediction* (example engine + CSV upload result), *Monitoring* with Prometheus data, and *MLOps Pipeline*
