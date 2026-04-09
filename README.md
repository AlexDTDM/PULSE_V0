# PULSE — Patient-Unified Likelihood Scheduling Engine

PULSE is a self-learning MCP Server that predicts the optimal time to interact with a participant. The model improves as new interactions are recorded.

## How it works

A participant is experiencing stress (entering a vulnerability state) and has a presentation in 5h (criticality window: 5 hours). PULSE receives two inputs: **participant_id** and **criticality** ("in the next 5 hours").

**Step 1 — Criticality assessment.** If the time window is ≤ 1 hour, the system returns immediately: "interact now." Urgency overrides optimization.

**Step 2 — Data loading.** The system fetches the full interaction history and the demographic table from the database.

**Step 3 — Demographic lookup.** The participant's profile (age, sex, and any other available features) is retrieved, along with any time constraints — time slots when the participant is unavailable.

**Step 4 — Cohort construction.** A tailored subgroup is built from the population by filtering on shared categorical features (e.g. same sex), then selecting the k nearest neighbors (default k=100) by continuous features (e.g. closest in age). If applying a categorical filter reduces the pool below n_min (default 100), that filter is dropped to ensure a sufficient sample size.

**Step 5 — Three probability distributions are computed:**
- **Population** — based on all other participants
- **Patient** — based on the participant's own history only
- **Tailored cohort** — based on the demographically matched subgroup

**Step 6 — Time window filtering.** The criticality input is parsed into a concrete day/time range. Time slots where the participant is unavailable (per their time constraints) are blocked. Only remaining slots within the criticality window are considered.

**Step 7 — Peak selection.** For each distribution, the top time slots with the highest interaction probability are identified within the available window.

**Step 8 — Automatic model selection.** The system counts how many interactions the patient (then the cohort) has in the criticality window's time bins, and picks the most personalized distribution that has at least 10.

**Step 9 — Output.** The system returns a JSON response containing the selected distribution's optimal timing, the cohort composition, and a rationale explaining the selection.

## MCP Tools

| Tool | Description |
|------|-------------|
| `get_best_times` | Returns optimal time slots for a participant given a criticality window |
| `participant_metadata` | Adds or updates participant demographic data (intervention, age, sex, ethnicity, education_level, employment, time_constraints) |
| `add_interaction` | Records new interaction timestamps for a participant |
| `add_intervention_context` | Stores free-text context for an intervention |
| `plot_all_distributions` | Full week + zoomed criticality window plots with all three distributions |
| `plot_p_distribution_demographic` | Population distributions grouped by demographic (sex, age range, intervention) |

## Architecture

- **App**: Single Python file (`app.py`) served via Gradio with MCP support
- **Database**: External PostgreSQL (tables: `interactions`, `df_demographic`, `best_times`, `intervention_context`)
- **Deployment**: Docker container on Coolify (Hetzner server)
- **Timezone**: Europe/Zurich

The database is not part of the Docker image — the app connects to it via `DATABASE_URL`.

## Requirements

- Python 3.11+
- PostgreSQL database

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (`postgresql://user:pass@host:port/db`) |

## Local Development

```bash
pip install -r requirements.txt
cp .env.example .env  # then edit DATABASE_URL
python app.py
```

The server starts at `http://localhost:7860`. MCP endpoint at `http://localhost:7860/gradio_api/mcp/`.

## Docker

```bash
docker compose up --build
```

## Deployment on Coolify

1. Create a PostgreSQL database in Coolify
2. Deploy this repo as a Docker Compose service
3. Set `DATABASE_URL` to the internal Postgres URL (change `postgres://` to `postgresql://`)
4. Uncomment the `coolify` network in `docker-compose.yml`
5. Enable **Force HTTPS** in the proxy settings

## Testing

A test notebook is provided at `tests/test_distribution_selection.ipynb`. It calls the running MCP server via the Gradio API and tests 8 participant profiles across 4 criticality windows, with heatmap and distribution plots.

```bash
# Start the server first
docker compose up --build
# Then run the notebook in tests/
```

## License

Proprietary — PRECIOUS project.
