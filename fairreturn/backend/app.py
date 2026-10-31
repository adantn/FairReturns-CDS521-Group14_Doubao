"""Flask REST API + static web front end.

    python -m fairreturn.backend                 # http://127.0.0.1:8000
    python -m fairreturn.backend --port 8080 --host 0.0.0.0 --llm mock

All endpoints return JSON; see GET /api/docs (or docs/API.md) for the list.
"""
from __future__ import annotations

import argparse
import io

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from fairreturn import config as cfg
from fairreturn.backend.service import ApiError, FairReturnService

ENDPOINTS = [
    ("GET", "/api/health", "status, loaded models, LLM backend"),
    ("GET", "/api/meta", "country groups, models, data report, split sizes"),
    ("GET", "/api/products?q=&limit=", "search the product catalogue"),
    ("GET", "/api/examples", "preset demo orders taken from real test lines"),
    ("POST", "/api/predict", "risk of one order line from every model, with factor contributions"),
    ("POST", "/api/counterfactual", "the same order for every country group (proxy_mode: with_proxies|country_only)"),
    ("POST", "/api/assistant", "decision assistant: debiased risk -> next step (LLM with guardrail, or rules)"),
    ("GET", "/api/invoices?group=&has_return=&q=&sort=country_effect&limit=&offset=", "real test invoices in the demo sample"),
    ("GET", "/api/invoices/<invoice>?as_group=", "line-level risks of a real invoice, optionally moved to another group"),
    ("GET", "/api/results", "experiment summary, figures, leakage probes, sweep and stress test"),
    ("GET", "/api/fairness?method=", "per-country-group TPR / FPR / flag rate / calibration"),
    ("GET", "/api/audit", "LLM audit summary and pair list"),
    ("GET", "/api/audit/<pair_id>", "one audited pair: prompts and answers under the four conditions"),
    ("GET", "/api/tags", "evaluation of the LLM-generated product tags"),
    ("POST", "/api/score-csv", "batch-score an uploaded CSV in the Online Retail format (?format=csv to download)"),
]


def create_app(service: FairReturnService | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    svc = service or FairReturnService()
    app.extensions["fairreturn_service"] = svc

    def body() -> dict:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ApiError("expected a JSON object in the request body")
        return data

    @app.errorhandler(ApiError)
    def _api_error(e: ApiError):
        return jsonify({"error": e.message}), e.status

    @app.errorhandler(404)
    def _not_found(_e):
        if request.path.startswith("/api/"):
            return jsonify({"error": f"no endpoint {request.path}"}), 404
        return send_from_directory(cfg.FRONTEND_DIR, "index.html")

    @app.errorhandler(Exception)
    def _server_error(e):  # keep the demo alive and the message readable
        app.logger.exception(e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    # ------------------------------------------------------------------ front end
    @app.get("/")
    def index():
        return send_from_directory(cfg.FRONTEND_DIR, "index.html")

    @app.get("/static/<path:path>")
    def static_files(path):
        return send_from_directory(cfg.FRONTEND_DIR, path)

    @app.get("/figures/<path:name>")
    def figures(name):
        return send_from_directory(cfg.FIG_DIR, name)

    # ------------------------------------------------------------------ API
    @app.get("/api/docs")
    def docs():
        return jsonify([{"method": m, "path": p, "description": d} for m, p, d in ENDPOINTS])

    @app.get("/api/health")
    def health():
        return jsonify(svc.health())

    @app.get("/api/meta")
    def meta():
        return jsonify(svc.meta_info())

    @app.get("/api/products")
    def products():
        return jsonify(svc.search_products(request.args.get("q", ""), int(request.args.get("limit", 30))))

    @app.get("/api/examples")
    def examples():
        return jsonify(svc.examples())

    @app.post("/api/predict")
    def predict():
        return jsonify(svc.predict(body()))

    @app.post("/api/counterfactual")
    def counterfactual():
        b = body()
        return jsonify(svc.counterfactual(b, b.get("proxy_mode", "with_proxies")))

    @app.post("/api/assistant")
    def assistant():
        return jsonify(svc.assistant(body()))

    @app.get("/api/invoices")
    def invoices():
        a = request.args
        return jsonify(svc.list_invoices(a.get("group") or None, a.get("has_return") or None, a.get("q") or None,
                                         int(a.get("limit", 50)), int(a.get("offset", 0)), a.get("sort") or None))

    @app.get("/api/invoices/<invoice>")
    def invoice(invoice):
        return jsonify(svc.invoice_detail(invoice, request.args.get("as_group") or None))

    @app.get("/api/results")
    def results():
        return jsonify(svc.results())

    @app.get("/api/fairness")
    def fairness():
        return jsonify(svc.fairness(request.args.get("method") or None))

    @app.get("/api/audit")
    def audit():
        return jsonify(svc.audit())

    @app.get("/api/audit/<pair_id>")
    def audit_pair(pair_id):
        return jsonify(svc.audit_pair(pair_id))

    @app.get("/api/tags")
    def tags():
        return jsonify(svc.tags())

    @app.post("/api/score-csv")
    def score_csv():
        if "file" in request.files:
            text = request.files["file"].read().decode("utf-8-sig", errors="replace")
        else:
            text = request.get_data(as_text=True)
        if not text.strip():
            raise ApiError("upload a CSV file (form field 'file') or send CSV text as the body")
        summary, csv_text = svc.score_csv(text)
        if request.args.get("format") == "csv":
            return send_file(io.BytesIO(csv_text.encode("utf-8")), mimetype="text/csv", as_attachment=True,
                             download_name="scored_orders.csv")
        return jsonify({**summary, "csv": csv_text})

    @app.after_request
    def no_cache(resp: Response):
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    return app


def main():
    ap = argparse.ArgumentParser(description="Fair return-risk demo server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--llm", choices=["auto", "api", "mock"], default="auto",
                    help="decision assistant backend (auto = API if LLM_* env vars are set)")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    svc = FairReturnService(llm_backend=a.llm)
    app = create_app(svc)
    print(f"\n  Fair return-risk demo:  http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}\n"
          f"  models: {', '.join(svc.models)}   assistant LLM: {svc._llm_client().describe()['backend']}\n")
    app.run(host=a.host, port=a.port, debug=a.debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
