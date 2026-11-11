.PHONY: up down seed test logs benchmark plots paper-artifacts

up:
	docker compose up --build

down:
	docker compose down

seed:
	docker compose exec app python -m app.seed

test:
	docker compose exec app python -m unittest discover -s tests -v

logs:
	docker compose logs -f app

benchmark:
	docker compose exec app python -m app.scripts.run_csai_benchmark

plots:
	docker compose exec app python -m app.scripts.plot_csai_figures

paper-artifacts: benchmark plots
