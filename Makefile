.PHONY: up down seed test logs

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