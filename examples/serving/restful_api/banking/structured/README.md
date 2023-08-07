# Banking System

```bash
(examples/serving/restful_api/banking/structured) $ export PYTHONPATH=.
(examples/serving/restful_api/banking/structured) $ python api/data/seed_database.py
```

Run FastAPI:

```bash
(examples/serving/restful_api/banking/structured) $ uvicorn app:app --reload
```