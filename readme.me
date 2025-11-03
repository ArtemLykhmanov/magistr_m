Traffic Classifier — README

Навчаємо модель, піднімаємо простий REST-API і показуємо, як він визначає атакувальний мережний трафік.

1) Що всередині
train_baseline.py — навчає модель і зберігає артефакти.
serve_api.py — REST-API (POST /predict) для онлайн-перевірки.
demo_batch.py — швидка офлайн-оцінка на частині CSV + звіт.
stream_sim_v3.py — “стрім”: імітація онлайн-потоку з автопорогом.
make_report_plots.py — картинки (матриця помилок, PR-крива).
data/UNSW_NB15_training-set.csv, data/UNSW_NB15_testing-set.csv — приклад даних.
artifacts/ — тут з’являються всі результати (модель, графіки, логи).

2)Швидкий старт (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

2.1) Навчити модель
python train_baseline.py

2.2) Підняти API
Відкрити документацію: http://127.0.0.1:8000/docs
Приклад запиту (через браузер у /docs):
{
  "values": {
    "dur": 2.5,
    "spkts": 12,
    "dpkts": 8,
    "sbytes": 1500,
    "dbytes": 900,
    "proto": "tcp",
    "service": "http",
    "state": "ESTABLISHED"
  }
}

Відповідь включає:
probability — ймовірність атаки (0..1),
prediction — 1 (атака) або 0 (норма),
threshold — поточний поріг.

3) Офлайн-демо (швидкі метрики і графіки)
На тестовому наборі 
$env:IDS_CSV="data/UNSW_NB15_testing-set.csv"
python demo_batch.py
python make_report_plots.py

4) Онлайн-демо (стрім із автопорогом)
Лишіть API працювати у першому терміналі:
python serve_api.py
У другому — стрім (на тестовому наборі):
$env:IDS_CSV="data/UNSW_NB15_testing-set.csv"
python stream_sim_v3.py

5) Як це працює (коротко)
Ми беремо ознаки потоку (dur, spkts, dpkts, sbytes, dbytes, proto, service, state тощо),
модель рахує ймовірність атаки, і якщо вона вище порога, повертаємо ALERT (1); інакше — OK (0).

