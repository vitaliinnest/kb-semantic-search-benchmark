"""
Скрипт для тестування всіх моделей семантичного пошуку
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
	import codecs
	sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
	sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


MODELS = [
	{
		"name": "SBERT (paraphrase-multilingual-MiniLM-L12-v2)",
		"type": "sbert",
		"args": ["--model-type", "sbert", "--model", "paraphrase-multilingual-MiniLM-L12-v2"],
	},
	{
		"name": "TF-IDF",
		"type": "tfidf",
		"args": ["--model-type", "tfidf", "--tfidf-max-features", "50000"],
	},
	{
		"name": "Word2Vec",
		"type": "word2vec",
		"args": [
			"--model-type",
			"word2vec",
			"--gensim-vector-size",
			"300",
			"--gensim-window",
			"5",
			"--gensim-epochs",
			"10",
		],
	},
	{
		"name": "FastText",
		"type": "fasttext",
		"args": [
			"--model-type",
			"fasttext",
			"--gensim-vector-size",
			"300",
			"--gensim-window",
			"5",
			"--gensim-epochs",
			"10",
		],
	},
	{
		"name": "BERT (bert-base-multilingual-cased)",
		"type": "bert",
		"args": [
			"--model-type",
			"bert",
			"--bert-model",
			"bert-base-multilingual-cased",
			"--bert-batch-size",
			"16",
		],
	},
]

TEST_QUERIES = [
	"машинне навчання",
	"нейронні мережі",
	"глибоке навчання",
	"штучний інтелект",
	"обробка природної мови",
]


class Logger:
	"""Логер для виводу у консоль та файл одночасно"""

	def __init__(self, log_file: Path):
		self.log_file = log_file
		self.log_file.parent.mkdir(parents=True, exist_ok=True)
		# Очищаємо файл при створенні
		with open(self.log_file, "w", encoding="utf-8") as f:
			f.write(f"Тестування моделей семантичного пошуку\n")
			f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			f.write("=" * 80 + "\n\n")

	def log(self, message: str, console: bool = True):
		"""Записує повідомлення у файл і опціонально у консоль"""
		if console:
			print(message)
		with open(self.log_file, "a", encoding="utf-8") as f:
			f.write(message + "\n")

	def log_dict(self, data: dict, console: bool = True):
		"""Записує словник у JSON форматі"""
		json_str = json.dumps(data, ensure_ascii=False, indent=2)
		self.log(json_str, console=console)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Тестування всіх моделей семантичного пошуку")
	parser.add_argument("--chunks", default="data/chunks.jsonl", help="Шлях до чанків")
	parser.add_argument("--skip-build", action="store_true", help="Пропустити побудову індексів")
	parser.add_argument(
		"--artifacts-root",
		default="artifacts",
		help="Коренева папка артефактів (кожна модель буде збережена в <root>/<model-type>)",
	)
	parser.add_argument(
		"--models",
		nargs="*",
		choices=["sbert", "tfidf", "word2vec", "fasttext", "bert"],
		help="Список моделей для тестування (за замовчуванням: всі)",
	)
	parser.add_argument(
		"--output",
		default="results",
		help="Папка результатів (буде створено test_results_YYYY-MM-DD_HH-MM-SS.txt/.json) або конкретний .txt файл",
	)
	return parser


def run_command(cmd: list[str], description: str, logger: Logger) -> tuple[bool, float, str]:
	"""Запускає команду і повертає (успіх, час виконання, вивід)"""
	logger.log(f"\n{'='*80}")
	logger.log(f"🚀 {description}")
	logger.log(f"{'='*80}")
	logger.log(f"Команда: {' '.join(cmd)}\n")

	start_time = time.time()
	
	# Встановлюємо UTF-8 кодування для subprocess на Windows
	env = None
	if sys.platform == "win32":
		import os
		env = os.environ.copy()
		env["PYTHONIOENCODING"] = "utf-8"
	
	try:
		result = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env, encoding="utf-8")
		elapsed = time.time() - start_time
		logger.log(f"\n✅ Завершено за {elapsed:.2f} сек")
		
		# Записуємо вивід у файл, але не у консоль (щоб не засмічувати)
		if result.stdout:
			logger.log("\nВивід команди:", console=False)
			logger.log(result.stdout, console=False)
		
		return True, elapsed, result.stdout
	except subprocess.CalledProcessError as e:
		elapsed = time.time() - start_time
		error_msg = f"\n❌ Помилка після {elapsed:.2f} сек: {e}"
		logger.log(error_msg)
		
		if e.stderr:
			logger.log("\nПомилка:", console=False)
			logger.log(e.stderr, console=False)
		
		return False, elapsed, e.stderr or ""


def main():
	args = build_arg_parser().parse_args()
	run_started_at = datetime.now()

	output_arg = Path(args.output)
	if output_arg.suffix.lower() == ".txt":
		output_file = output_arg
		json_file = output_file.with_suffix(".json")
	else:
		timestamp = run_started_at.strftime("%Y-%m-%d_%H-%M-%S")
		base_results_dir = output_arg / "tests"
		output_file = base_results_dir / f"test_results_{timestamp}.txt"
		json_file = base_results_dir / f"test_results_{timestamp}.json"
		counter = 2
		while output_file.exists() or json_file.exists():
			output_file = base_results_dir / f"test_results_{timestamp}-{counter}.txt"
			json_file = base_results_dir / f"test_results_{timestamp}-{counter}.json"
			counter += 1

	run_dir = output_file.parent

	# Створюємо логер
	logger = Logger(output_file)

	chunks_path = Path(args.chunks)
	artifacts_root = Path(args.artifacts_root)
	if not chunks_path.exists():
		logger.log(f"❌ Файл чанків не знайдено: {chunks_path}")
		logger.log("Спочатку запустіть:")
		logger.log(f"  python src/ingest_chunk.py --input data/raw --output {chunks_path}")
		sys.exit(1)

	# Фільтруємо моделі за вибором користувача
	models_to_test = MODELS
	if args.models:
		models_to_test = [m for m in MODELS if m["type"] in args.models]

	logger.log(f"\n📊 Тестування {len(models_to_test)} модел(і/ей)\n")
	logger.log(f"📝 Результати зберігаються у: {output_file.absolute()}\n")
	logger.log(f"📂 Папка результатів: {run_dir.absolute()}\n")
	logger.log(f"📦 Артефакти моделей: {artifacts_root.absolute()} / <model-type>\n")

	results = []

	for model_config in models_to_test:
		model_name = model_config["name"]
		model_type = model_config["type"]
		artifacts_dir = artifacts_root / model_type

		logger.log(f"\n{'#'*80}")
		logger.log(f"# Модель: {model_name}")
		logger.log(f"{'#'*80}")

		# Побудова індексу
		if not args.skip_build:
			build_cmd = [
				sys.executable,
				"src/build_index.py",
				"--chunks",
				str(chunks_path),
				"--artifacts",
				str(artifacts_dir),
			] + model_config["args"]

			success, build_time, output = run_command(
				build_cmd, f"Побудова індексу для {model_name}", logger
			)
			if not success:
				results.append(
					{
						"model": model_name,
						"type": model_type,
						"build_time": build_time,
						"build_status": "FAILED",
						"query_status": "SKIPPED",
					}
				)
				continue
		else:
			build_time = 0.0
			logger.log(f"⏭️  Пропущено побудову індексу (--skip-build)")

		# Тестові запити
		query_times = []
		query_success = True
		for query_text in TEST_QUERIES:
			query_cmd = [
				sys.executable,
				"src/query.py",
				query_text,
				"--artifacts",
				str(artifacts_dir),
				"--top-k",
				"3",
			]

			success, query_time, output = run_command(
				query_cmd, f"Запит: '{query_text}'", logger
			)
			if not success:
				query_success = False
				break
			query_times.append(query_time)

		avg_query_time = sum(query_times) / len(query_times) if query_times else 0.0

		results.append(
			{
				"model": model_name,
				"type": model_type,
				"build_time": build_time,
				"build_status": "OK" if not args.skip_build else "SKIPPED",
				"query_status": "OK" if query_success else "FAILED",
				"avg_query_time": avg_query_time,
				"queries_tested": len(query_times),
			}
		)

	# Підсумок
	logger.log(f"\n{'='*80}")
	logger.log("📈 ПІДСУМОК ТЕСТУВАННЯ")
	logger.log(f"{'='*80}\n")

	logger.log(f"{'Модель':<20} {'Індекс':<10} {'Час (сек)':<12} {'Пошук':<10} {'Сер. час (сек)':<15}")
	logger.log("-" * 80)

	for r in results:
		build_str = f"{r['build_time']:.2f}" if r["build_status"] == "OK" else "-"
		query_str = f"{r['avg_query_time']:.3f}" if r["query_status"] == "OK" else "-"
		logger.log(
			f"{r['model']:<20} {r['build_status']:<10} {build_str:<12} {r['query_status']:<10} {query_str:<15}"
		)

	logger.log("\n✅ Тестування завершено!")
	logger.log(f"\n📁 Повні результати збережено у: {output_file.absolute()}")

	# Зберігаємо результати у JSON
	with open(json_file, "w", encoding="utf-8") as f:
		json.dump(
			{
				"timestamp": run_started_at.isoformat(),
				"results": results,
			},
			f,
			ensure_ascii=False,
			indent=2,
		)
	logger.log(f"📊 JSON результати: {json_file.absolute()}")
	
	print(f"\n{'='*80}")
	print(f"✅ Готово! Результати збережено:")
	print(f"   📝 {output_file.absolute()}")
	print(f"   📊 {json_file.absolute()}")
	print(f"{'='*80}")


if __name__ == "__main__":
	main()
