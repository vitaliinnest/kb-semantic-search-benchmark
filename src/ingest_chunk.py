import argparse
import json
import re
import logging
import sys
import docx
import PyPDF2
from pathlib import Path
from tqdm import tqdm

# Fix Windows console encoding
if sys.platform == "win32":
	import codecs
	sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
	sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)

def read_text_file(path: Path) -> str:
	for encoding in ("utf-8", "cp1251", "latin-1"):
		try:
			return path.read_text(encoding=encoding)
		except UnicodeDecodeError:
			continue
	return path.read_text(errors="ignore")


def read_pdf_file(path: Path) -> str:
	text_parts = []
	try:
		with path.open("rb") as handle:
			reader = PyPDF2.PdfReader(handle)
			total_pages = len(reader.pages)
			
			# Показуємо прогрес для великих PDF (>50 сторінок)
			if total_pages > 50:
				pages_iter = tqdm(
					reader.pages,
					desc=f"Читання {path.name}",
					total=total_pages,
					leave=False
				)
			else:
				pages_iter = reader.pages
			
			for page_num, page in enumerate(pages_iter, 1):
				try:
					text_parts.append(page.extract_text() or "")
				except Exception as e:
					logging.warning(f"Помилка витягування сторінки {page_num} з {path.name}: {e}")
					continue
	except Exception as e:
			logging.error(f"Помилка читання PDF {path.name}: {e}")
			return ""
			
	return "\n".join(text_parts)


def read_docx_file(path: Path) -> str:
	document = docx.Document(str(path))
	parts = [para.text for para in document.paragraphs if para.text]
	return "\n".join(parts)


def read_document(path: Path) -> str:
	suffix = path.suffix.lower()
	if suffix in {".txt", ".md", ".rst"}:
		return read_text_file(path)
	if suffix == ".pdf":
		return read_pdf_file(path)
	if suffix == ".docx":
		return read_docx_file(path)
	return ""


def normalize_text(text: str) -> str:
	text = re.sub(r"\s+", " ", text)
	return text.strip()


def chunk_words(words: list[str], min_words: int, max_words: int, overlap: int) -> list[str]:
	if not words:
		return []

	chunks: list[str] = []
	start = 0
	total = len(words)

	while start < total:
		end = min(start + max_words, total)
		size = end - start

		if size < min_words and chunks:
			chunks[-1] = " ".join([chunks[-1]] + words[start:end])
			break

		chunk = " ".join(words[start:end])
		chunks.append(chunk)

		if end >= total:
			break

		start = max(0, end - overlap)

	return chunks


def ingest_chunks(
	input_dir: Path,
	output_file: Path,
	min_words: int,
	max_words: int,
	overlap: int,
) -> int:
	records = []
	processed_files = 0
	
	# Збираємо всі файли для обробки
	files = [p for p in sorted(input_dir.rglob("*")) if p.is_file()]
	logging.info(f"Знайдено {len(files)} файлів для обробки")
	
	for path in tqdm(files, desc="Обробка файлів"):
		logging.info(f"Обробка {path.name}")
		
		try:
			text = normalize_text(read_document(path))
			if not text:
				logging.warning(f"Текст не витягнуто з {path.name}")
				continue

			words = text.split(" ")
			chunks = chunk_words(words, min_words, max_words, overlap)
			source = str(path.relative_to(input_dir))
			
			logging.info(f"Створено {len(chunks)} чанків з {path.name} ({len(words)} слів)")

			for idx, chunk in enumerate(chunks):
				records.append(
					{
						"chunk_id": f"{source}:{idx}",
						"source": source,
						"text": chunk,
					}
				)
			processed_files += 1
		except Exception as e:
			logging.error(f"Помилка обробки {path.name}: {e}")
			continue

	logging.info(f"Успішно оброблено {processed_files}/{len(files)} файлів, створено {len(records)} чанків")
	
	output_file.parent.mkdir(parents=True, exist_ok=True)
	with output_file.open("w", encoding="utf-8") as handle:
		for record in records:
			handle.write(json.dumps(record, ensure_ascii=False) + "\n")

	return len(records)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Обробка та чанкінг документів.")
	parser.add_argument("--input", default="data/raw", help="Шлях до вихідних документів")
	parser.add_argument("--output", default="data/chunks.jsonl", help="Шлях для збереження результату")
	parser.add_argument("--min-words", type=int, default=300, help="Мінімальна кількість слів у чанку")
	parser.add_argument("--max-words", type=int, default=800, help="Максимальна кількість слів у чанку")
	parser.add_argument("--overlap", type=int, default=80, help="Перекриття між чанками (слова)")
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	input_dir = Path(args.input)
	output_file = Path(args.output)

	count = ingest_chunks(
		input_dir=input_dir,
		output_file=output_file,
		min_words=args.min_words,
		max_words=args.max_words,
		overlap=args.overlap,
	)
	print(f"Збережено {count} чанків у файл {output_file}")


if __name__ == "__main__":
	main()
