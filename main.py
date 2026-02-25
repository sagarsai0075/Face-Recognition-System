import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def ask_yes_no(prompt):
	while True:
		answer = input(prompt).strip().lower()
		if answer in {"y", "yes"}:
			return True
		if answer in {"n", "no"}:
			return False
		print("[WARN] Please enter 'y' or 'n'.")


def run_step(title, script_path, extra_args=None):
	if extra_args is None:
		extra_args = []

	command = [sys.executable, script_path, *extra_args]
	print(f"\n[STEP] {title}")
	print(f"[CMD] {' '.join(command)}")

	result = subprocess.run(command, cwd=ROOT_DIR)
	if result.returncode != 0:
		raise RuntimeError(f"{title} failed with exit code {result.returncode}")


def main():
	
	print("Face Recognition System ")
	print("=" * 60)

	do_capture = ask_yes_no(
		"Do you want to start camera and capture your image for Dataset Collection (y/n): "
	)

	if do_capture:
		person_name = input("Enter person name : ").strip()
		while not person_name:
			print("[WARN] Name cannot be empty.")
			person_name = input("Enter person name : ").strip()

		run_step("Collect Dataset", "scripts/collect_dataset.py", ["--name", person_name])
		run_step("Extract Embeddings", "scripts/extract_embeddings.py")
		run_step("Train SVM", "scripts/train_svm.py")
	else:
		print("[INFO] Skipping capture + retraining, using existing model files.")

	start_recognition = ask_yes_no("Do you want to start face recognition now? (y/n): ")
	if start_recognition:
		run_step("Recognize Faces", "scripts/recognize.py")
	else:
		print("[INFO] Recognition skipped.")


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\n[INFO] Process interrupted by user.")
	except Exception as exc:
		print(f"[ERROR] {exc}")
		sys.exit(1)
