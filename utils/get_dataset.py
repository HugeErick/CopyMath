from huggingface_hub import snapshot_download

path = None
try:
  path = snapshot_download(
      repo_id="deepcopy/MathWriting-human",
      repo_type="dataset",
      local_dir=".",
  )
finally:
  if path:
    print(f"Downloaded dataset at {path}")
  else:
    print("Download failed or was interrupted.")
