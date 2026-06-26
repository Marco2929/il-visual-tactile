import os
import json
from io import BytesIO
from PIL import Image
import torch
from torchvision.transforms import ToPILImage
from ollama import Client
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Konfiguration
REPO_ID = "foil_lifting_baseline_1024pc"
# WICHTIG: Der Pfad muss bis in den Datensatz-Ordner führen!
ROOT_DIR = "/home/mm/dev/il-visual-tactile/data/lerobot_datasets/foil_lifting_baseline_1024pc"
CAMERA_KEY = "observation.images.front"
OLLAMA_MODEL = "gemma4:latest"
OUTPUT_LOG = "vlm_labeling_results.json"

client = Client(host='http://localhost:11434')
to_pil = ToPILImage()


def evaluate_image(image_bytes: bytes) -> str:
    prompt = (
        "You are an objective robotic behavior evaluator.\n"
        "Task: Determine if the robot hand is successfully pinching and lifting the foil.\n\n"
        "Rules:\n"
        "- Reply 'success' if the hand is pinching/grasping/holding the foil\n"
        "- Reply 'failed' if the foil is laying flat on the ground and not being held\n\n"
        "Reply with ONLY one word: success or failed"
    )

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_bytes]
        }],
        options={"temperature": 0.0}
    )

    print(response)

    result = response['message']['content'].strip().upper()
    return "SUCCESS" if "SUCCESS" in result else "FAILED"


def main():
    print(f"Lade lokalen Datensatz '{REPO_ID}'...")

    # Durch den exakten Pfad wird die lokale info.json sofort gefunden
    dataset = LeRobotDataset(REPO_ID, root=ROOT_DIR)

    results = {}
    episodes_meta = dataset.meta.episodes
    num_episodes = len(episodes_meta["episode_index"])

    print(f"Starte Evaluierung von {num_episodes} Episoden.")

    for ep_idx in range(num_episodes):
        to_idx = episodes_meta["dataset_to_index"][ep_idx]
        last_frame_idx = to_idx - 1

        frame_data = dataset[last_frame_idx]
        image_tensor = frame_data[CAMERA_KEY]

        pil_img = to_pil(image_tensor)
        byte_arr = BytesIO()
        pil_img.save(byte_arr, format='JPEG')
        image_bytes = byte_arr.getvalue()

        status = evaluate_image(image_bytes)
        results[f"episode_{ep_idx}"] = status
        print(f"Episode {ep_idx} (Frame {last_frame_idx}) -> {status}")

    with open(OUTPUT_LOG, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluierung abgeschlossen. Log unter: {OUTPUT_LOG}")


if __name__ == "__main__":
    main()