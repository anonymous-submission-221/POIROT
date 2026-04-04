import os
import torch
import sys
import yaml
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict

DATASET_ROOT = "./videos"
DATASET_LABEL_ROOT = "./"
OUTPUT_LABEL_DIR = os.path.join(DATASET_LABEL_ROOT, "labels")
PROCESSED_LOG_PATH = os.path.join(DATASET_LABEL_ROOT, "processed_videos.log")

CONFIG_PATH = "GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "groundingdino_swint_ogc.pth"

GLOBAL_BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.22

CLASS_MAP = {
    "person": 0, "man": 0, "woman": 0, "child": 0, "baby": 0, "face": 1, "head": 1, "hand": 2, "arm": 2, "leg": 3, "foot": 3,
    "clothing": 4, "shirt": 4, "pants": 4, "jacket": 4, "dress": 4, "hat": 4, "shoe": 4, "glasses": 5, "sunglasses": 5,
    "bag": 6, "backpack": 6, "suitcase": 6, "handbag": 6, "wallet": 6, "purse": 6,
    "animal": 7, "dog": 7, "cat": 7, "horse": 7, "cow": 7, "bird": 8, "chicken": 8, "insect": 9, "bug": 9, "spider": 9, "fish": 10,
    "vehicle": 11, "car": 11, "truck": 11, "bus": 11, "van": 11, "bicycle": 12, "bike": 12, "motorcycle": 12,
    "train": 13, "subway": 13, "airplane": 14, "plane": 14, "helicopter": 14, "boat": 15, "ship": 15, "yacht": 15,
    "traffic sign": 16, "traffic light": 16, "stop sign": 16,
    "furniture": 17, "chair": 18, "sofa": 18, "couch": 18, "bench": 18, "seat": 18, "bed": 19, "table": 20, "desk": 20, "cabinet": 21, "shelf": 21, "wardrobe": 21, "drawer": 21,
    "appliance": 22, "refrigerator": 22, "oven": 22, "microwave": 22, "washing machine": 22, "fan": 22,
    "screen": 23, "tv": 23, "television": 23, "monitor": 23, "display": 23, "laptop": 24, "computer": 24, "keyboard": 24, "mouse": 24, "phone": 25, "smartphone": 25, "mobile": 25, "camera": 26,
    "food": 27, "fruit": 28, "apple": 28, "banana": 28, "orange": 28, "vegetable": 29, "carrot": 29, "tomato": 29, "meat": 30, "chicken wing": 30, "sausage": 30, "bread": 31, "cake": 31, "pizza": 31, "sandwich": 31, "hamburger": 31,
    "drink": 32, "coffee": 32, "tea": 32, "bottle": 33, "cup": 33, "glass": 33, "mug": 33, "bowl": 34, "plate": 34, "dish": 34, "utensil": 35, "knife": 35, "fork": 35, "spoon": 35, "chopstick": 35,
    "tool": 36, "hammer": 36, "wrench": 36, "screwdriver": 36, "scissors": 36, "drill": 36, "pliers": 36,
    "weapon": 37, "gun": 37, "rifle": 37, "pistol": 37, "sword": 37,
    "sport equipment": 38, "racket": 38, "skateboard": 38, "snowboard": 38, "skis": 38, "ball": 39, "football": 39, "basketball": 39, "soccer": 39, "baseball": 39, "tennis ball": 39,
    "toy": 40, "doll": 40, "teddy bear": 40,
    "paper": 41, "book": 41, "document": 41, "newspaper": 41, "notebook": 41, "magazine": 41,
    "box": 42, "package": 42, "carton": 42, "cardboard": 42, "container": 42,
    "text": 43, "words": 43, "signage": 43, "logo": 43, "brand": 43, "title": 43, "poster": 44, "painting": 44, "picture": 44, "photo": 44,
    "plant": 45, "flower": 45, "grass": 45, "bush": 45, "tree": 46, "leaf": 46,
    "building": 47, "house": 47, "skyscraper": 47, "door": 48, "window": 48, "gate": 48, "stairs": 49, "staircase": 49, "ladder": 49,
    "road": 50, "street": 50, "highway": 50, "bridge": 50, "sidewalk": 50,
    "water": 51, "lake": 51, "river": 51, "ocean": 51, "sea": 51, "pool": 51,
    "mountain": 52, "hill": 52, "rock": 52, "ground": 52, "sand": 52, "dirt": 52,
    "sky": 53, "cloud": 53, "sun": 53, "moon": 53, "star": 53,
    "light": 54, "lamp": 54, "lantern": 54, "bulb": 54,
    "trash can": 55, "garbage can": 55, "waste bin": 55,
    "instrument": 56, "guitar": 56, "piano": 56, "drum": 56, "violin": 56,
    "cosmetics": 57, "makeup": 57, "lipstick": 57, "perfume": 57,
    "medical": 58, "medicine": 58, "pill": 58, "syringe": 58, "mask": 58, "wheelchair": 58
}

CUSTOM_NAMES = {
    0: "person", 1: "face_head", 2: "hand_arm", 3: "leg_foot", 4: "clothing", 5: "eyewear",
    6: "bag_wallet", 7: "animal", 8: "bird", 9: "insect", 10: "aquatic_animal",
    11: "vehicle", 12: "two_wheeler", 13: "train", 14: "aircraft", 15: "watercraft",
    16: "traffic_element", 17: "furniture", 18: "seat", 19: "bed", 20: "table", 21: "storage_furniture",
    22: "appliance", 23: "screen", 24: "computer_peripherals", 25: "phone", 26: "camera",
    27: "food", 28: "fruit", 29: "vegetable", 30: "meat", 31: "baked_fast_food",
    32: "drink", 33: "drinkware", 34: "tableware", 35: "cutlery",
    36: "tool", 37: "weapon", 38: "sport_equipment", 39: "ball", 40: "toy",
    41: "paper_book", 42: "box_container", 43: "text_sign", 44: "wall_art",
    45: "plant", 46: "tree", 47: "building", 48: "door_window", 49: "stairs",
    50: "road_path", 51: "water_body", 52: "terrain", 53: "sky_element",
    54: "lighting", 55: "trash_bin", 56: "instrument", 57: "cosmetics", 58: "medical"
}

TEXT_PROMPT = " . ".join(CLASS_MAP.keys()) + " ."

def match_phrase_to_id(phrase):
    phrase = phrase.lower().strip()
    sorted_keys = sorted(CLASS_MAP.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in phrase:
            return CLASS_MAP[key]
    return -1

def get_processed_videos():
    if not os.path.exists(PROCESSED_LOG_PATH): return set()
    with open(PROCESSED_LOG_PATH, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f.readlines() if line.strip()])

def mark_video_as_processed(vid_id):
    with open(PROCESSED_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"{vid_id}\n")

def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["PYTHONUTF8"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(DATASET_ROOT):
        return

    all_dirs = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))])
    processed_ids = get_processed_videos()
    todo_dirs = [d for d in all_dirs if d not in processed_ids]

    if not todo_dirs:
        return

    model = load_model(CONFIG_PATH, WEIGHTS_PATH).to(device)

    for vid_id in tqdm(todo_dirs):
        vid_path = os.path.join(DATASET_ROOT, vid_id)
        images = [f for f in os.listdir(vid_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not images:
            mark_video_as_processed(vid_id)
            continue

        for img_name in images:
            img_path = os.path.join(vid_path, img_name)
            try:
                rel_path = os.path.relpath(img_path, DATASET_ROOT)
                label_path = os.path.join(OUTPUT_LABEL_DIR, os.path.splitext(rel_path)[0] + ".txt")
                os.makedirs(os.path.dirname(label_path), exist_ok=True)

                image_source, image = load_image(img_path)
                with torch.no_grad():
                    boxes, logits, phrases = predict(
                        model=model, image=image.to(device), caption=TEXT_PROMPT,
                        box_threshold=GLOBAL_BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD
                    )
                    boxes = boxes.cpu()

                with open(label_path, "w", encoding="utf-8") as f:
                    for box, phrase in zip(boxes, phrases):
                        class_id = match_phrase_to_id(phrase)
                        if class_id != -1:
                            cx, cy, bw, bh = box.tolist()
                            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception:
                pass

        mark_video_as_processed(vid_id)

    update_dataset_config()

def update_dataset_config():
    content = "".join([f"{CUSTOM_NAMES.get(i, f'unknown_{i}')}\n" for i in range(max(CUSTOM_NAMES.keys()) + 1)])
    for path in [OUTPUT_LABEL_DIR, DATASET_LABEL_ROOT]:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "classes.txt"), "w") as f: f.write(content)

    yaml_content = {'path': DATASET_LABEL_ROOT, 'train': 'train.txt', 'val': 'val.txt', 'names': CUSTOM_NAMES}
    with open(os.path.join(DATASET_LABEL_ROOT, "data.yaml"), "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

if __name__ == "__main__":
    main()