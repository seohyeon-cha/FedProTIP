# imagenet_r_classes = [
#     "Goldfish", "Great white shark", "Hammerhead", "Stingray", "Hen", "Ostrich", 
#     "Goldfinch", "Junco", "Bald eagle", "Vulture", "Newt", "Axolotl", "Tree frog", 
#     "Iguana", "African chameleon", "Cobra", "Scorpion", "Tarantula", "Centipede", 
#     "Peacock", "Lorikeet", "Hummingbird", "Toucan", "Duck", "Goose", "Black swan", 
#     "Koala", "Jellyfish", "Snail", "Lobster", "Hermit crab", "Flamingo", 
#     "American egret", "Pelican", "King penguin", "Grey whale", "Killer whale", 
#     "Sea lion", "Chihuahua", "Shih Tzu", "Afghan hound", "Basset hound", "Beagle", 
#     "Bloodhound", "Italian greyhound", "Whippet", "Weimaraner", "Yorkshire terrier", 
#     "Boston terrier", "Scottish terrier", "West Highland white terrier", 
#     "Golden retriever", "Labrador retriever", "Cocker spaniels", "Collie", 
#     "Border collie", "Rottweiler", "German shepherd dog", "Boxer", "French bulldog", 
#     "Saint Bernard", "Husky", "Dalmatian", "Pug", "Pomeranian", "Chow chow", 
#     "Pembroke Welsh corgi", "Toy poodle", "Standard poodle", "Timber wolf", "Hyena", 
#     "Red fox", "Tabby cat", "Leopard", "Snow leopard", "Lion", "Tiger", "Cheetah", 
#     "Polar bear", "Meerkat", "Ladybug", "Fly", "Bee", "Ant", "Grasshopper", 
#     "Cockroach", "Mantis", "Dragonfly", "Monarch butterfly", "Starfish", 
#     "Wood rabbit", "Porcupine", "Fox squirrel", "Beaver", "Guinea pig", "Zebra", 
#     "Pig", "Hippopotamus", "Bison", "Gazelle", "Llama", "Skunk", "Badger", 
#     "Orangutan", "Gorilla", "Chimpanzee", "Gibbon", "Baboon", "Panda", "Eel", 
#     "Clown fish", "Puffer fish", "Accordion", "Ambulance", "Assault rifle", 
#     "Backpack", "Barn", "Wheelbarrow", "Basketball", "Bathtub", "Lighthouse", 
#     "Beer glass", "Binoculars", "Birdhouse", "Bow tie", "Broom", "Bucket", 
#     "Cauldron", "Candle", "Cannon", "Canoe", "Carousel", "Castle", "Mobile phone", 
#     "Cowboy hat", "Electric guitar", "Fire engine", "Flute", "Gasmask", 
#     "Grand piano", "Guillotine", "Hammer", "Harmonica", "Harp", "Hatchet", "Jeep", 
#     "Joystick", "Lab coat", "Lawn mower", "Lipstick", "Mailbox", "Missile", 
#     "Mitten", "Parachute", "Pickup truck", "Pirate ship", "Revolver", "Rugby ball", 
#     "Sandal", "Saxophone", "School bus", "Schooner", "Shield", "Soccer ball", 
#     "Space shuttle", "Spider web", "Steam locomotive", "Scarf", "Submarine", "Tank", 
#     "Tennis ball", "Tractor", "Trombone", "Vase", "Violin", "Military aircraft", 
#     "Wine bottle", "Ice cream", "Bagel", "Pretzel", "Cheeseburger", "Hotdog", 
#     "Cabbage", "Broccoli", "Cucumber", "Bell pepper", "Mushroom", "Granny Smith", 
#     "Strawberry", "Lemon", "Pineapple", "Banana", "Pomegranate", "Pizza", "Burrito", 
#     "Espresso", "Volcano", "Baseball player", "Scuba diver", "Acorn"
# ]

# # Save to a text file
# with open("imagenet_r_label.txt", "w") as f:
#     for label in imagenet_r_classes:
#         f.write(label + "\n")

# print("ImageNet-R labels saved to 'imagenet_r_label.txt'")

import clip
print(dir(clip))
import torch
from tqdm import tqdm
import pickle

def load_labels(label_file):
    """
    Load class labels from a text file.
    
    Args:
        label_file: Path to the text file containing labels (one label per line).
    
    Returns:
        List of class names.
    """
    with open(label_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def generate_prompted_embeddings(model, class_names, prompt_template="a class of a {}.", device="cuda"):
    """
    Generate label text embeddings using a prompt template and CLIP model.
    
    Args:
        model: Pre-trained CLIP model.
        class_names: List of class names.
        prompt_template: Template to generate prompts for each class name.
        device: Device to run the model on.
    
    Returns:
        A dictionary mapping class names to their embeddings.
    """
    with torch.no_grad():
        lte_pool = {}  # Label Text Embedding (LTE) pool
        for class_name in tqdm(class_names, desc="Generating prompted embeddings"):
            # Create a prompted text using the template
            prompt = prompt_template.format(class_name)
            text_token = clip.tokenize([prompt]).to(device)  # Tokenize the prompt
            embedding = model.encode_text(text_token).cpu()  # Generate embedding
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize embedding
            lte_pool[class_name] = embedding.numpy()
    return lte_pool

# Load labels from the ImageNet-R label file
label_file = "label_embedding/imagenet-r_label.txt"  # Replace with the path to your label file
class_names = load_labels(label_file)

# Load pre-trained CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Generate label text embeddings with prompt engineering
prompt_template = "a class of a {}."
lte_pool = generate_prompted_embeddings(model, class_names, prompt_template, device=device)

# Save LTE pool to a file
output_file = "label_embedding/imagenet-r_le.pickle"  # Output file path
with open(output_file, "wb") as f:
    pickle.dump(lte_pool, f)

print(f"Label text embeddings (LTE Pool) for ImageNet-R generated and saved to {output_file}!")
