import clip
import torch
from tqdm import tqdm

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

# Load labels from a text file
label_file = "cifar100_label.txt"  # Replace with the path to your label file
class_names = load_labels(label_file)

# Load pre-trained CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Generate label text embeddings with prompt engineering
prompt_template = "a class of a {}."
lte_pool = generate_prompted_embeddings(model, class_names, prompt_template, device=device)

# Save LTE pool to a file
import pickle
output_file = "lte_pool.pkl"  # Output file path
with open(output_file, "wb") as f:
    pickle.dump(lte_pool, f)

print(f"Label text embeddings (LTE Pool) generated and saved to {output_file}!")
