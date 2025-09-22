import os
import pickle
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor


def load_dom_file(file_path):
    """
    Load DomainNet class names and index mapping from a Python file containing
    a dictionary named `domainnet_classnames` mapping {index(int): name(str)}.
    """
    import importlib.util

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Class name file {file_path} not found.")

    spec = importlib.util.spec_from_file_location("class_names", file_path)
    class_names_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(class_names_module)

    if not hasattr(class_names_module, "domainnet_classnames"):
        raise AttributeError(f"{file_path} does not contain 'domainnet_classnames' dictionary.")

    domainnet_classnames = class_names_module.domainnet_classnames
    # Sort by numeric index, preserve given indices
    pairs = sorted(domainnet_classnames.items(), key=lambda x: x[0])
    class_names = [name for idx, name in pairs]
    class_to_index = {name: idx for idx, name in pairs}
    return class_names, class_to_index


@torch.no_grad()
def generate_prompted_embeddings(model, processor, class_names, class_to_index,
                                 prompt_template="a class of a {}.",
                                 device="cuda"):
    """
    Generate label text embeddings (LTE) using a prompt template and CLIP model (HF transformers).
    Returns: dict[int -> np.ndarray] mapping class index to normalized embedding.
    """
    lte_pool = {}
    for class_name in tqdm(class_names, desc="Generating prompted embeddings"):
        prompt = prompt_template.format(class_name)
        inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_text_features(**inputs)           # (1, D)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        lte_pool[class_to_index[class_name]] = feats.cpu().numpy()  # (1, D)
    return lte_pool


@torch.no_grad()
def generate_domainnet_prompted_embeddings(model, processor, class_names, class_to_index,
                                           prompt_template="a class of a {} of domain {}.",
                                           device="cuda"):
    """
    Domain-aware LTEs. For each domain, make a domain-specific prompt and store
    embeddings at indices: base_idx + domain_idx * num_classes.
    """
    domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    num_classes = len(class_names)
    lte_pool = {}

    for dom_idx, dom_name in enumerate(domain_names):
        for class_name in tqdm(class_names, desc=f"Embeddings for domain={dom_name}", leave=False):
            # NOTE: template expects two {} placeholders: class then domain
            prompt = prompt_template.format(class_name, dom_name)
            inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model.get_text_features(**inputs)       # (1, D)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            key = class_to_index[class_name] + dom_idx * num_classes
            lte_pool[key] = feats.cpu().numpy()             # (1, D)
    return lte_pool


if __name__ == "__main__":
    # ----------------- Config -----------------
    base_dir = "../data/DomainNet"
    class_file = "./utils/datautils/class_names.py"  # must define domainnet_classnames dict
    output_dir = "./label_embedding"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------- Load classes -------------
    class_names, class_to_index = load_dom_file(class_file)
    print(f"Loaded {len(class_names)} classes.")

    # --------------- Load CLIP ----------------
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # ---- Option A: single (domain-agnostic) LTEs ----
    # prompt_template = "a class of a {}."
    # lte_pool = generate_prompted_embeddings(model, processor, class_names, class_to_index,
    #                                         prompt_template=prompt_template, device=device)
    # out_path = os.path.join(output_dir, "DomainNet_le.pickle")

    # ---- Option B: domain-aware LTEs (6*|classes| entries) ----
    prompt_template = "a class of a {} of domain {}."
    lte_pool = generate_domainnet_prompted_embeddings(model, processor, class_names, class_to_index,
                                                      prompt_template=prompt_template, device=device)
    out_path = os.path.join(output_dir, "DomainIL_le.pickle")

    # ----------------- Save -------------------
    with open(out_path, "wb") as f:
        pickle.dump(lte_pool, f)
    print(f"Indexed label text embeddings (LTE Pool) saved to: {out_path}")


# import clip
# import torch
# from tqdm import tqdm

# # def load_labels(label_file):
# #     """
# #     Load class labels from a text file.
    
# #     Args:
# #         label_file: Path to the text file containing labels (one label per line).
    
# #     Returns:
# #         List of class names.
# #     """
# #     with open(label_file, "r") as f:
# #         labels = [line.strip() for line in f.readlines()]
# #     return labels

# # def generate_prompted_embeddings(model, class_names, prompt_template="a class of a {}.", device="cuda"):
# #     """
# #     Generate label text embeddings using a prompt template and CLIP model.
    
# #     Args:
# #         model: Pre-trained CLIP model.
# #         class_names: List of class names.
# #         prompt_template: Template to generate prompts for each class name.
# #         device: Device to run the model on.
    
# #     Returns:
# #         A dictionary mapping class names to their embeddings.
# #     """
# #     with torch.no_grad():
# #         lte_pool = {}  # Label Text Embedding (LTE) pool
# #         for class_name in tqdm(class_names, desc="Generating prompted embeddings"):
# #             # Create a prompted text using the template
# #             prompt = prompt_template.format(class_name)
# #             text_token = clip.tokenize([prompt]).to(device)  # Tokenize the prompt
# #             embedding = model.encode_text(text_token).cpu()  # Generate embedding
# #             embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize embedding
# #             lte_pool[class_name] = embedding.numpy()
# #     return lte_pool

# # # Load labels from a text file
# # label_file = "cifar100_label.txt"  # Replace with the path to your label file
# # class_names = load_labels(label_file)

# # # Load pre-trained CLIP model
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # model, preprocess = clip.load("ViT-B/32", device=device)

# # # Generate label text embeddings with prompt engineering
# # prompt_template = "a class of a {}."
# # lte_pool = generate_prompted_embeddings(model, class_names, prompt_template, device=device)

# # # Save LTE pool to a file
# # import pickle
# # output_file = "lte_pool.pkl"  # Output file path
# # with open(output_file, "wb") as f:
# #     pickle.dump(lte_pool, f)

# # print(f"Label text embeddings (LTE Pool) generated and saved to {output_file}!")
# import clip
# import torch
# from tqdm import tqdm
# import os
# import pickle


# def load_dom_file(file_path):
#     """
#     Load class names and index mapping from a Python file containing 
#     a dictionary named `domainnet_classnames`.

#     Args:
#         file_path: Path to the Python file (e.g., class_names.py)

#     Returns:
#         class_names: List of class names in index order.
#         class_to_index: Dictionary mapping class name to index.
#     """
#     import importlib.util

#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Class name file {file_path} not found.")

#     spec = importlib.util.spec_from_file_location("class_names", file_path)
#     class_names_module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(class_names_module)

#     if not hasattr(class_names_module, "domainnet_classnames"):
#         raise AttributeError(f"{file_path} does not contain 'domainnet_classnames' dictionary.")

#     domainnet_classnames = class_names_module.domainnet_classnames

#     # Sort by index to get an ordered class_names list
#     class_names = [name for idx, name in sorted(domainnet_classnames.items(), key=lambda x: x[0])]
#     class_to_index = {name: idx for idx, name in sorted(domainnet_classnames.items(), key=lambda x: x[0])}

#     return class_names, class_to_index

# def load_supercategories(supercategory_file):
#     """
#     Load supercategories and their associated classes from a file.
#     Args:
#         supercategory_file: Path to the supercategories.txt file.
#     Returns:
#         class_names: List of all class names in order.
#         class_to_index: Dictionary mapping class names to indices (0–99).
#     """
#     class_names = []
#     class_to_index = {}
#     current_index = 0

#     if not os.path.exists(supercategory_file):
#         raise FileNotFoundError(f"Supercategory file {supercategory_file} not found.")
    
#     with open(supercategory_file, "r") as f:
#         for line in f:
#             line = line.strip()
#             if line and not line.startswith("#"):
#                 _, classes = line.split(":")
#                 class_list = [cls.strip() for cls in classes.split(",")]
#                 class_names.extend(class_list)
#                 for cls in class_list:
#                     class_to_index[cls] = current_index
#                     current_index += 1
#     return class_names, class_to_index


# def generate_prompted_embeddings(model, class_names, class_to_index, prompt_template="a class of a {}.", device="cuda"):
#     """
#     Generate label text embeddings (LTE) using a prompt template and CLIP model.
#     Args:
#         model: Pre-trained CLIP model.
#         class_names: List of class names.
#         class_to_index: Dictionary mapping class names to indices.
#         prompt_template: Template to generate prompts for each class name.
#         device: Device to run the model on.
#     Returns:
#         A dictionary mapping class indices to their embeddings.
#     """
#     with torch.no_grad():
#         lte_pool = {}  # Label Text Embedding (LTE) pool
#         for class_name in tqdm(class_names, desc="Generating prompted embeddings"):
#             # Create a prompted text using the template
#             prompt = prompt_template.format(class_name)
#             text_token = clip.tokenize([prompt]).to(device)  # Tokenize the prompt
#             embedding = model.encode_text(text_token).cpu()  # Generate embedding
#             embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize embedding
#             lte_pool[class_to_index[class_name]] = embedding.numpy()  # Use index as the key
#     return lte_pool



# def generate_domainnet_prompted_embeddings(model, class_names, class_to_index, prompt_template="a class of a {} of domain {}.", device="cuda"):
#     """
#     Generate label text embeddings (LTE) using a prompt template and CLIP model.
#     Args:
#         model: Pre-trained CLIP model.
#         class_names: List of class names.
#         class_to_index: Dictionary mapping class names to indices.
#         prompt_template: Template to generate prompts for each class name.
#         device: Device to run the model on.
#     Returns:
#         A dictionary mapping class indices to their embeddings.
#     """
#     domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
#     with torch.no_grad():
#         lte_pool = {}  # Label Text Embedding (LTE) pool
#         for dom_idx in range(6):
#             dom_name = domain_names[dom_idx]
#             for class_name in tqdm(class_names, desc="Generating prompted embeddings"):
#                 # Create a prompted text using the template
#                 prompt = prompt_template.format(class_name)
#                 text_token = clip.tokenize([prompt]).to(device)  # Tokenize the prompt
#                 embedding = model.encode_text(text_token).cpu()  # Generate embedding
#                 embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize embedding
#                 lte_pool[class_to_index[class_name] + dom_idx * 345] = embedding.numpy()  # Use index as the key
#     return lte_pool

# # # Set up file paths and configurations
# # base_dir = "../data/dn4il"
# # supercategory_file = os.path.join(base_dir, "supercategories.txt")

# # # Load class names and mapping to indices
# # print("Loading class names and indices from supercategories...")
# # class_names, class_to_index = load_supercategories(supercategory_file)
# # print(f"Loaded {len(class_names)} classes.")

# # # Load pre-trained CLIP model
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # print("Loading CLIP model...")
# # model, preprocess = clip.load("ViT-B/32", device=device)

# # # Generate label text embeddings with prompt engineering
# # print("Generating label text embeddings...")
# # prompt_template = "a class of a {}."
# # lte_pool = generate_prompted_embeddings(model, class_names, class_to_index, prompt_template, device=device)

# # # Save LTE pool to a file
# # output_file = os.path.join('./label_embedding', "DomainNet_le.pickle")
# # with open(output_file, "wb") as f:
# #     pickle.dump(lte_pool, f)

# # print(f"Indexed label text embeddings (LTE Pool) generated and saved to {output_file}!")




# # Set up file paths and configurations
# base_dir = "../data/DomainNet"

# # Load class names and mapping to indices
# class_names, class_to_index = load_dom_file('./utils/datautils/class_names.py')
# print(f"Loaded {len(class_names)} classes.")

# # Load pre-trained CLIP model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Loading CLIP model...")
# model, preprocess = clip.load("ViT-B/32", device=device)

# # Generate label text embeddings with prompt engineering
# print("Generating label text embeddings...")
# prompt_template = "a class of a {}."
# lte_pool = generate_domainnet_prompted_embeddings(model, class_names, class_to_index, prompt_template, device=device)

# # Save LTE pool to a file
# output_file = os.path.join('./label_embedding', "DomainIL_le.pickle")
# with open(output_file, "wb") as f:
#     pickle.dump(lte_pool, f)

# print(f"Indexed label text embeddings (LTE Pool) generated and saved to {output_file}!")

