import os
import json
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Initialize the CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

model.to(device)

# Path to the folder containing your images
image_folder = "final_images"

# Path to the metadata file in JSONL format
metadata_file = "testimages/metadata.jsonl"

# Read metadata from JSONL file
metadata_list = []
with open(metadata_file, 'r') as file:
    for line in file:
        metadata_list.append(json.loads(line))

# Extract image filenames and captions from metadata
image_filenames = [metadata['file_name'] for metadata in metadata_list]
original_captions = [metadata['title'] for metadata in metadata_list]

# Initialize lists to store generated captions and similarity scores
generated_captions = []
bleu_scores = []
meteor_scores = []
rouge_scores = []

# Loop through each image in the dataset
for image_filename, original_caption in zip(image_filenames, original_captions):
    # Load the image
    image_path = os.path.join(image_folder, image_filename)
    image = Image.open(image_path)

    # Tokenize the input text
    tokenized_text = processor.tokenize(original_caption, return_tensors="pt", padding=True, truncation=True).to(device)

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate the caption using CLIP
    with torch.no_grad():
        outputs = model(input_ids=tokenized_text.input_ids, pixel_values=inputs.pixel_values)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Decode the generated caption
    generated_caption = processor.decode(probs.argmax(-1))

    # Tokenize hypothesis and reference sentences
    tokenized_hypothesis = word_tokenize(generated_caption.lower())
    tokenized_reference = word_tokenize(original_caption.lower())

    # Store the generated caption
    generated_captions.append(generated_caption)

    # Compute BLEU score for the generated caption
    bleu_score = sentence_bleu([tokenized_reference], tokenized_hypothesis)
    bleu_scores.append(bleu_score)

    # Compute METEOR score for the generated caption
    meteor_score_val = meteor_score([tokenized_reference], tokenized_hypothesis)
    meteor_scores.append(meteor_score_val)

    # Compute ROUGE scores for the generated caption
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores_val = scorer.score(original_caption, generated_caption)
    rouge_scores.append(rouge_scores_val)

# Print the generated captions and similarity scores
for i in range(len(image_filenames)):
    print("Image Filename:", image_filenames[i])
    print("Original Caption:", original_captions[i])
    print("Generated Caption:", generated_captions[i])
    print("BLEU Score:", bleu_scores[i])
    print("METEOR Score:", meteor_scores[i])
    print("ROUGE Scores:", rouge_scores[i])
    print()

# Compute average scores
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
average_meteor_score = sum(meteor_scores) / len(meteor_scores)
average_rouge1_f1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
average_rougeL_f1 = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)

print("Average BLEU Score:", average_bleu_score)
print("Average METEOR Score:", average_meteor_score)
print("Average ROUGE-1 F1 Score:", average_rouge1_f1)
print("Average ROUGE-L F1 Score:", average_rougeL_f1)


