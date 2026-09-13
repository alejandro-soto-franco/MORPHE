import pandas as pd

from morphe.embeddings.autoencoder import export_region_images

# pyrefly: ignore [unknown-name]
rgb_df = pd.read_csv(snakemake.input.rgb_csv)
# pyrefly: ignore [unknown-name]
out_dir = snakemake.output[0]
# pyrefly: ignore [unknown-name]
image_size = snakemake.config["embeddings"]["region_image_size"]

export_region_images(rgb_df, out_dir, image_size=image_size)
