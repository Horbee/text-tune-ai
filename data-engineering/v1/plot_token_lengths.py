import numpy as np
import matplotlib.pyplot as plt

# Calculate lengths


def plot_lengths(tokenizer, ds_train, prefix):
    src_lengths = [len(tokenizer.encode(prefix + x["de_corrupted"]))
                   for x in ds_train]
    tgt_lengths = [len(tokenizer.encode(x["de_correct"])) for x in ds_train]

    # Stats
    print(f"🔤 Max src length: {max(src_lengths)}")
    print(f"🔤 Max tgt length: {max(tgt_lengths)}")
    print(f"📊 95th percentile src length: {np.percentile(src_lengths, 95)}")
    print(f"📊 95th percentile tgt length: {np.percentile(tgt_lengths, 95)}")

    # Optional: Plot histograms
    plt.hist(src_lengths, bins=50, alpha=0.6,
             label=f'de_corrupted ({prefix}input)'
             )
    plt.hist(tgt_lengths, bins=50, alpha=0.6, label='de_correct (corrected)')
    plt.axvline(128, color='red', linestyle='--', label='max_length=128')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Token Lengths')
    plt.show()
