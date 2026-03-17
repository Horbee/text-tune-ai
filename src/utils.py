def process_in_batches(sentences: list[str], batch_size: int = 10, verbose: bool = True):
    for i in range(0, len(sentences), batch_size):
        if verbose:
            print(f"Processing batch {i//batch_size + 1} ({i} to {min(i + batch_size, len(sentences))} of {len(sentences)})...")
        yield sentences[i:i + batch_size]