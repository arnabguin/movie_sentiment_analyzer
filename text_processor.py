from tensorflow.keras.preprocessing.text import Tokenizer

def demonstrate_tokenizer():
    # Example texts
    texts = [
        "I love this movie",
        "I hate this movie",
        "great movie",
        "terrible movie"
    ]

    # Create and fit tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    # Print the word index (vocabulary)
    print("\nWord Index (vocabulary):")
    print(tokenizer.word_index)

    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    print("\nOriginal texts and their sequences:")
    for text, seq in zip(texts, sequences):
        print(f"Text: '{text}'")
        print(f"Sequence: {seq}")

    # Try with new text
    new_text = ["I love great movies"]
    new_seq = tokenizer.texts_to_sequences(new_text)
    print("\nNew text conversion:")
    print(f"Text: '{new_text[0]}'")
    print(f"Sequence: {new_seq[0]}")
    print("\nNote: 'movies' is not in sequence because it wasn't in training vocabulary")

if __name__ == "__main__":
    demonstrate_tokenizer()