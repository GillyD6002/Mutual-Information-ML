import re
import numpy as np

# This module builds fixed-length numeric sequences from natural-language
# text, for use as a 1D analog of the 2D image datasets in src/image.py (via
# src/sequence.py's generic 1D splicing/bipartition logic). It is a NEW
# module: src/image.py and src/mine.py are not modified by anything here.
#
# ------------------------------------------------------------------------
# Design choices, made explicit here rather than left implicit:
#
# - CORPUS: the 18 public-domain texts bundled with NLTK's "gutenberg"
#   corpus (Austen, Melville, Shakespeare, the King James Bible, etc.,
#   ~2.6 million words combined). This is a standard, well-tested corpus
#   access path (nltk.download("gutenberg")) that avoids hand-writing
#   Project Gutenberg header/footer-stripping logic.
#
# - WHAT EACH "TOKEN" IS TURNED INTO - a single scalar, not an embedding:
#   every word is mapped to log(rank), where `rank` is that word's
#   frequency rank (1 = most common) in the whole corpus. This is the
#   "quick-and-crude" encoding discussed before implementing: it reuses
#   the exact 1D pipeline built for stock-return sequences with zero
#   architecture changes (each sequence is still a plain
#   (num_samples, length) array of scalars, matching
#   ks.Input(shape=(length,))), at the cost of being a much cruder proxy
#   for linguistic structure than a real embedding would be - it captures
#   *how common* a word is, not *what it means*, so any MI this pipeline
#   finds reflects positional structure in word COMMONNESS (e.g. function
#   words like "the" clustering in predictable positions relative to
#   content words), not semantic/topical coherence. log(rank) rather than
#   raw rank is used because word frequency in natural language follows an
#   approximately Zipfian (power-law) distribution, so log-rank compresses
#   the heavy-tailed raw rank into a much more tractable range - the
#   standard transform for this kind of data. The log-ranks are further
#   standardized (zero mean, unit variance) purely so the network trains
#   on an O(1)-scale input, matching ordinary good practice.
#
# - WINDOWING: each sample is a window of `length` CONSECUTIVE tokens
#   (native word-by-word resolution, no subsampling - the stock-return
#   experiment found that subsampling coarser than the native resolution
#   can silently redefine what "window length" means and obscure real
#   short-range dependence), drawn from a uniformly random starting
#   position within one randomly-chosen source text. Windows are allowed to
#   overlap (there is no requirement that samples be disjoint - the "same
#   underlying process, many independent-ish draws" framing here is closer
#   to "many windows onto a large corpus" than the stock experiment's "one
#   window per company").

GUTENBERG_FILEIDS = None # None = use every text in the corpus

DEFAULT_NUM_POINTS = 100
DEFAULT_NUM_SEQUENCES = 3000

# Fixed (not None) by default, matching src/stocks.py's convention: repeated
# calls with the same seed draw the exact same set of windows, so a length
# sweep varies only the partition length, not the underlying dataset.
DEFAULT_SEED = 20170101

_TOKEN_CACHE = {}

def _ensure_corpus_downloaded():

    import nltk
    try:
        nltk.data.find("corpora/gutenberg")
    except LookupError:
        nltk.download("gutenberg")

def get_book_tokens(fileids = GUTENBERG_FILEIDS):

    # This function returns a list of per-book word-token lists (lowercased,
    # alphabetic tokens only - punctuation, numbers, and contraction
    # fragments like "n't" are dropped for simplicity). Results are cached
    # in-process, since re-tokenizing ~2.6 million words is the most
    # expensive step here and this function is called once per sweep length.

    _ensure_corpus_downloaded()
    from nltk.corpus import gutenberg

    cache_key = tuple(fileids) if fileids is not None else "all"
    if cache_key in _TOKEN_CACHE:
        return _TOKEN_CACHE[cache_key]

    books = []
    for fileid in (fileids if fileids is not None else gutenberg.fileids()):
        words = [word.lower() for word in gutenberg.words(fileid) if word.isalpha()]
        books.append(words)
    _TOKEN_CACHE[cache_key] = books
    return books

def get_rank_table(books):

    # Builds a {word: log(rank)} lookup table from word-frequency rank
    # across the given books (rank 1 = most frequent word), combined into a
    # single corpus-wide vocabulary/frequency count.

    counts = {}
    for words in books:
        for word in words:
            counts[word] = counts.get(word, 0) + 1
    ranked_words = sorted(counts, key = counts.get, reverse = True)
    return {word: np.log(rank + 1) for (rank, word) in enumerate(ranked_words)}

def get_sequences(num_sequences = DEFAULT_NUM_SEQUENCES, num_points = DEFAULT_NUM_POINTS,
        seed = DEFAULT_SEED, fileids = GUTENBERG_FILEIDS):

    # This is a get_images()-style loader: it returns a (sequences, cov,
    # mean) triple, matching the shape/return convention that
    # src/image.py's get_images uses for real (non-Gaussian) datasets -
    # `sequences` has shape (num_sequences, num_points), analogous to
    # (num_samples, height, width) for images, and cov/mean are
    # placeholders (identity / zero) since these sequences are not
    # Gaussian, exactly as image.py does for its real-image sources.
    #
    # Each sequence is `num_points` consecutive words, drawn from a
    # uniformly random position within a uniformly random book, mapped
    # word-by-word through the corpus-wide log(rank) table, then
    # standardized to zero mean / unit variance.
    #
    # That standardization is computed from the actually-SAMPLED log-ranks
    # (weighted by how often words occur in real running text), not from a
    # plain average over the vocabulary list - common words like "the"
    # dominate real usage but are just one entry each in the vocabulary, so
    # averaging over the vocabulary instead would substantially
    # mis-estimate the mean/std of what real text actually looks like.

    books = get_book_tokens(fileids)
    rank_table = get_rank_table(books)

    rand = np.random.RandomState(seed)
    book_lengths = np.array([len(words) for words in books])
    valid_books = np.nonzero(book_lengths >= num_points)[0]

    raw_sequences = np.empty((num_sequences, num_points), dtype = np.float64)
    for i in range(num_sequences):
        book_index = valid_books[rand.randint(0, valid_books.shape[0])]
        words = books[book_index]
        start = rand.randint(0, len(words) - num_points + 1)
        window = words[start:start + num_points]
        raw_sequences[i] = [rank_table[word] for word in window]

    sequences = ((raw_sequences - raw_sequences.mean()) / raw_sequences.std()).astype(np.float32)

    cov = np.eye(num_points)
    mean = np.zeros(num_points)
    return (sequences, cov, mean)

DEFAULT_EMBEDDING_MODEL = "glove-wiki-gigaword-50"

_EMBEDDING_CACHE = {}

def get_embedding_model(model_name = DEFAULT_EMBEDDING_MODEL):

    # Loads (and process-caches) a pretrained GloVe embedding table via
    # gensim's downloader, which caches the model file itself under
    # ~/gensim-data on first use. "glove-wiki-gigaword-50" is the smallest
    # standard GloVe size (50-dimensional, ~66 MB download, 400,000-word
    # vocabulary) - deliberately small, since each embedded sequence below
    # is num_points x embedding_dim, and a smaller embedding_dim keeps the
    # flattened input (and therefore the first Dense layer's parameter
    # count) more in proportion to how many sequences we can practically
    # sample. Token-level vocabulary coverage against the Gutenberg corpus
    # used here is ~98.6%, so out-of-vocabulary words are rare.

    if model_name in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[model_name]
    import gensim.downloader as api
    model = api.load(model_name)
    _EMBEDDING_CACHE[model_name] = model
    return model

def get_embedded_sequences(num_sequences = DEFAULT_NUM_SEQUENCES, num_points = DEFAULT_NUM_POINTS,
        seed = DEFAULT_SEED, fileids = GUTENBERG_FILEIDS, model_name = DEFAULT_EMBEDDING_MODEL):

    # Richer alternative to get_sequences: instead of collapsing each word
    # to a single log-rank scalar, each word is mapped to its pretrained
    # GloVe embedding vector, so `sequences` has shape
    # (num_sequences, num_points, embedding_dim) rather than
    # (num_sequences, num_points) - a genuine per-token representation of
    # meaning, not just commonness. Out-of-vocabulary words (rare given the
    # ~98.6% token coverage - see get_embedding_model) are mapped to a zero
    # vector.
    #
    # This is still returned as a (sequences, cov, mean) triple like
    # get_sequences, with cov/mean now sized for the flattened
    # (num_points * embedding_dim) representation, purely for interface
    # consistency - they are unused by the embedded-sequence pipeline.
    #
    # No changes are needed anywhere in src/sequence.py for this 3D input:
    # numpy's `array[:, left:right]` slicing (used throughout
    # get_finite_sequence_dataset) leaves any trailing dimensions - here,
    # embedding_dim - fully intact automatically, and mine.py's
    # Model.build_model already accepts any input shape via
    # ks.Input(shape=image_shape) + Flatten(). See src/sequence.py's module
    # docstring for the general version of this argument.

    books = get_book_tokens(fileids)
    model = get_embedding_model(model_name)
    embedding_dim = model.vector_size

    rand = np.random.RandomState(seed)
    book_lengths = np.array([len(words) for words in books])
    valid_books = np.nonzero(book_lengths >= num_points)[0]

    sequences = np.zeros((num_sequences, num_points, embedding_dim), dtype = np.float32)
    for i in range(num_sequences):
        book_index = valid_books[rand.randint(0, valid_books.shape[0])]
        words = books[book_index]
        start = rand.randint(0, len(words) - num_points + 1)
        window = words[start:start + num_points]
        for (t, word) in enumerate(window):
            if word in model.key_to_index:
                sequences[i, t] = model[word]

    cov = np.eye(num_points * embedding_dim)
    mean = np.zeros(num_points * embedding_dim)
    return (sequences, cov, mean)

def shuffle_word_order(sequences, seed = None):

    # Negative-control helper: independently shuffles the WORD ORDER within
    # each sequence, destroying positional/sequential structure while
    # preserving each sequence's own multiset of (log-rank) values -
    # directly analogous to image.shuffle_pixels_independent. Used to
    # check that any MI found by the main experiment reflects genuine
    # sequential dependence, not just a "bag of words" artifact.

    rand = np.random.RandomState(seed)
    shuffled = np.empty_like(sequences)
    for i in range(sequences.shape[0]):
        shuffled[i] = sequences[i, rand.permutation(sequences.shape[1])]
    return shuffled
