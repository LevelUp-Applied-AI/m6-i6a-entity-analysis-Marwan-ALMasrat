"""
Module 6 Week A — Integration: Entity Analysis Pipeline

Build a corpus-level entity analysis pipeline that preprocesses
climate articles (with language-aware handling), extracts entities,
computes statistics, and produces visualizations.

Run: python entity_analysis.py
"""

import unicodedata
from itertools import combinations
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import spacy


def load_corpus(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    df = pd.read_csv(filepath)
    return df


def preprocess_corpus(df):
    """Add a language-aware `processed_text` column to the corpus.

    For every row, apply Unicode NFC normalization to `text` so that
    visually identical characters (composed vs. decomposed diacritics)
    compare equal downstream. The processed form preserves
    capitalization and punctuation — those are signals NER depends on.

    For Arabic rows (`language == 'ar'`), do not attempt English NLP
    processing: either pass the NFC-normalized text through unchanged
    or store an empty string. Either choice must not crash the
    pipeline.

    Args:
        df: DataFrame returned by load_corpus.

    Returns:
        Copy of df with a new `processed_text` column. The original
        `text` column is left intact so NER can still consume it.
    """
    result = df.copy()

    def process_row(row):
        normalized = unicodedata.normalize('NFC', str(row['text']))
        if row['language'] == 'en':
            return normalized
        else:
            # Arabic or other non-English: pass NFC text through unchanged
            return normalized

    result['processed_text'] = result.apply(process_row, axis=1)
    return result


def run_ner_pipeline(df, nlp):
    """Run spaCy NER on the English rows of a preprocessed corpus.

    Args:
        df: DataFrame with columns id, text, language, processed_text.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    english_df = df[df['language'] == 'en'].copy()

    rows = []
    for _, article in english_df.iterrows():
        doc = nlp(str(article['text']))
        for ent in doc.ents:
            rows.append({
                'text_id':      article['id'],
                'entity_text':  ent.text,
                'entity_label': ent.label_,
                'start_char':   ent.start_char,
                'end_char':     ent.end_char,
            })

    entity_df = pd.DataFrame(rows, columns=[
        'text_id', 'entity_text', 'entity_label', 'start_char', 'end_char'
    ])
    return entity_df


def aggregate_entity_stats(entity_df, articles_df):
    """Compute frequency, co-occurrence, and per-category statistics.

    Args:
        entity_df: DataFrame with columns text_id, entity_text,
                   entity_label.
        articles_df: The source corpus DataFrame (with columns id,
                     category, ...). Used to join category onto
                     each entity for per-category aggregation.

    Returns:
        Dictionary with keys:
          'top_entities': DataFrame of top 20 entities by frequency
                          (columns: entity_text, entity_label, count)
          'label_counts': dict of entity_label -> total count
          'co_occurrence': DataFrame of entity pairs appearing in the
                           same text (columns: entity_a, entity_b,
                           co_count). Cap at top 50 pairs by co_count.
          'per_category': DataFrame of entity-label counts broken out
                          by article category (columns: category,
                          entity_label, count)
    """
    # ── 1. Top 20 most frequent entities ──────────────────────────────
    freq = (
        entity_df
        .groupby(['entity_text', 'entity_label'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    # ── 2. Total count per entity label ───────────────────────────────
    label_counts = (
        entity_df
        .groupby('entity_label')
        .size()
        .to_dict()
    )

    # ── 3. Co-occurrence (entity pairs in the same text) ──────────────
    co_counter = defaultdict(int)

    for text_id, group in entity_df.groupby('text_id'):
        unique_entities = (
            group[['entity_text', 'entity_label']]
            .drop_duplicates()
            ['entity_text']
            .tolist()
        )
        for a, b in combinations(sorted(set(unique_entities)), 2):
            co_counter[(a, b)] += 1

    co_rows = [
        {'entity_a': a, 'entity_b': b, 'co_count': cnt}
        for (a, b), cnt in co_counter.items()
        if cnt >= 2          # filter out hapax pairs
    ]
    co_occurrence = (
        pd.DataFrame(co_rows, columns=['entity_a', 'entity_b', 'co_count'])
        .sort_values('co_count', ascending=False)
        .head(50)
        .reset_index(drop=True)
    )

    # ── 4. Per-category entity-label counts ───────────────────────────
    merged = entity_df.merge(
        articles_df[['id', 'category']],
        left_on='text_id',
        right_on='id',
        how='left'
    )
    per_category = (
        merged
        .groupby(['category', 'entity_label'])
        .size()
        .reset_index(name='count')
        .sort_values(['category', 'count'], ascending=[True, False])
        .reset_index(drop=True)
    )

    # ── 5. Console summary ────────────────────────────────────────────
    print("\n── Entity Statistics Summary ──────────────────────────────")
    print(f"Total entities extracted : {len(entity_df):,}")
    print(f"Unique entity labels     : {len(label_counts)}")
    print(f"Top label                : {max(label_counts, key=label_counts.get)!r} "
          f"({max(label_counts.values()):,} occurrences)")
    print(f"Co-occurrence pairs kept : {len(co_occurrence)}")
    print("────────────────────────────────────────────────────────────\n")

    return {
        'top_entities': freq,
        'label_counts': label_counts,
        'co_occurrence': co_occurrence,
        'per_category':  per_category,
    }


def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    """Create a bar chart of the top 20 entities by frequency.

    Args:
        stats: Dictionary from aggregate_entity_stats (must contain
               'top_entities' DataFrame).
        output_path: File path to save the chart.
    """
    top = stats['top_entities'].copy()

    # Assign a distinct colour to every unique entity label
    unique_labels = top['entity_label'].unique()
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}
    colors = top['entity_label'].map(label_to_color).tolist()

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(
        y=top['entity_text'],
        width=top['count'],
        color=colors,
        edgecolor='white',
        linewidth=0.6,
    )

    # Invert y-axis so highest frequency is at the top
    ax.invert_yaxis()

    # Value labels on bars
    for bar, val in zip(bars, top['count']):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va='center',
            ha='left',
            fontsize=9,
        )

    # Legend for entity labels
    legend_patches = [
        mpatches.Patch(color=label_to_color[lbl], label=lbl)
        for lbl in unique_labels
    ]
    ax.legend(
        handles=legend_patches,
        title='Entity Type',
        bbox_to_anchor=(1.01, 1),
        loc='upper left',
        fontsize=9,
        title_fontsize=10,
    )

    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Entity', fontsize=12)
    ax.set_title('Top 20 Most Frequent Named Entities — Climate Articles Corpus',
                 fontsize=13, fontweight='bold', pad=14)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(stats, co_occurrence):
    """Generate a text summary of entity analysis findings.

    Args:
        stats: Dictionary from aggregate_entity_stats.
        co_occurrence: Co-occurrence DataFrame from stats.

    Returns:
        String containing a structured report with: entity counts
        per type, top 5 most frequent entities, top 3 co-occurring
        pairs, and a brief summary.
    """
    lines = []
    sep = "=" * 56

    # ── Header ────────────────────────────────────────────────────────
    lines.append(sep)
    lines.append("  ENTITY ANALYSIS REPORT — Climate Articles Corpus")
    lines.append(sep)

    # ── Section 1: Entity counts per type ─────────────────────────────
    lines.append("\n[1] Entity Counts by Type")
    lines.append("-" * 40)
    label_counts = stats['label_counts']
    total = sum(label_counts.values())
    for label, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        lines.append(f"  {label:<14} {cnt:>6,}  ({pct:.1f}%)")
    lines.append(f"\n  Total entities: {total:,}")

    # ── Section 2: Top 5 most frequent entities ────────────────────────
    lines.append("\n[2] Top 5 Most Frequent Entities")
    lines.append("-" * 40)
    top5 = stats['top_entities'].head(5)
    for rank, (_, row) in enumerate(top5.iterrows(), start=1):
        lines.append(
            f"  {rank}. {row['entity_text']!r:<30}  "
            f"[{row['entity_label']}]  ×{row['count']}"
        )

    # ── Section 3: Top 3 co-occurring entity pairs ────────────────────
    lines.append("\n[3] Top 3 Co-occurring Entity Pairs")
    lines.append("-" * 40)
    if co_occurrence is not None and len(co_occurrence) > 0:
        top3_co = co_occurrence.head(3)
        for _, row in top3_co.iterrows():
            lines.append(
                f"  • {row['entity_a']!r}  ↔  {row['entity_b']!r}"
                f"  (co_count={row['co_count']})"
            )
    else:
        lines.append("  No co-occurrence data available.")

    # ── Section 4: Summary paragraph ──────────────────────────────────
    lines.append("\n[4] Summary")
    lines.append("-" * 40)

    # Dynamically derive the dominant label and top entity for the narrative
    top_label = max(label_counts, key=label_counts.get)
    top_entity_row = stats['top_entities'].iloc[0]

    if co_occurrence is not None and len(co_occurrence) > 0:
        top_pair = co_occurrence.iloc[0]
        co_sentence = (
            f"The most frequently co-occurring pair is "
            f"{top_pair['entity_a']!r} and {top_pair['entity_b']!r} "
            f"(appearing together in {top_pair['co_count']} texts), "
            f"suggesting a tight thematic link between these entities in "
            f"climate discourse."
        )
    else:
        co_sentence = "Co-occurrence data was not available for this corpus."

    summary = (
        f"The corpus yields {total:,} named entities across "
        f"{len(label_counts)} distinct types. {top_label} entities are "
        f"the most prevalent, accounting for "
        f"{label_counts[top_label]/total*100:.1f}% of all extractions, "
        f"which reflects the heavy presence of institutional actors "
        f"(governments, research bodies, NGOs) typical of climate policy "
        f"and science reporting. The single most mentioned entity is "
        f"{top_entity_row['entity_text']!r} "
        f"(type: {top_entity_row['entity_label']}, "
        f"frequency: {top_entity_row['count']}), underscoring its "
        f"centrality to climate narratives. {co_sentence}"
    )
    # Word-wrap the paragraph at ~70 chars for readability
    import textwrap
    for wrapped_line in textwrap.wrap(summary, width=70):
        lines.append("  " + wrapped_line)

    lines.append("\n" + sep)

    return "\n".join(lines)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    # Load and preprocess the corpus
    raw = load_corpus()
    if raw is not None:
        corpus = preprocess_corpus(raw)
        if corpus is not None:
            print(f"Corpus: {len(corpus)} articles")
            print(f"Languages: {corpus['language'].value_counts().to_dict()}")
            print(f"Categories: {corpus['category'].value_counts().to_dict()}")

            # Run NER on English rows
            entities = run_ner_pipeline(corpus, nlp)
            if entities is not None:
                print(f"\nExtracted {len(entities)} entities")

                # Aggregate statistics
                stats = aggregate_entity_stats(entities, corpus)
                if stats is not None:
                    print(f"\nLabel counts: {stats['label_counts']}")
                    print(f"\nTop 5 entities:")
                    print(stats["top_entities"].head())
                    print(f"\nPer-category counts (head):")
                    print(stats["per_category"].head())

                    # Visualize
                    visualize_entity_distribution(stats)
                    print("\nVisualization saved to entity_distribution.png")

                    # Generate report
                    report = generate_report(stats, stats.get("co_occurrence"))
                    if report is not None:
                        print(f"\n{'='*50}")
                        print(report)