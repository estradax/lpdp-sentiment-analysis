import os
import logging
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from wordcloud import WordCloud

logger = logging.getLogger(__name__)

# Consistent colors for the 3 sentiment classes
SENTIMENT_COLORS = {
    "Positif": "#2ecc71",
    "Netral":  "#3498db",
    "Negatif": "#e74c3c",
}


def plot_sentiment_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Generates and saves bar and pie charts of sentiment label distribution."""
    logger.info("Generating sentiment distribution plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    dist = df["label"].value_counts()
    labels = dist.index.tolist()
    counts = dist.values.tolist()
    colors = [SENTIMENT_COLORS.get(l, "#95a5a6") for l in labels]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribusi Kelas Sentimen Tweet LPDP", fontsize=14, fontweight="bold")
    
    # 1. Bar chart
    bars = axes[0].bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_xlabel("Kelas Sentimen", fontsize=11)
    axes[0].set_ylabel("Jumlah Tweet", fontsize=11)
    axes[0].set_title("Bar Chart", fontsize=12)
    for bar, count in zip(bars, counts):
        percentage = (count / len(df)) * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{count:,}\n({percentage:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
        
    # 2. Pie chart
    axes[1].pie(
        counts, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Pie Chart", fontsize=12)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "sentiment_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved distribution plot to {plot_path}")


def generate_wordclouds(df: pd.DataFrame, output_dir: str) -> None:
    """Generates and saves overall wordcloud and wordclouds per sentiment class."""
    logger.info("Generating wordclouds...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. WordCloud for all tweets
    corpus_all = " ".join(df["processed_text"].dropna())
    if corpus_all.strip():
        wc_all = WordCloud(
            background_color="white",
            max_words=200,
            colormap="viridis",
            width=1000, height=500,
            collocations=False,
        ).generate(corpus_all)
        
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.imshow(wc_all, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("WordCloud — Semua Tweet LPDP (setelah preprocessing)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        
        plot_path_all = os.path.join(output_dir, "wordcloud_all.png")
        plt.savefig(plot_path_all, dpi=300)
        plt.close(fig)
        logger.info(f"Saved overall wordcloud to {plot_path_all}")
    else:
        logger.warning("Empty corpus. Overall WordCloud skipped.")
        
    # 2. WordCloud per Sentiment Class
    cmap_map = {"Positif": "Greens", "Netral": "Blues", "Negatif": "Reds"}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("WordCloud per Kelas Sentimen", fontsize=14, fontweight="bold")
    
    classes = ["Positif", "Netral", "Negatif"]
    for i, kelas in enumerate(classes):
        ax = axes[i]
        subset = df[df["label"] == kelas]["processed_text"]
        
        if subset.empty or not " ".join(subset.dropna()).strip():
            ax.axis("off")
            ax.set_title(f"{kelas} (tidak ada data)", fontsize=11)
            continue
            
        corpus = " ".join(subset.dropna())
        wc = WordCloud(
            background_color="white", max_words=150,
            colormap=cmap_map.get(kelas, "Blues"),
            width=800, height=400, collocations=False,
        ).generate(corpus)
        
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            f"Sentimen: {kelas}  ({len(subset):,} tweet)",
            fontsize=11, fontweight="bold",
            color=SENTIMENT_COLORS.get(kelas, "black"),
        )
        
    plt.tight_layout()
    plot_path_classes = os.path.join(output_dir, "wordcloud_by_sentiment.png")
    plt.savefig(plot_path_classes, dpi=300)
    plt.close(fig)
    logger.info(f"Saved sentiment-specific wordclouds to {plot_path_classes}")


def plot_top_words(df: pd.DataFrame, output_dir: str, top_n: int = 20) -> None:
    """Plots and saves the horizontal bar chart of the top N most frequent words."""
    logger.info(f"Generating top {top_n} words plot...")
    os.makedirs(output_dir, exist_ok=True)
    
    all_words = " ".join(df["processed_text"].dropna()).split()
    if not all_words:
        logger.warning("No words found. Top words plot skipped.")
        return
        
    freq = Counter(all_words).most_common(top_n)
    words_top, counts_top = zip(*freq)
    
    fig, ax = plt.subplots(figsize=(13, 7))
    palette = sns.color_palette("mako_r", len(words_top))
    
    bars = ax.barh(
        list(words_top)[::-1],
        list(counts_top)[::-1],
        color=palette,
    )
    
    ax.set_xlabel("Frekuensi Kemunculan", fontsize=11)
    ax.set_title(
        f"Top-{top_n} Kata Paling Sering Muncul (setelah preprocessing)",
        fontsize=13, fontweight="bold",
    )
    
    for bar, count in zip(bars, list(counts_top)[::-1]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center", fontsize=9,
        )
        
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "top_words.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved top words plot to {plot_path}")


def plot_sentiment_trends(df: pd.DataFrame, output_dir: str) -> None:
    """Extracts date components and saves monthly and daily/hourly trends."""
    logger.info("Generating sentiment trend plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if created_at column is present
    if "created_at" not in df.columns:
        logger.warning("Column 'created_at' not found. Trend plots skipped.")
        return
        
    # Standardize time
    df_time = df.copy()
    df_time["created_at"] = pd.to_datetime(df_time["created_at"], errors="coerce", utc=True)
    df_time = df_time.dropna(subset=["created_at"])
    
    if df_time.empty:
        logger.warning("No valid timestamps in 'created_at' column. Trend plots skipped.")
        return
        
    # 1. Monthly trend
    df_time["bulan"] = df_time["created_at"].dt.to_period("M").astype(str)
    pivot_bulan = (
        df_time.groupby(["bulan", "label"])
        .size().unstack(fill_value=0)
        .sort_index()
    )
    
    fig, ax = plt.subplots(figsize=(15, 5))
    for kelas in ["Positif", "Netral", "Negatif"]:
        if kelas in pivot_bulan.columns:
            ax.plot(
                pivot_bulan.index.astype(str),
                pivot_bulan[kelas],
                marker="o", markersize=5, linewidth=2,
                label=kelas, color=SENTIMENT_COLORS[kelas],
            )
            
    ax.set_title("Tren Sentimen Tweet LPDP per Bulan", fontsize=13, fontweight="bold")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Jumlah Tweet")
    ax.legend(title="Sentimen")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    monthly_path = os.path.join(output_dir, "trend_monthly.png")
    plt.savefig(monthly_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved monthly trend plot to {monthly_path}")
    
    # 2. Day-of-week and Hour-of-day trends
    df_time["hari"] = df_time["created_at"].dt.day_name()
    df_time["jam"]  = df_time["created_at"].dt.hour
    
    hari_urut = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Tren Sentimen Berdasarkan Hari & Jam", fontsize=13, fontweight="bold")
    
    # Day trend
    pivot_hari = (
        df_time.groupby(["hari", "label"])
        .size().unstack(fill_value=0)
        .reindex([h for h in hari_urut if h in df_time["hari"].unique()])
    )
    for kelas in ["Positif", "Netral", "Negatif"]:
        if kelas in pivot_hari.columns:
            axes[0].plot(
                pivot_hari.index.astype(str), pivot_hari[kelas],
                marker="o", markersize=5, linewidth=2,
                label=kelas, color=SENTIMENT_COLORS[kelas],
            )
    axes[0].set_title("Per Hari dalam Seminggu")
    axes[0].set_xlabel("Hari")
    axes[0].set_ylabel("Jumlah Tweet")
    axes[0].legend(title="Sentimen")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    
    # Hour trend
    pivot_jam = (
        df_time.groupby(["jam", "label"])
        .size().unstack(fill_value=0)
        .sort_index()
    )
    for kelas in ["Positif", "Netral", "Negatif"]:
        if kelas in pivot_jam.columns:
            axes[1].plot(
                pivot_jam.index.astype(str), pivot_jam[kelas],
                marker="o", markersize=5, linewidth=2,
                label=kelas, color=SENTIMENT_COLORS[kelas],
            )
    axes[1].set_title("Per Jam dalam Sehari")
    axes[1].set_xlabel("Jam")
    axes[1].set_ylabel("Jumlah Tweet")
    axes[1].legend(title="Sentimen")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    day_hour_path = os.path.join(output_dir, "trend_day_hour.png")
    plt.savefig(day_hour_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved day/hour trend plot to {day_hour_path}")


def run_all_eda(df: pd.DataFrame, output_dir: str) -> None:
    """Executes the full suite of EDA visualizations."""
    logger.info(f"Running exploratory data analysis and saving plots to {output_dir}...")
    plot_sentiment_distribution(df, output_dir)
    generate_wordclouds(df, output_dir)
    plot_top_words(df, output_dir)
    plot_sentiment_trends(df, output_dir)
    logger.info("EDA completed successfully.")
