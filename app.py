
import streamlit as st
import pandas as pd

from src.recommend import recommend, known_users, movie_ids

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #090b18, #15112b, #101b30);
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background: #111426;
}

.hero {
    padding: 35px;
    border-radius: 22px;
    background: linear-gradient(120deg, #5425a8, #176b9b);
    margin-bottom: 25px;
    text-align: center;
}

.hero h1 {
    font-size: 48px;
    color: white;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    color: #e5e7ff;
}

.stat {
    background: #1c2038;
    border: 1px solid #34395c;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    min-height: 115px;
}

.stat h2 {
    color: #bba4ff;
    margin: 5px;
}

.stat p {
    color: #c5c9df;
    margin: 0;
}

.movie-card {
    background: #1b2036;
    border: 1px solid #34395c;
    border-radius: 17px;
    padding: 20px;
    margin-bottom: 15px;
}

.movie-title {
    font-size: 20px;
    font-weight: bold;
    color: #ffffff;
}

.movie-genres {
    color: #b6b9d0;
    font-size: 13px;
}

.section-title {
    color: #d8ccff;
    font-size: 25px;
    font-weight: bold;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HELPERS ----------------
def movie_emoji(genres):
    genres = str(genres).lower()

    if "comedy" in genres:
        return "😂"
    if "horror" in genres:
        return "👻"
    if "romance" in genres:
        return "❤️"
    if "action" in genres:
        return "⚔️"
    if "animation" in genres:
        return "🧸"
    if "sci-fi" in genres:
        return "🚀"
    if "adventure" in genres:
        return "🌍"
    if "drama" in genres:
        return "🎭"

    return "🎬"


def display_movie_card(row, position):
    title = str(row["Movie"])
    genres = str(row["Genres"])
    score = float(row["Ranking Score"])
    score = max(0.0, min(score, 1.0))

    st.markdown(f"""
    <div class="movie-card">
        <div style="font-size:42px">{movie_emoji(genres)}</div>
        <div class="movie-title">{position}. {title}</div>
        <div class="movie-genres">{genres}</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"✨ AI RANKING SCORE: {score:.4f}")
    st.progress(score)


# ---------------- SIDEBAR ----------------
st.sidebar.markdown("# 🎬 CineMatch AI")
st.sidebar.caption("Your intelligent movie discovery companion")

user_options = sorted(list(known_users))

selected_user = st.sidebar.selectbox(
    "👤 Select User ID",
    user_options
)

top_n = st.sidebar.slider(
    "🎯 Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 AI Technology")
st.sidebar.write("• TensorFlow Recommenders")
st.sidebar.write("• Two-Tower Retrieval")
st.sidebar.write("• FAISS Similarity Search")
st.sidebar.write("• Deep Neural Ranking")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python, TensorFlow & Streamlit")


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch AI</h1>
    <p>Discover movies you'll love, powered by personalized AI recommendations.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- STATS ----------------
col1, col2, col3, col4 = st.columns(4)

stats = [
    (col1, "🧠", "AI Models", "2"),
    (col2, "🎞️", "Indexed Movies", f"{len(movie_ids):,}"),
    (col3, "👥", "Known Users", f"{len(known_users):,}"),
    (col4, "✨", "Recommendation", "Personalized")
]

for column, emoji, label, value in stats:
    with column:
        st.markdown(f"""
        <div class="stat">
            <div style="font-size:25px">{emoji}</div>
            <h2>{value}</h2>
            <p>{label}</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- RECOMMENDATIONS ----------------
st.markdown(
    '<div class="section-title">✨ Your Personalized Recommendations</div>',
    unsafe_allow_html=True
)

st.write(
    f"Discover your next favorite movies based on user ID **{selected_user}**."
)

if st.button("🚀 Generate My Recommendations", use_container_width=True):

    with st.spinner("Our AI is finding movies you'll love..."):
        try:
            results = recommend(int(selected_user), top_n=top_n)
            st.session_state["recommendations"] = results
            st.session_state["recommendation_user"] = selected_user
        except Exception as error:
            st.error(f"Recommendation failed: {error}")


# ---------------- DISPLAY RESULTS ----------------
if "recommendations" in st.session_state:

    results = st.session_state["recommendations"]

    if results is not None and not results.empty:

        st.success(
            f"Successfully generated {len(results)} personalized recommendations!"
        )

        genre_options = sorted({
            genre.strip()
            for genres in results["Genres"].dropna()
            for genre in str(genres).split("|")
            if genre.strip()
        })

        selected_genres = st.multiselect(
            "🎭 Filter recommendations by genre",
            options=genre_options
        )

        if selected_genres:
            filtered = results[
                results["Genres"].apply(
                    lambda genres: any(
                        genre in str(genres).split("|")
                        for genre in selected_genres
                    )
                )
            ]
        else:
            filtered = results

        st.markdown("### 🍿 Movies selected for you")

        left, right = st.columns(2)

        for index, (_, row) in enumerate(filtered.iterrows()):
            with (left if index % 2 == 0 else right):
                display_movie_card(row, index + 1)

        if filtered.empty:
            st.info("No movies match those genres. Try another filter.")

    else:
        st.warning("No recommendations were returned. Try another user.")


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#9298b8; padding:15px">
    <h3>💜 CineMatch AI</h3>
    <p>Two-stage personalized recommendation system</p>
    <p>Built using TensorFlow • FAISS • Deep Learning • Streamlit</p>
    <small>Educational recommendation prototype. Ranking scores are not calibrated probabilities.</small>
</div>
""", unsafe_allow_html=True)
