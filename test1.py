import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.title("ML Service Recommendation System")
st.write(
    "This app uses **Unsupervised Machine Learning** (One-Hot Encoding + Cosine Similarity) "
    "to recommend services based on your preferences.")

df = pd.read_csv("service_recommendation_data (1).csv")

feature_cols = ["Target_Business_Type","Price_Category","Language_Support","Location_Area"]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_services = encoder.fit_transform(df[feature_cols])

st.sidebar.header("Your Preferences")
business_type = st.sidebar.selectbox("Target Business Type",sorted(df["Target_Business_Type"].unique()))
price_category = st.sidebar.selectbox("Price Category",sorted(df["Price_Category"].unique()))
language_support = st.sidebar.selectbox("Language Support",sorted(df["Language_Support"].unique()))
location_area = st.sidebar.selectbox("Location / Area",sorted(df["Location_Area"].unique()))
top_n = st.sidebar.slider("Number of recommendations", 1, 10, 3)

if st.sidebar.button("Recommend Services"):
    user_df = pd.DataFrame({
        "Target_Business_Type": [business_type],
        "Price_Category": [price_category],
        "Language_Support": [language_support],
        "Location_Area": [location_area],})

    user_vec = encoder.transform(user_df[feature_cols])
    similarity_scores = cosine_similarity(user_vec, encoded_services)[0]

    df_result = df.copy()
    df_result["Similarity_Score"] = similarity_scores

    df_result = df_result.sort_values(by="Similarity_Score", ascending=False).head(top_n)

    st.subheader("Top Recommended Services")

    show_cols = [
        "Service_ID",
        "Service_Name",
        "Target_Business_Type",
        "Price_Category",
        "Language_Support",
        "Location_Area",
        "Similarity_Score",
        "Description",
    ]

    st.dataframe(
        df_result[show_cols].style.format({"Similarity_Score": "{:.2f}"}),
        use_container_width=True
    )

else:
    st.info("Choose options in the sidebar and press **Recommend Services**.")

