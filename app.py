import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI CSV Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# LOAD CSS
# ---------------------------------------------------

def load_css():

    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

    with open("assets/animation.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.markdown(
    '<div class="sidebar-animate">',
    unsafe_allow_html=True
)

st.sidebar.title("⚡ Navigation")

menu = st.sidebar.radio(
    "Menu",
    [
        "Dashboard",
        "Search & Filter",
        "Data Cleaning",
        "Visualization",
        "Prediction",
        "AI Insights"
    ]
)

st.sidebar.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    """
    <div class="fade-in">
        <div class="title">
            📊 AI CSV Analyzer Dashboard
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload CSV Files",
    type=["csv"],
    accept_multiple_files=True
)

# ---------------------------------------------------
# PROCESS FILES
# ---------------------------------------------------

if uploaded_files:

    all_data = []

    for file in uploaded_files:

        df = pd.read_csv(file)

        df["Source File"] = file.name

        all_data.append(df)

    combined_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # ---------------------------------------------------
    # DASHBOARD
    # ---------------------------------------------------

    if menu == "Dashboard":

        st.markdown(
            '<div class="slide-up">',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Rows",
            combined_df.shape[0]
        )

        col2.metric(
            "Columns",
            combined_df.shape[1]
        )

        col3.metric(
            "Files",
            len(uploaded_files)
        )

        col4.metric(
            "Missing Values",
            combined_df.isnull().sum().sum()
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("👀 Dataset Preview")

        st.dataframe(
            combined_df.head(),
            use_container_width=True
        )

    # ---------------------------------------------------
    # SEARCH & FILTER
    # ---------------------------------------------------

    elif menu == "Search & Filter":

        st.subheader("🔍 Search & Filter Data")

        filtered_df = combined_df.copy()

        # SEARCH
        search = st.text_input(
            "Search Anything"
        )

        if search:

            filtered_df = filtered_df[
                filtered_df.astype(str)
                .apply(
                    lambda x: x.str.contains(
                        search,
                        case=False
                    )
                )
                .any(axis=1)
            ]

        # COLUMN FILTER
        st.subheader("🎯 Select Columns")

        selected_columns = st.multiselect(
            "Choose Columns",
            combined_df.columns,
            default=combined_df.columns
        )

        filtered_df = filtered_df[
            selected_columns
        ]

        # ROW FILTER
        st.subheader("📌 Filter By Value")

        filter_column = st.selectbox(
            "Select Column",
            combined_df.columns
        )

        unique_values = combined_df[
            filter_column
        ].astype(str).unique()

        selected_value = st.selectbox(
            "Select Value",
            unique_values
        )

        filtered_df = filtered_df[
            combined_df[filter_column]
            .astype(str)
            == selected_value
        ]

        # SHOW DATA
        st.dataframe(
            filtered_df,
            use_container_width=True
        )

        # DOWNLOAD CSV
        csv = filtered_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Filtered CSV",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )

    # ---------------------------------------------------
    # DATA CLEANING
    # ---------------------------------------------------

    elif menu == "Data Cleaning":

        st.subheader("🧹 Data Cleaning Tools")

        cleaned_df = combined_df.copy()

        col1, col2, col3 = st.columns(3)

        # REMOVE DUPLICATES
        with col1:

            if st.button("Remove Duplicates"):

                cleaned_df = cleaned_df.drop_duplicates()

                st.success(
                    "Duplicates Removed ✅"
                )

        # FILL MISSING
        with col2:

            if st.button("Fill Missing Values"):

                cleaned_df = cleaned_df.fillna("N/A")

                st.success(
                    "Missing Values Filled ✅"
                )

        # DROP MISSING
        with col3:

            if st.button("Drop Missing Rows"):

                cleaned_df = cleaned_df.dropna()

                st.success(
                    "Missing Rows Removed ✅"
                )

        st.subheader("📄 Cleaned Dataset")

        st.dataframe(
            cleaned_df,
            use_container_width=True
        )

        # DOWNLOAD CLEANED CSV
        cleaned_csv = cleaned_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Cleaned CSV",
            data=cleaned_csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    # ---------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------

    elif menu == "Visualization":

        st.subheader("📊 Charts & Graphs")

        numeric_columns = combined_df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_columns) > 0:

            chart_column = st.selectbox(
                "Select Numeric Column",
                numeric_columns
            )

            # HISTOGRAM
            fig = px.histogram(
                combined_df,
                x=chart_column,
                title=f"{chart_column} Distribution",
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # PIE CHART
            pie_column = st.selectbox(
                "Select Pie Chart Column",
                combined_df.columns
            )

            pie_data = combined_df[
                pie_column
            ].value_counts().reset_index()

            pie_data.columns = [
                pie_column,
                "Count"
            ]

            fig2 = px.pie(
                pie_data,
                names=pie_column,
                values="Count",
                title=f"{pie_column} Distribution",
                template="plotly_dark"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

    # ---------------------------------------------------
    # MACHINE LEARNING PREDICTION
    # ---------------------------------------------------

    elif menu == "Prediction":

        st.subheader(
            "📈 Machine Learning Prediction"
        )

        numeric_columns = combined_df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_columns) >= 2:

            x_column = st.selectbox(
                "Select X Column",
                numeric_columns
            )

            y_column = st.selectbox(
                "Select Y Column",
                numeric_columns,
                index=1
            )

            X = combined_df[[x_column]]
            y = combined_df[y_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            model = LinearRegression()

            model.fit(
                X_train,
                y_train
            )

            st.success(
                "Model Trained Successfully ✅"
            )

            input_value = st.number_input(
                f"Enter {x_column}"
            )

            prediction = model.predict(
                [[input_value]]
            )

            st.info(
                f"Predicted {y_column}: {prediction[0]:.2f}"
            )

    # ---------------------------------------------------
    # AI INSIGHTS
    # ---------------------------------------------------

    elif menu == "AI Insights":

        st.subheader("🤖 Smart AI Insights")

        st.info(f"""

        📌 Total Rows:
        {combined_df.shape[0]}

        📌 Total Columns:
        {combined_df.shape[1]}

        📌 Duplicate Rows:
        {combined_df.duplicated().sum()}

        📌 Missing Values:
        {combined_df.isnull().sum().sum()}

        📌 Numeric Columns:
        {len(combined_df.select_dtypes(include='number').columns)}

        """)

        st.subheader("📈 Statistical Summary")

        st.dataframe(
            combined_df.describe(),
            use_container_width=True
        )

else:

    st.warning(
        "Please Upload CSV Files"
    )