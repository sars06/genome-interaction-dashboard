import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Genome Interaction Strength Predictor",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_FILE = ARTIFACTS_DIR / "dashboard_models.joblib"
METRICS_FILE = ARTIFACTS_DIR / "model_metrics.csv"
PROCESSED_DATA_FILE = ARTIFACTS_DIR / "processed_data.csv"


# ============================================================
# FIND ORIGINAL DATASET
# ============================================================

def find_dataset():

    possible_files = [
        BASE_DIR / "Molm_cleaning_work.xlsx.xlsx",
        BASE_DIR / "Molm_cleaning_work.xlsx",
        BASE_DIR / "Molm_cleaning_work.xls",
        BASE_DIR / "Molm_cleaning_work.csv"
    ]

    for file in possible_files:
        if file.exists():
            return file

    # Search for Excel files if exact filename isn't found
    for file in BASE_DIR.glob("*.xlsx"):
        return file

    for file in BASE_DIR.glob("*.xls"):
        return file

    for file in BASE_DIR.glob("*.csv"):
        return file

    return None


DATA_FILE = find_dataset()


# ============================================================
# LOAD MODEL PACKAGE
# ============================================================

@st.cache_resource
def load_models():

    if not MODEL_FILE.exists():
        return None, (
            f"Model file was not found:\n\n"
            f"{MODEL_FILE}"
        )

    try:

        package = joblib.load(MODEL_FILE)

        required_keys = [
            "random_forest",
            "extra_trees",
            "tuned_random_forest",
            "preprocessor"
        ]

        missing = [
            key for key in required_keys
            if key not in package
        ]

        if missing:
            return None, (
                "The dashboard model package is missing:\n"
                + ", ".join(missing)
            )

        return package, None

    except Exception as e:

        return None, str(e)


package, model_error = load_models()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset(file_path):

    if file_path is None:
        return None, "Dataset file was not found."

    try:

        if str(file_path).lower().endswith(".csv"):
            df = pd.read_csv(file_path)

        else:
            df = pd.read_excel(file_path)

        return df, None

    except Exception as e:

        return None, str(e)


# ============================================================
# LOAD MODEL METRICS
# ============================================================

@st.cache_data
def load_metrics():

    if not METRICS_FILE.exists():
        return None, "model_metrics.csv was not found."

    try:

        if METRICS_FILE.stat().st_size == 0:
            return None, "model_metrics.csv is empty."

        metrics = pd.read_csv(METRICS_FILE)

        if metrics.empty:
            return None, "model_metrics.csv contains no rows."

        return metrics, None

    except Exception as e:

        return None, str(e)


# ============================================================
# DATASET SELECTION
# ============================================================

# Always use the original project dataset.
# No file-upload option is provided in the dashboard.
df = None
data_error = None

if DATA_FILE is not None:
    df, data_error = load_dataset(DATA_FILE)
else:
    data_error = "Original dataset was not found."


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):

    if df is None:
        return None

    columns = list(df.columns)

    # Exact match
    for candidate in candidates:

        for col in columns:

            if str(col).lower() == candidate.lower():
                return col

    # Partial match
    for candidate in candidates:

        candidate_lower = candidate.lower()

        for col in columns:

            if candidate_lower in str(col).lower():
                return col

    return None


def calculate_features(input_df):

    result = input_df.copy()

    # --------------------------------------------------------
    # Interactor width
    # --------------------------------------------------------

    start_col = find_column(
        result,
        [
            "Interactor_Start",
            "Interactor start",
            "InteractorStart"
        ]
    )

    end_col = find_column(
        result,
        [
            "Interactor_End",
            "Interactor end",
            "InteractorEnd"
        ]
    )

    width_col = find_column(
        result,
        [
            "Interactor_Width",
            "Interactor Width",
            "InteractorWidth"
        ]
    )

    if width_col is not None and start_col is not None and end_col is not None:

        result[width_col] = (
            pd.to_numeric(result[end_col], errors="coerce")
            -
            pd.to_numeric(result[start_col], errors="coerce")
        ).abs()

    # --------------------------------------------------------
    # Genomic distance
    # --------------------------------------------------------

    feature_start_col = find_column(
        result,
        [
            "Feature_Start",
            "Feature start",
            "FeatureStart"
        ]
    )

    interactor_start_col = find_column(
        result,
        [
            "Interactor_Start",
            "Interactor start",
            "InteractorStart"
        ]
    )

    distance_col = find_column(
        result,
        [
            "Genomic_Distance",
            "Genomic Distance",
            "GenomicDistance"
        ]
    )

    if (
        distance_col is not None
        and feature_start_col is not None
        and interactor_start_col is not None
    ):

        feature_start = pd.to_numeric(
            result[feature_start_col],
            errors="coerce"
        )

        interactor_start = pd.to_numeric(
            result[interactor_start_col],
            errors="coerce"
        )

        result[distance_col] = (
            feature_start - interactor_start
        ).abs()

    # --------------------------------------------------------
    # Log genomic distance
    # --------------------------------------------------------

    log_distance_col = find_column(
        result,
        [
            "Log_Genomic_Distance",
            "Log Genomic Distance",
            "LogGenomicDistance"
        ]
    )

    if (
        log_distance_col is not None
        and distance_col is not None
    ):

        distance = pd.to_numeric(
            result[distance_col],
            errors="coerce"
        )

        result[log_distance_col] = np.log1p(
            distance.clip(lower=0)
        )

    return result


def detect_strength_columns(df):
    """
    Detect condition-specific interaction-strength columns.

    The MOLM-1 dataset stores interaction strength as two replicate
    support-pair columns for each condition, rather than columns named
    Normal_Strength / Gemcitabine_Strength / Carboplatin_Strength.

    Normal      -> MN1_SuppPairs + MN2_SuppPairs
    Carboplatin -> MC1_SuppPairs + MC2_SuppPairs
    Gemcitabine -> MG1_SuppPairs + MG2_SuppPairs
    """

    if df is None:
        return {}

    columns = list(df.columns)

    # Exact dataset structure used by the project notebook.
    condition_specs = {
        "Normal": ["MN1_SuppPairs", "MN2_SuppPairs"],
        "Carboplatin": ["MC1_SuppPairs", "MC2_SuppPairs"],
        "Gemcitabine": ["MG1_SuppPairs", "MG2_SuppPairs"],
    }

    result = {}

    for condition, candidates in condition_specs.items():
        found = []

        for candidate in candidates:
            for col in columns:
                if str(col).strip().lower() == candidate.lower():
                    found.append(col)
                    break

        if len(found) == 2:
            result[condition] = found

    # Flexible fallback for minor column-name variations.
    if len(result) < 3:
        prefixes = {
            "Normal": ("mn1", "mn2"),
            "Carboplatin": ("mc1", "mc2"),
            "Gemcitabine": ("mg1", "mg2"),
        }

        for condition, reps in prefixes.items():
            if condition in result:
                continue

            found = []
            for prefix in reps:
                matches = [
                    col for col in columns
                    if str(col).lower().replace(" ", "").replace("-", "_").startswith(prefix)
                    and "supp" in str(col).lower()
                ]
                if matches:
                    found.append(matches[0])

            if len(found) == 2:
                result[condition] = found

    return result


def calculate_condition_strengths(df, strength_columns):
    """Create one numeric strength value per condition as the replicate mean."""

    result = df.copy()

    for condition, columns in strength_columns.items():
        numeric = result[columns].apply(pd.to_numeric, errors="coerce")
        result[f"{condition} Strength"] = numeric.mean(axis=1)

    return result


def detect_treatment_column(df):

    return find_column(
        df,
        [
            "Treatment",
            "Condition",
            "Drug",
            "Treatment_Group",
            "Condition_Name"
        ]
    )


def detect_strength_column(df):

    return find_column(
        df,
        [
            "Interaction_Strength",
            "Interaction Strength",
            "Strength",
            "interaction_strength",
            "strength"
        ]
    )


def detect_distance_column(df):

    return find_column(
        df,
        [
            "Genomic_Distance",
            "Genomic Distance",
            "GenomicDistance",
            "Distance",
            "distance"
        ]
    )


def safe_numeric(value, default=0.0):

    try:
        return float(value)

    except Exception:
        return default


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard")

st.sidebar.subheader("Select Analysis")

page = st.sidebar.radio(
    "",
    [
        "Project Introduction",
        "Dataset Analysis",
        "Model Comparison",
        "New Interaction Prediction"
    ],
    index=0
)


st.sidebar.divider()

st.sidebar.subheader("Display Settings")

show_data = st.sidebar.checkbox(
    "Show sample data",
    value=False
)


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🧬 Genome Interaction Strength Predictor")

st.write(
    "Analyze the relationship between genomic distance and "
    "interaction strength, compare machine-learning models, "
    "and predict interaction strength for new genomic interactions."
)

st.divider()


# ============================================================
# PROJECT INTRODUCTION
# ============================================================

if page == "Project Introduction":

    st.header("📖 A–Z Project Introduction")
    st.write(
        "A beginner-friendly guide to understanding what this project studies, "
        "why it matters, what the biological terms mean, and how the machine-learning "
        "dashboard turns genomic interaction data into analysis and predictions."
    )

    st.divider()

    st.subheader("🧬 1. What is the project about?")
    st.markdown(
        """
This project studies **genome interactions** — relationships between two regions of DNA
inside a cell. Instead of looking only at where a DNA region is located, we ask whether
two regions appear to interact and how **strongly** that interaction is represented in the data.

The dashboard focuses on the relationship between **genomic distance** and **interaction strength**
under three experimental conditions:

- 🟢 **Normal** — the reference condition
- 🔵 **Gemcitabine** — a drug-treatment condition
- 🟣 **Carboplatin** — another drug-treatment condition

The project has two main goals: **understand patterns in the existing dataset** and **predict
interaction strength for a new genomic interaction** using trained machine-learning models.
"""
    )

    st.subheader("🔬 2. Why do genome interactions matter?")
    st.markdown(
        """
DNA is extremely long, but it is packed into the nucleus in a three-dimensional structure.
Regions that are far apart along the DNA sequence can still come close to each other in
three-dimensional space. These physical or regulatory contacts can be related to how genes
and other genomic regions are controlled.

Studying these interactions can therefore help us understand how genomic organization changes
under different experimental conditions. In this project, the treatment groups provide a way
to compare interaction patterns between a reference state and two drug-treatment states.

**Important:** this dashboard is a computational analysis tool. A predicted interaction strength
is an estimate from the trained model; it is not, by itself, proof that a biological interaction
occurs in a cell.
"""
    )

    st.subheader("📚 3. Biological terms in simple language")

    terms = {
        "Genome": "The complete set of DNA information in a cell or organism.",
        "DNA": "The molecule that stores genetic information. It can be thought of as a very long sequence containing biological instructions.",
        "Genomic region": "A particular section of DNA identified by its chromosome and genomic coordinates.",
        "Chromosome": "A packaged DNA molecule. Coordinates such as chr10 identify a chromosome and positions along it.",
        "Feature": "The genomic region being used as one side of an interaction in this dataset.",
        "Interactor": "The genomic region paired with the feature to represent a possible interaction.",
        "Genomic coordinates": "Numerical positions that specify where a genomic region starts and ends on a chromosome.",
        "Genomic distance": "The absolute difference between the relevant genomic positions used by this project. A larger value means the two positions are farther apart along the DNA sequence.",
        "Interaction": "A relationship/contact between two genomic regions represented by the experimental data.",
        "Interaction strength": "A numerical measure used in this project to represent how strongly an interaction is supported in the dataset.",
        "Support pairs (SuppPairs)": "The support-pair measurements used as the interaction-strength signal in the dataset. The dashboard averages the two replicate measurements for each condition.",
        "Replicate": "A repeated measurement for the same experimental condition. Replicates help provide a more robust summary than relying on one measurement alone.",
        "Normal condition": "The reference experimental condition represented by the MN1 and MN2 support-pair measurements.",
        "Gemcitabine condition": "The treatment condition represented by the MG1 and MG2 support-pair measurements.",
        "Carboplatin condition": "The treatment condition represented by the MC1 and MC2 support-pair measurements.",
        "Log genomic distance": "A logarithmic transformation of genomic distance, calculated as log(1 + distance), which can make a highly skewed distance distribution easier for a model to work with.",
        "MOLM-1": "The project dashboard identifies the dataset/model context as MOLM-1. In this dashboard, it refers to the biological dataset being analyzed rather than to a prediction result itself.",
    }

    for term, explanation in terms.items():
        with st.expander(term):
            st.write(explanation)

    st.subheader("⚙️ 4. How does the analysis work?")
    st.markdown(
        """
The analysis follows a simple pipeline:

**Step 1 — Start with the genomic interaction dataset**  
Each row represents a genomic interaction record with genomic coordinates and other measured
features. The dashboard reads the original project dataset automatically; there is no upload
box on the dashboard.

**Step 2 — Calculate genomic distance**  
The dashboard uses the relevant feature and interactor positions to calculate the genomic
distance used for the analysis. It also calculates the log-transformed distance when that
feature is required.

**Step 3 — Identify the three conditions**  
The dataset stores condition-specific interaction support in replicate columns:

- **Normal:** MN1_SuppPairs + MN2_SuppPairs
- **Gemcitabine:** MG1_SuppPairs + MG2_SuppPairs
- **Carboplatin:** MC1_SuppPairs + MC2_SuppPairs

For each row, the dashboard takes the mean of the two replicate values to obtain one
interaction-strength value for each condition.

**Step 4 — Explore the relationship**  
The Dataset Analysis page plots interaction strength against genomic distance and calculates
a correlation value for each condition. This helps us see whether interaction strength tends
to increase, decrease, or show a weak relationship as genomic distance changes.

**Step 5 — Train machine-learning models**  
The project uses regression models to learn patterns between the available genomic features
and interaction strength. The dashboard can compare:

- Random Forest
- Extra Trees
- Tuned Random Forest

**Step 6 — Predict a new interaction**  
When a user enters genomic coordinates and interaction characteristics, the same preprocessing
pipeline used during training prepares the new row. The trained models then estimate its
interaction strength.
"""
    )

    st.subheader("🤖 5. Why use machine learning?")
    st.markdown(
        """
The dataset contains many genomic features, and the relationship between those features and
interaction strength may not be a simple straight-line relationship. Tree-based ensemble models
can learn non-linear patterns and interactions among multiple features.

**Random Forest** builds many decision trees and combines their predictions.

**Extra Trees** is another tree-ensemble method that introduces additional randomness when
creating trees, which can produce a different and sometimes more generalizable model.

**Tuned Random Forest** is a Random Forest whose important hyperparameters have been adjusted
using the project's training process to improve test-set performance.

The Model Comparison page reports **MAE, RMSE, and R²** when those metrics are available:

- **MAE:** average absolute prediction error; lower is better.
- **RMSE:** error measure that penalizes large errors more strongly; lower is better.
- **R²:** indicates how much variation in the target is explained by the model; higher is generally better.
"""
    )

    st.subheader("🔮 6. How should I understand a prediction?")
    st.markdown(
        """
A prediction is the model's estimated interaction-strength value for the genomic information
entered by the user. The dashboard displays predictions from all three trained models so that
their outputs can be compared. It also shows their average as a simple overall estimate.

A prediction should be interpreted together with the model's validation performance. A model
with better test-set metrics is generally more trustworthy than one with poorer performance,
but even a good model can make individual predictions that are inaccurate.
"""
    )

    st.subheader("🧭 7. How to use this dashboard")
    st.markdown(
        """
**Start Here → Project Introduction**  
Learn the biological and machine-learning terminology.

**Dataset Analysis**  
Explore the relationship between genomic distance and interaction strength for Normal,
Gemcitabine, and Carboplatin.

**Model Comparison**  
Compare the trained models using their test-set performance.

**New Interaction Prediction**  
Enter the coordinates and interaction characteristics of a new row and obtain predicted
interaction-strength values from the trained models.
"""
    )

    st.subheader("🧠 8. The project in one sentence")
    st.success(
        "This project uses genomic interaction data to study how interaction strength relates "
        "to genomic distance across Normal, Gemcitabine, and Carboplatin conditions, and then "
        "uses machine-learning regression models to estimate interaction strength for new genomic interactions."
    )

    st.info(
        "Tip: If you are new to genomics, read the expandable biological terms above first, "
        "then open Dataset Analysis and Model Comparison. You can use New Interaction Prediction last."
    )


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "Dataset Analysis":

    st.header("📊 Interaction Strength Analysis")

    st.write(
        "This section analyzes how interaction strength "
        "changes with genomic distance for Normal, "
        "Gemcitabine, and Carboplatin conditions."
    )

    if df is None:

        st.error(
            "Could not load the dataset."
        )

        if data_error:
            st.code(data_error)

    else:

        # ----------------------------------------------------
        # Dataset overview
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Rows",
                f"{len(df):,}"
            )

        with col2:

            st.metric(
                "Features",
                len(df.columns)
            )

        distance_col = detect_distance_column(df)

        with col3:

            st.metric(
                "Distance Available",
                "Yes" if distance_col else "No"
            )

        treatment_col = detect_treatment_column(df)
        strength_columns_for_card = detect_strength_columns(df)
        treatment_available = (
            treatment_col is not None
            or len(strength_columns_for_card) > 0
        )

        with col4:

            st.metric(
                "Treatment Available",
                "Yes" if treatment_available else "No"
            )

        st.divider()

        # ----------------------------------------------------
        # Show actual columns
        # ----------------------------------------------------

        with st.expander("View dataset columns"):

            st.write(list(df.columns))

        # ----------------------------------------------------
        # Detect condition-specific strength columns
        # ----------------------------------------------------

        strength_columns = detect_strength_columns(df)

        if distance_col is not None and len(strength_columns) > 0:

            st.subheader("Interaction Strength vs Genomic Distance")

            st.success(
                "Treatment conditions detected from replicate support-pair columns."
            )

            st.write(
                "**Strength calculation:** each condition's interaction strength "
                "is the mean of its two replicate support-pair measurements."
            )

            detection_rows = []
            for condition, columns in strength_columns.items():
                detection_rows.append({
                    "Condition": condition,
                    "Replicate 1": columns[0],
                    "Replicate 2": columns[1],
                    "Strength": f"Mean({columns[0]}, {columns[1]})"
                })

            st.dataframe(
                pd.DataFrame(detection_rows),
                use_container_width=True,
                hide_index=True
            )

            analysis_df = calculate_condition_strengths(
                df, strength_columns
            )

            distance_values = pd.to_numeric(
                analysis_df[distance_col],
                errors="coerce"
            )

            # Build a long-form table for plotting.
            plot_parts = []

            for condition in strength_columns:

                strength_col = f"{condition} Strength"

                temp = pd.DataFrame({
                    "Genomic Distance": distance_values,
                    "Interaction Strength": pd.to_numeric(
                        analysis_df[strength_col],
                        errors="coerce"
                    ),
                    "Condition": condition
                }).dropna()

                if len(temp) > 10000:
                    temp = temp.sample(
                        10000,
                        random_state=42
                    )

                plot_parts.append(temp)

            if plot_parts:

                plot_df = pd.concat(
                    plot_parts,
                    ignore_index=True
                )

                if not plot_df.empty:

                    fig = px.scatter(
                        plot_df,
                        x="Genomic Distance",
                        y="Interaction Strength",
                        color="Condition",
                        opacity=0.45,
                        title=(
                            "Interaction Strength vs "
                            "Genomic Distance"
                        )
                    )

                    fig.update_layout(
                        height=600,
                        xaxis_title="Genomic Distance",
                        yaxis_title="Mean Support Pairs"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    # ------------------------------------------------
                    # Condition summary
                    # ------------------------------------------------

                    st.subheader("Treatment-wise Strength Summary")

                    summary_rows = []

                    for condition in strength_columns:

                        values = pd.to_numeric(
                            analysis_df[f"{condition} Strength"],
                            errors="coerce"
                        ).dropna()

                        summary_rows.append({
                            "Condition": condition,
                            "Rows": len(values),
                            "Mean Strength": values.mean(),
                            "Median Strength": values.median(),
                            "Minimum": values.min(),
                            "Maximum": values.max()
                        })

                    summary_df = pd.DataFrame(summary_rows)

                    st.dataframe(
                        summary_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # ------------------------------------------------
                    # Correlation
                    # ------------------------------------------------

                    st.subheader(
                        "Correlation with Genomic Distance"
                    )

                    correlation_data = []

                    for condition in strength_columns:

                        temp = pd.DataFrame({
                            "distance": pd.to_numeric(
                                analysis_df[distance_col],
                                errors="coerce"
                            ),
                            "strength": pd.to_numeric(
                                analysis_df[f"{condition} Strength"],
                                errors="coerce"
                            )
                        }).dropna()

                        if len(temp) > 1:
                            correlation = temp[
                                "distance"
                            ].corr(
                                temp["strength"]
                            )
                        else:
                            correlation = np.nan

                        correlation_data.append({
                            "Condition": condition,
                            "Correlation": correlation
                        })

                    correlation_df = pd.DataFrame(
                        correlation_data
                    )

                    st.dataframe(
                        correlation_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.info(
                        "Correlation values closer to -1 indicate "
                        "that interaction strength tends to decrease "
                        "as genomic distance increases. Values closer "
                        "to +1 indicate an increasing relationship."
                    )

            else:
                st.warning(
                    "The detected support-pair columns do not contain "
                    "usable numeric values."
                )

        else:

            st.warning(
                "The dashboard could not automatically identify the "
                "required genomic distance and treatment-specific "
                "strength columns."
            )

            st.write(
                "Detected distance column:",
                distance_col
            )

            st.write(
                "Detected treatment column:",
                treatment_col
            )

            st.write(
                "Detected strength columns:",
                strength_columns
            )

            st.info(
                "This dataset represents treatments through replicate "
                "support-pair columns: MN1/MN2, MC1/MC2 and MG1/MG2."
            )

            st.subheader("Dataset Columns")

            st.write(list(df.columns))

        # ----------------------------------------------------
        # Dataset preview
        # ----------------------------------------------------

        if show_data:

            st.divider()

            st.subheader(
                "Dataset Preview"
            )

            st.dataframe(
                df.head(100),
                use_container_width=True
            )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.header("🤖 Model Comparison")

    st.write(
        "Comparison of the three trained regression models "
        "using their test-set performance."
    )

    metrics, metrics_error = load_metrics()

    if metrics is None:

        st.warning(
            "Model metrics are currently unavailable."
        )

        st.info(
            f"Reason: {metrics_error}\n\n"
            "Run the model-training notebook again and "
            "generate artifacts/model_metrics.csv."
        )

        st.code(
            "Model,MAE,RMSE,R2\n"
            "Random Forest,...,...,...\n"
            "Extra Trees,...,...,...\n"
            "Tuned Random Forest,...,...,..."
        )

    else:

        st.subheader("Test-set Performance")

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Metric cards
        # ----------------------------------------------------

        for _, row in metrics.iterrows():

            model_name = row.get(
                "Model",
                "Model"
            )

            st.markdown(
                f"### {model_name}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                if "MAE" in metrics.columns:

                    st.metric(
                        "MAE",
                        f"{row['MAE']:.4f}"
                    )

            with col2:

                if "RMSE" in metrics.columns:

                    st.metric(
                        "RMSE",
                        f"{row['RMSE']:.4f}"
                    )

            with col3:

                if "R2" in metrics.columns:

                    st.metric(
                        "R²",
                        f"{row['R2']:.4f}"
                    )

        # ----------------------------------------------------
        # Model comparison chart
        # ----------------------------------------------------

        if "Model" in metrics.columns:

            available_metrics = [
                x
                for x in ["MAE", "RMSE", "R2"]
                if x in metrics.columns
            ]

            if available_metrics:

                st.subheader(
                    "Model Performance Comparison"
                )

                melted = metrics.melt(
                    id_vars=["Model"],
                    value_vars=available_metrics,
                    var_name="Metric",
                    value_name="Value"
                )

                fig = px.bar(
                    melted,
                    x="Model",
                    y="Value",
                    color="Metric",
                    barmode="group",
                    title="Model Performance"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if "R2" in metrics.columns:

            best_row = metrics.loc[
                metrics["R2"].idxmax()
            ]

            st.success(
                f"Best model based on R²: "
                f"**{best_row['Model']}** "
                f"(R² = {best_row['R2']:.4f})"
            )


# ============================================================
# NEW INTERACTION PREDICTION
# ============================================================

# ============================================================
# NEW INTERACTION PREDICTION
# ============================================================

elif page == "New Interaction Prediction":

    st.header("🔮 New Interaction Prediction")

    st.write(
        "Enter the genomic coordinates and interaction "
        "characteristics to estimate interaction strength."
    )

    if package is None:

        st.error("Prediction models are not available.")

        st.info("Expected model file:")

        st.code(str(MODEL_FILE))

    else:

        st.success(
            "Prediction models loaded successfully."
        )

        preprocessor = package["preprocessor"]

        # ----------------------------------------------------
        # Get feature names used during training
        # ----------------------------------------------------

        if hasattr(preprocessor, "feature_names_in_"):

            expected_columns = list(
                preprocessor.feature_names_in_
            )

        else:

            expected_columns = []

            try:

                for transformer_name, transformer, columns in (
                    preprocessor.transformers_
                ):

                    if transformer_name != "remainder":

                        if isinstance(columns, (list, tuple)):

                            expected_columns.extend(
                                columns
                            )

            except Exception:
                pass

        if not expected_columns:

            st.error(
                "Could not determine the features used "
                "by the trained preprocessor."
            )

            st.stop()

        # ----------------------------------------------------
        # Display expected features
        # ----------------------------------------------------

        with st.expander(
            "View features expected by the trained model"
        ):

            st.write(expected_columns)

            st.write(
                f"Total expected features: "
                f"{len(expected_columns)}"
            )

        # ----------------------------------------------------
        # INPUT FORM
        # ----------------------------------------------------

        st.subheader("Genomic Coordinates")

        col1, col2 = st.columns(2)

        with col1:

            feature_chromosome = st.text_input(
                "Feature Chromosome",
                value="chr10"
            )

            feature_start = st.number_input(
                "Feature Start",
                min_value=0,
                value=27150016,
                step=1
            )

        with col2:

            interactor_chromosome = st.text_input(
                "Interactor Chromosome",
                value="chr10"
            )

            interactor_start = st.number_input(
                "Interactor Start",
                min_value=0,
                value=26619026,
                step=1
            )

            interactor_end = st.number_input(
                "Interactor End",
                min_value=0,
                value=26620708,
                step=1
            )

        st.subheader("Interaction Information")

        col1, col2 = st.columns(2)

        with col1:

            strand = st.text_input(
                "Strand",
                value="+"
            )

        with col2:

            annotation = st.text_input(
                "Annotation",
                value="unknown"
            )

        interaction_type = st.selectbox(
            "Interaction Type",
            [
                "cis",
                "trans"
            ]
        )

        # ----------------------------------------------------
        # DERIVED FEATURES
        # ----------------------------------------------------

        interactor_width = abs(
            interactor_end - interactor_start
        )

        genomic_distance = abs(
            feature_start - interactor_start
        )

        log_genomic_distance = np.log1p(
            genomic_distance
        )

        st.subheader(
            "Automatically Calculated Features"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Interactor Width",
                f"{interactor_width:,}"
            )

        with c2:

            st.metric(
                "Genomic Distance",
                f"{genomic_distance:,}"
            )

        with c3:

            st.metric(
                "Log Genomic Distance",
                f"{log_genomic_distance:.4f}"
            )

        # ----------------------------------------------------
        # CREATE INITIAL INPUT
        # ----------------------------------------------------

        input_data = {}

        for column in expected_columns:

            column_lower = str(column).lower().strip()

            # -----------------------------------------------
            # Feature chromosome
            # -----------------------------------------------

            if column_lower in [
                "feature_chromosome",
                "feature chromosome",
                "feature_chr",
                "featurechromosome"
            ]:

                input_data[column] = feature_chromosome

            # -----------------------------------------------
            # Interactor chromosome
            # -----------------------------------------------

            elif column_lower in [
                "interactor_chromosome",
                "interactor chromosome",
                "interactor_chr",
                "interactorchromosome"
            ]:

                input_data[column] = (
                    interactor_chromosome
                )

            # -----------------------------------------------
            # Feature start
            # -----------------------------------------------

            elif column_lower in [
                "feature_start",
                "feature start",
                "featurestart"
            ]:

                input_data[column] = feature_start

            # -----------------------------------------------
            # Interactor start
            # -----------------------------------------------

            elif column_lower in [
                "interactor_start",
                "interactor start",
                "interactorstart"
            ]:

                input_data[column] = interactor_start

            # -----------------------------------------------
            # Interactor end
            # -----------------------------------------------

            elif column_lower in [
                "interactor_end",
                "interactor end",
                "interactorend"
            ]:

                input_data[column] = interactor_end

            # -----------------------------------------------
            # Interaction type
            # -----------------------------------------------

            elif column_lower in [
                "interaction_type",
                "interaction type",
                "interactiontype",
                "type"
            ]:

                input_data[column] = interaction_type

            # -----------------------------------------------
            # Strand
            # -----------------------------------------------

            elif column_lower == "strand":

                input_data[column] = strand

            # -----------------------------------------------
            # Annotation
            # -----------------------------------------------

            elif column_lower == "annotation":

                input_data[column] = annotation

            # -----------------------------------------------
            # Interactor width
            # -----------------------------------------------

            elif (
                "interactor_width"
                in column_lower
            ):

                input_data[column] = (
                    interactor_width
                )

            elif column_lower in [
                "interactor width",
                "interactorwidth"
            ]:

                input_data[column] = (
                    interactor_width
                )

            # -----------------------------------------------
            # Genomic distance
            # -----------------------------------------------

            elif (
                "log_genomic_distance"
                in column_lower
            ):

                input_data[column] = (
                    log_genomic_distance
                )

            elif (
                "log genomic distance"
                in column_lower
            ):

                input_data[column] = (
                    log_genomic_distance
                )

            elif (
                "genomic_distance"
                in column_lower
            ):

                input_data[column] = (
                    genomic_distance
                )

            elif (
                "genomic distance"
                in column_lower
            ):

                input_data[column] = (
                    genomic_distance
                )

            elif column_lower == "distance":

                input_data[column] = (
                    genomic_distance
                )

            # -----------------------------------------------
            # Existing dataset column
            # -----------------------------------------------

            elif (
                df is not None
                and column in df.columns
            ):

                # Get original column
                original_series = df[column]

                # Try to determine if numeric
                numeric_series = pd.to_numeric(
                    original_series,
                    errors="coerce"
                )

                numeric_ratio = (
                    numeric_series.notna().mean()
                )

                # Mostly numeric
                if numeric_ratio >= 0.8:

                    median_value = numeric_series.median()

                    if pd.isna(median_value):

                        median_value = 0.0

                    input_data[column] = (
                        float(median_value)
                    )

                # Categorical
                else:

                    non_empty = (
                        original_series
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )

                    non_empty = non_empty[
                        non_empty != ""
                    ]

                    if len(non_empty) > 0:

                        input_data[column] = (
                            non_empty.mode().iloc[0]
                        )

                    else:

                        input_data[column] = "unknown"

            # -----------------------------------------------
            # Unknown feature
            # -----------------------------------------------

            else:

                input_data[column] = 0.0

        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [input_data],
            columns=expected_columns
        )

        # ----------------------------------------------------
        # CLEAN INPUT TYPES
        # ----------------------------------------------------

        for column in input_df.columns:

            # If dataset exists, use its type to help
            # determine whether this should be numeric.

            if (
                df is not None
                and column in df.columns
            ):

                original_series = df[column]

                numeric_series = pd.to_numeric(
                    original_series,
                    errors="coerce"
                )

                numeric_ratio = (
                    numeric_series.notna().mean()
                )

                if numeric_ratio >= 0.8:

                    value = pd.to_numeric(
                        input_df[column],
                        errors="coerce"
                    )

                    if value.isna().any():

                        median_value = (
                            numeric_series.median()
                        )

                        if pd.isna(median_value):

                            median_value = 0.0

                        value = value.fillna(
                            median_value
                        )

                    input_df[column] = value.astype(float)

                else:

                    input_df[column] = (
                        input_df[column]
                        .fillna("unknown")
                        .astype(str)
                    )

        # ----------------------------------------------------
        # IMPORTANT:
        # REMOVE EMPTY STRINGS
        # ----------------------------------------------------

        input_df = input_df.replace(
            r"^\s*$",
            np.nan,
            regex=True
        )

        # Fill remaining missing values based on
        # the training dataset.

        if df is not None:

            for column in input_df.columns:

                if column in df.columns:

                    numeric_series = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    )

                    numeric_ratio = (
                        numeric_series.notna().mean()
                    )

                    if numeric_ratio >= 0.8:

                        median_value = (
                            numeric_series.median()
                        )

                        if pd.isna(median_value):

                            median_value = 0.0

                        input_df[column] = (
                            pd.to_numeric(
                                input_df[column],
                                errors="coerce"
                            )
                            .fillna(median_value)
                        )

                    else:

                        input_df[column] = (
                            input_df[column]
                            .fillna("unknown")
                        )

        # ----------------------------------------------------
        # SHOW MODEL INPUT
        # ----------------------------------------------------

        with st.expander(
            "🔍 View data sent to the model"
        ):

            st.dataframe(
                input_df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        st.divider()

        predict_button = st.button(
            "🔮 Predict Interaction Strength",
            type="primary",
            use_container_width=True
        )

        if predict_button:

            try:

                # -------------------------------------------
                # FINAL CHECK FOR EMPTY STRINGS
                # -------------------------------------------

                empty_cells = []

                for column in input_df.columns:

                    value = input_df.iloc[0][column]

                    if (
                        isinstance(value, str)
                        and value.strip() == ""
                    ):

                        empty_cells.append(column)

                if empty_cells:

                    st.error(
                        "The following features contain "
                        "empty values:"
                    )

                    st.write(empty_cells)

                    st.stop()

                # -------------------------------------------
                # TRANSFORM
                # -------------------------------------------

                X_input = preprocessor.transform(
                    input_df
                )

                # -------------------------------------------
                # MODEL PREDICTIONS
                # -------------------------------------------

                rf_prediction = float(
                    np.asarray(
                        package["random_forest"].predict(X_input)
                        ).reshape(-1)[0]
                )

                extra_prediction = float(
                    np.asarray(
                        package["extra_trees"].predict(X_input)
                    ).reshape(-1)[0]
                )

                tuned_prediction = float(
                    np.asarray(
                        package["tuned_random_forest"].predict(X_input)
                    ).reshape(-1)[0]
                )

                # -------------------------------------------
                # RESULTS
                # -------------------------------------------

                st.success(
                    "Prediction completed successfully!"
                )

                st.subheader(
                    "Predicted Interaction Strength"
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Random Forest",
                        f"{rf_prediction:.4f}"
                    )

                with c2:

                    st.metric(
                        "Extra Trees",
                        f"{extra_prediction:.4f}"
                    )

                with c3:

                    st.metric(
                        "Tuned Random Forest",
                        f"{tuned_prediction:.4f}"
                    )

                # -------------------------------------------
                # RESULT TABLE
                # -------------------------------------------

                result_df = pd.DataFrame({

                    "Model": [
                        "Random Forest",
                        "Extra Trees",
                        "Tuned Random Forest"
                    ],

                    "Predicted Interaction Strength": [
                        rf_prediction,
                        extra_prediction,
                        tuned_prediction
                    ]
                })

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )

                # -------------------------------------------
                # CHART
                # -------------------------------------------

                fig = px.bar(
                    result_df,
                    x="Model",
                    y="Predicted Interaction Strength",
                    title=(
                        "Predicted Interaction "
                        "Strength by Model"
                    )
                )

                fig.update_layout(
                    height=500
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # -------------------------------------------
                # AVERAGE PREDICTION
                # -------------------------------------------

                average_prediction = np.mean([
                    rf_prediction,
                    extra_prediction,
                    tuned_prediction
                ])

                st.info(
                    f"Average predicted interaction "
                    f"strength: **{average_prediction:.4f}**"
                )

                # -------------------------------------------
                # INPUT SUMMARY
                # -------------------------------------------

                st.subheader(
                    "Prediction Input Summary"
                )

                summary1, summary2, summary3 = (
                    st.columns(3)
                )

                with summary1:

                    st.write(
                        "**Genomic Distance**"
                    )

                    st.write(
                        f"{genomic_distance:,}"
                    )

                with summary2:

                    st.write(
                        "**Interactor Width**"
                    )

                    st.write(
                        f"{interactor_width:,}"
                    )

                with summary3:

                    st.write(
                        "**Log Genomic Distance**"
                    )

                    st.write(
                        f"{log_genomic_distance:.4f}"
                    )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)

                st.warning(
                    "The error is coming from the trained "
                    "preprocessing pipeline. Check the "
                    "'View data sent to the model' section "
                    "above to identify the problematic feature."
                )