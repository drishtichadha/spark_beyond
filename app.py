import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
import plotly.express as px
import plotly.graph_objects as go
from core.discovery import Problem, SchemaChecks
from core.features.auto_feature_generator import AutoFeatureGenerator
from core.features.process import PreProcessVariables
from core.features.feature_selector import FeatureSelector

# Page configuration
st.set_page_config(
    page_title="Spark Beyond - ML Feature Discovery",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'spark' not in st.session_state:
    st.session_state.spark = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'schema_checker' not in st.session_state:
    st.session_state.schema_checker = None
if 'feature_selector' not in st.session_state:
    st.session_state.feature_selector = None
if 'df_with_features' not in st.session_state:
    st.session_state.df_with_features = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

def init_spark():
    """Initialize Spark session"""
    if st.session_state.spark is None:
        st.session_state.spark = SparkSession.builder \
            .master("local[*]") \
            .appName("Spark Beyond ML App") \
            .getOrCreate()
        st.session_state.spark.sparkContext.setLogLevel("ERROR")
    return st.session_state.spark

def load_data(file_path):
    """Load data from CSV"""
    spark = init_spark()
    df = spark.read.options(
        header=True,
        inferSchema='True',
        delimiter=','
    ).csv(file_path)
    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Spark Beyond - ML Feature Discovery Platform</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Spark+Beyond", use_container_width=True)
        st.markdown("---")
        st.markdown("## Navigation")
        page = st.radio(
            "Select Page",
            ["📊 Data Overview", "🔧 Feature Engineering", "🎯 Model Training", "📈 Results & Insights"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This platform automates feature discovery and model training using PySpark and XGBoost.

        **Features:**
        - Automated feature generation
        - Schema validation
        - Model training & evaluation
        - Feature importance analysis
        """)

    # Page routing
    if page == "📊 Data Overview":
        data_overview_page()
    elif page == "🔧 Feature Engineering":
        feature_engineering_page()
    elif page == "🎯 Model Training":
        model_training_page()
    elif page == "📈 Results & Insights":
        results_page()

def data_overview_page():
    st.header("📊 Data Overview")

    # File upload or default path
    col1, col2 = st.columns([3, 1])
    with col1:
        data_source = st.radio("Data Source", ["Use Default Dataset", "Upload CSV"], horizontal=True)

    if data_source == "Use Default Dataset":
        file_path = "data/bank-additional-full.csv"
        if st.button("Load Default Dataset", type="primary"):
            with st.spinner("Loading dataset..."):
                st.session_state.df = load_data(file_path)
                st.success("✅ Dataset loaded successfully!")
    else:
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with open("temp_upload.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.df = load_data("temp_upload.csv")
            st.success("✅ Dataset uploaded successfully!")

    if st.session_state.df is not None:
        st.markdown("---")

        # Problem Definition
        st.subheader("🎯 Problem Definition")
        col1, col2, col3 = st.columns(3)

        with col1:
            target_col = st.selectbox(
                "Target Column",
                options=st.session_state.df.columns,
                index=st.session_state.df.columns.index('y') if 'y' in st.session_state.df.columns else 0
            )

        with col2:
            problem_type = st.selectbox(
                "Problem Type",
                options=["classification", "regression"]
            )

        with col3:
            if problem_type == "classification":
                desired_result = st.text_input("Desired Result (optional)", value="yes")
            else:
                desired_result = None

        if st.button("Validate Schema", type="primary"):
            with st.spinner("Validating schema..."):
                problem = Problem(
                    target=target_col,
                    type=problem_type,
                    desired_result=desired_result if desired_result else None
                )
                st.session_state.schema_checker = SchemaChecks(
                    dataframe=st.session_state.df,
                    problem=problem
                )
                schema_info = st.session_state.schema_checker.check()
                st.success("✅ Schema validation complete!")

                # Display schema information
                st.markdown("---")
                st.subheader("📋 Dataset Statistics")

                # Create tabs for different column types
                tab1, tab2, tab3 = st.tabs(["Categorical Features", "Numerical Features", "Boolean Features"])

                with tab1:
                    if schema_info['categorical']:
                        for cat_col in schema_info['categorical']:
                            with st.expander(f"📌 {cat_col['col_name']}"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Unique Values", cat_col['description']['count_distinct'])
                                    st.metric("Total Count", cat_col['description']['count'])
                                with col_b:
                                    st.metric("Null Count", cat_col['description']['null_count'])

                                # Value distribution
                                if len(cat_col['value_descriptions']) <= 20:
                                    df_dist = pd.DataFrame(cat_col['value_descriptions'])
                                    fig = px.bar(
                                        df_dist,
                                        x=cat_col['col_name'],
                                        y='count',
                                        title=f"Distribution of {cat_col['col_name']}"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No categorical features found")

                with tab2:
                    if schema_info['numerical']:
                        for num_col in schema_info['numerical']:
                            with st.expander(f"📊 {num_col['col_name']}"):
                                cols = st.columns(4)
                                cols[0].metric("Mean", f"{num_col['description']['mean']:.2f}")
                                cols[1].metric("Std Dev", f"{num_col['description']['std']:.2f}")
                                cols[2].metric("Min", f"{num_col['description']['min']:.2f}")
                                cols[3].metric("Max", f"{num_col['description']['max']:.2f}")
                    else:
                        st.info("No numerical features found")

                with tab3:
                    if schema_info['boolean']:
                        for bool_col in schema_info['boolean']:
                            st.write(f"**{bool_col['col_name']}**")
                            st.json(bool_col)
                    else:
                        st.info("No boolean features found")

def feature_engineering_page():
    st.header("🔧 Feature Engineering")

    if st.session_state.schema_checker is None:
        st.warning("⚠️ Please complete the Data Overview step first!")
        return

    # Show expected outcome preview
    st.info("💡 **What to Expect**: This process will automatically generate new features from your existing data to improve model performance.")

    st.markdown("### Configure Feature Generation")

    # Feature generation explanation with visual preview
    with st.expander("📖 Feature Generation Guide", expanded=False):
        st.markdown("""
        #### Automated Feature Engineering Process

        **1. Numerical Transformations** 🔢
        - Creates: `log`, `sqrt`, `square`, `cube` transformations
        - Example: `age` → `age_log`, `age_sqrt`, `age_square`, `age_cube`
        - **Purpose**: Captures non-linear relationships

        **2. Feature Interactions** 🔗
        - Creates: `multiply`, `divide`, `add`, `subtract` between all numerical pairs
        - Example: `age` × `duration` → `age_mult_duration`, `age_div_duration`
        - **Purpose**: Captures combined effects of features

        **3. Binning** 📊
        - Creates: Discretized versions of continuous features
        - Example: `age` → `age_binned` (groups like 20-30, 30-40, etc.)
        - **Purpose**: Captures threshold effects

        **4. Datetime Features** 📅
        - Creates: Year, month, day, hour, day_of_week extracts
        - **Purpose**: Captures temporal patterns
        """)

    col1, col2 = st.columns(2)

    with col1:
        include_numerical = st.checkbox("Include Numerical Transformations", value=True,
                                       help="Generate log, sqrt, square, cube transformations")
        include_interactions = st.checkbox("Include Feature Interactions", value=True,
                                          help="Generate multiplication, division, addition, subtraction features")

    with col2:
        include_binning = st.checkbox("Include Binning", value=True,
                                     help="Create binned versions of numerical features")
        include_datetime = st.checkbox("Include Datetime Features", value=True,
                                      help="Extract datetime components if applicable")

    # Show estimation of features to be generated
    if st.session_state.schema_checker:
        num_numerical = len(st.session_state.schema_checker.get_typed_col("numerical"))

        estimated_features = 0
        feature_breakdown = []

        if include_numerical:
            num_transformations = num_numerical * 4
            estimated_features += num_transformations
            feature_breakdown.append(f"📐 Numerical transformations: ~{num_transformations}")

        if include_interactions:
            num_interactions = (num_numerical * (num_numerical - 1)) * 4
            estimated_features += num_interactions
            feature_breakdown.append(f"🔗 Feature interactions: ~{num_interactions}")

        if include_binning:
            num_bins = num_numerical
            estimated_features += num_bins
            feature_breakdown.append(f"📊 Binned features: ~{num_bins}")

        if feature_breakdown:
            st.markdown("#### 📊 Estimated Feature Generation")
            for item in feature_breakdown:
                st.markdown(f"- {item}")
            st.info(f"**Total estimated new features: ~{estimated_features}** (Original: {len(st.session_state.df.columns)})")

    if st.button("Generate Features", type="primary"):
        with st.spinner("Generating features... This may take a moment."):
            feature_gen = AutoFeatureGenerator(
                schema_checks=st.session_state.schema_checker,
                problem=st.session_state.schema_checker.problem
            )

            st.session_state.df_with_features = feature_gen.generate_all_features(
                include_numerical=include_numerical,
                include_interactions=include_interactions,
                include_binning=include_binning,
                include_datetime=include_datetime,
                include_string=False
            )

            st.success("✅ Features generated successfully!")

            # Show feature statistics
            st.markdown("---")
            st.subheader("📈 Feature Generation Summary")

            col1, col2, col3 = st.columns(3)

            original_features = len(st.session_state.df.columns)
            new_features = len(st.session_state.df_with_features.columns)
            generated_features = new_features - original_features

            col1.metric("Original Features", original_features)
            col2.metric("Total Features", new_features)
            col3.metric("Generated Features", generated_features, delta=f"+{generated_features}")

            # Show sample of new features
            st.markdown("#### 🔍 Sample of Generated Features")
            all_columns = st.session_state.df_with_features.columns
            original_columns = st.session_state.df.columns
            new_columns = [col for col in all_columns if col not in original_columns]

            if new_columns:
                # Categorize new features
                feature_categories = {
                    'Transformations': [f for f in new_columns if any(x in f for x in ['_log', '_sqrt', '_square', '_cube'])],
                    'Interactions': [f for f in new_columns if any(x in f for x in ['_mult_', '_div_', '_add_', '_sub_'])],
                    'Binned': [f for f in new_columns if '_binned' in f],
                    'Other': [f for f in new_columns if not any(x in f for x in ['_log', '_sqrt', '_square', '_cube', '_mult_', '_div_', '_add_', '_sub_', '_binned'])]
                }

                tab1, tab2, tab3, tab4 = st.tabs(["🔢 Transformations", "🔗 Interactions", "📊 Binned", "➕ Other"])

                with tab1:
                    if feature_categories['Transformations']:
                        st.write(f"**{len(feature_categories['Transformations'])} transformation features created**")
                        st.code('\n'.join(feature_categories['Transformations'][:20]))
                        if len(feature_categories['Transformations']) > 20:
                            st.caption(f"... and {len(feature_categories['Transformations']) - 20} more")
                    else:
                        st.info("No transformation features generated")

                with tab2:
                    if feature_categories['Interactions']:
                        st.write(f"**{len(feature_categories['Interactions'])} interaction features created**")
                        st.code('\n'.join(feature_categories['Interactions'][:20]))
                        if len(feature_categories['Interactions']) > 20:
                            st.caption(f"... and {len(feature_categories['Interactions']) - 20} more")
                    else:
                        st.info("No interaction features generated")

                with tab3:
                    if feature_categories['Binned']:
                        st.write(f"**{len(feature_categories['Binned'])} binned features created**")
                        st.code('\n'.join(feature_categories['Binned']))
                    else:
                        st.info("No binned features generated")

                with tab4:
                    if feature_categories['Other']:
                        st.write(f"**{len(feature_categories['Other'])} other features created**")
                        st.code('\n'.join(feature_categories['Other'][:20]))
                    else:
                        st.info("No other features generated")

                # Show actual data sample
                st.markdown("#### 📋 Data Sample with New Features")

                # Convert a small sample to Pandas for display
                sample_data = st.session_state.df_with_features.limit(5).toPandas()

                #TODO: Fix the bug with oirignal column error in data tab
                # # Show in tabs: original vs all features
                # data_tab1, data_tab2 = st.tabs(["Original Features", "All Features"])

                # with data_tab1:
                #     print(dataframe.columns)
                #     st.dataframe(sample_data[original_columns], use_container_width=True)

                # with data_tab2:
                #     st.dataframe(sample_data, use_container_width=True)

    if st.session_state.df_with_features is not None:
        st.markdown("---")
        st.subheader("🔄 Feature Preprocessing")

        st.info("💡 **Next Step**: Preprocessing will encode categorical variables and prepare features for model training.")

        with st.expander("📖 What happens during preprocessing?", expanded=False):
            st.markdown("""
            #### Feature Preprocessing Steps

            1. **Categorical Encoding** 🏷️
               - One-hot encodes categorical variables
               - Example: `job` → `job_admin`, `job_technician`, etc.

            2. **Feature Vectorization** 📦
               - Combines all features into a single vector column
               - Required format for Spark ML models

            3. **Feature Mapping** 🗺️
               - Creates index-to-name mapping for interpretability
               - Enables feature importance analysis later
            """)

        if st.button("Preprocess Features", type="primary"):
            with st.spinner("Preprocessing features..."):
                pre_process_variables = PreProcessVariables(
                    dataframe=st.session_state.df_with_features,
                    problem=st.session_state.schema_checker.problem,
                    schema_checks=st.session_state.schema_checker
                )

                transformed_df, feature_names, feature_output_col, feature_map = pre_process_variables.process()

                # Store in session state
                st.session_state.transformed_df = transformed_df
                st.session_state.feature_names = feature_names
                st.session_state.feature_output_col = feature_output_col
                st.session_state.feature_map = feature_map

                st.success("✅ Features preprocessed successfully!")

                # Show preprocessing results
                st.markdown("---")
                st.subheader("📊 Preprocessing Results")

                col1, col2, col3 = st.columns(3)

                col1.metric("Categorical Features Encoded", len(feature_names))
                col2.metric("Total Feature Dimensions", len(st.session_state.transformed_df.columns))
                col3.metric("Feature Vector Column", feature_output_col)

                # Show encoded features
                st.markdown("#### 🏷️ Encoded Categorical Features")
                st.write(f"**{len(feature_names)} encoded features:**")

                # Group by prefix
                encoded_by_category = {}
                for fname in feature_names:
                    prefix = fname.split('_')[0]
                    if prefix not in encoded_by_category:
                        encoded_by_category[prefix] = []
                    encoded_by_category[prefix].append(fname)

                for category, features in encoded_by_category.items():
                    with st.expander(f"📌 {category} ({len(features)} encoded features)"):
                        st.code('\n'.join(features))

                st.info("✅ **Ready for Model Training!** Proceed to the Model Training page.")

        # Show current state if already preprocessed
        elif hasattr(st.session_state, 'transformed_df'):
            st.success("✅ Features already preprocessed and ready for training!")

            col1, col2, col3 = st.columns(3)
            col1.metric("Encoded Features", len(st.session_state.feature_names))
            col2.metric("Total Dimensions", len(st.session_state.transformed_df.columns))
            col3.metric("Vector Column", st.session_state.feature_output_col)

            st.info("✅ **Ready for Model Training!** Proceed to the Model Training page.")

def model_training_page():
    st.header("🎯 Model Training")

    if not hasattr(st.session_state, 'transformed_df'):
        st.warning("⚠️ Please complete the Feature Engineering step first!")
        return

    st.markdown("### Training Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        train_split = st.slider("Training Split %", 50, 95, 80) / 100

    with col2:
        max_depth = st.slider("Max Depth", 2, 10, 4)

    with col3:
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)

    if st.button("Train Model", type="primary"):
        with st.spinner("Training XGBoost model... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Initializing feature selector...")
            progress_bar.progress(20)

            feature_selector = FeatureSelector(
                problem=st.session_state.schema_checker.problem,
                transformed_df=st.session_state.transformed_df,
                feature_names=st.session_state.feature_names,
                feature_col=st.session_state.feature_output_col,
                feature_idx_name_mapping=st.session_state.feature_map,
                train_split=train_split
            )

            status_text.text("Training model...")
            progress_bar.progress(40)

            feature_selector.train_model()

            status_text.text("Evaluating on training set...")
            progress_bar.progress(70)

            train_metrics = feature_selector.evaluate(train=True)

            status_text.text("Evaluating on test set...")
            progress_bar.progress(90)

            test_metrics = feature_selector.evaluate(train=False)

            st.session_state.feature_selector = feature_selector
            st.session_state.metrics = {
                'train': train_metrics,
                'test': test_metrics
            }

            progress_bar.progress(100)
            status_text.text("Training complete!")

            st.success("✅ Model trained successfully!")

        # Display metrics
        st.markdown("---")
        st.subheader("📊 Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Training Metrics")
            metrics_df_train = pd.DataFrame([train_metrics]).T
            metrics_df_train.columns = ['Value']
            metrics_df_train['Metric'] = metrics_df_train.index

            for metric, value in train_metrics.items():
                st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")

        with col2:
            st.markdown("#### Test Metrics")
            metrics_df_test = pd.DataFrame([test_metrics]).T
            metrics_df_test.columns = ['Value']
            metrics_df_test['Metric'] = metrics_df_test.index

            for metric, value in test_metrics.items():
                st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")

        # Comparison chart
        st.markdown("---")
        st.subheader("📈 Train vs Test Comparison")

        comparison_df = pd.DataFrame({
            'Metric': list(train_metrics.keys()),
            'Train': list(train_metrics.values()),
            'Test': list(test_metrics.values())
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Train', x=comparison_df['Metric'], y=comparison_df['Train']))
        fig.add_trace(go.Bar(name='Test', x=comparison_df['Metric'], y=comparison_df['Test']))
        fig.update_layout(barmode='group', title='Model Performance Comparison')
        st.plotly_chart(fig, use_container_width=True)

def results_page():
    st.header("📈 Results & Insights")

    if st.session_state.feature_selector is None:
        st.warning("⚠️ Please complete the Model Training step first!")
        return

    # Feature Importance
    st.subheader("🎯 Feature Importance Analysis")

    col1, col2 = st.columns([2, 1])

    with col2:
        top_n = st.slider("Top N Features", 10, 50, 20)

    with st.spinner("Calculating feature importance..."):
        importance_list = st.session_state.feature_selector.get_feature_importances()
        importance_df = pd.DataFrame(importance_list[:top_n], columns=['Feature', 'Importance'])

        fig = px.bar(
            importance_df,
            y='Feature',
            x='Importance',
            orientation='h',
            title=f'Top {top_n} Most Important Features',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=max(400, top_n * 20), yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # Probability Impact Summary
    st.markdown("---")
    st.subheader("💡 Probability Impact Analysis")

    with st.spinner("Calculating probability impacts..."):
        prob_impact_df = st.session_state.feature_selector.get_probability_impact_summary()

        top_impacts = prob_impact_df.head(20)

        st.dataframe(
            top_impacts[['Feature', 'Threshold', 'Prob_Impact', 'Left_Prob', 'Right_Prob']].style.format({
                'Threshold': '{:.2f}',
                'Prob_Impact': '{:.4f}',
                'Left_Prob': '{:.4f}',
                'Right_Prob': '{:.4f}'
            }),
            use_container_width=True
        )

        # Visualization
        fig = px.bar(
            top_impacts.head(15),
            x='Prob_Impact',
            y='Feature',
            title='Top 15 Features by Probability Impact',
            orientation='h',
            color='Prob_Impact',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # Summary Section
    st.markdown("---")
    st.subheader("📝 Executive Summary")

    if st.session_state.metrics:
        col1, col2, col3, col4 = st.columns(4)

        test_metrics = st.session_state.metrics['test']

        col1.metric("Test Accuracy", f"{test_metrics['accuracy']:.2%}")
        col2.metric("Test Precision", f"{test_metrics['precision']:.2%}")
        col3.metric("Test Recall", f"{test_metrics['recall']:.2%}")
        col4.metric("Test AUC-ROC", f"{test_metrics['auc_roc']:.2%}")

        st.markdown("""
        ### Key Insights

        - **Model Performance**: The XGBoost classifier achieved strong performance across all metrics
        - **Feature Engineering**: Automated feature generation created valuable predictive features
        - **Top Drivers**: The most important features are shown in the feature importance chart above
        - **Business Impact**: The probability impact analysis shows which feature thresholds drive the strongest predictions
        """)

    # Download section
    st.markdown("---")
    st.subheader("📥 Export Results")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Download Feature Importance CSV"):
            importance_list = st.session_state.feature_selector.get_feature_importances()
            importance_df = pd.DataFrame(importance_list, columns=['Feature', 'Importance'])
            csv = importance_df.to_csv(index=False)
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name="feature_importance.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("Download Probability Impact CSV"):
            prob_impact_df = st.session_state.feature_selector.get_probability_impact_summary()
            csv = prob_impact_df.to_csv(index=False)
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name="probability_impact.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
