import streamlit as st
import pandas as pd
import plotly.express as px
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io

st.set_page_config(page_title="Car Price EDA & Prediction", layout="wide")

st.title("Car Price EDA & Prediction App")

# Sidebar
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("Model Import")
uploaded_model = st.sidebar.file_uploader("Upload Model (.pkl)", type=["pkl", "pickle"])

if uploaded_model is not None:
    try:
        imported_data = pickle.load(uploaded_model)
        st.sidebar.success(f"Model Loaded Successfully!")
        
        # Initialize trained_results in session state if not present
        if 'trained_results' not in st.session_state:
            st.session_state['trained_results'] = []
            
        new_result = None
        
        # Check if it's our exported format (dict)
        if isinstance(imported_data, dict) and 'model' in imported_data:
            model_obj = imported_data['model']
            metrics = imported_data.get('metrics', {})
            coefs = imported_data.get('coefficients', pd.Series())
            
            # Use filename as model name if possible, or generic
            model_name = f"Imported: {uploaded_model.name}"
            
            new_result = {
                'Model': model_name,
                'R2': metrics.get('R2', 0.0),
                'RMSE': metrics.get('RMSE', 0.0),
                'Coefficients': coefs,
                'ModelObject': model_obj
            }
            
            if 'metrics' in imported_data:
                st.sidebar.subheader("Imported Model Metrics")
                st.sidebar.write(f"R2: {metrics.get('R2', 'N/A')}")
                st.sidebar.write(f"RMSE: {metrics.get('RMSE', 'N/A')}")

        # Check if it's a raw sklearn model (has predict method)
        elif hasattr(imported_data, 'predict'):
            model_name = f"Imported Raw: {uploaded_model.name}"
            
            # Try to extract coefficients if available
            coefs = pd.Series()
            if hasattr(imported_data, 'coef_'):
                # We don't know feature names for sure, so we'll use indices or try to infer if we have X
                # For now, just raw indices
                coefs = pd.Series(imported_data.coef_)
            
            new_result = {
                'Model': model_name,
                'R2': 0.0, # Unknown
                'RMSE': 0.0, # Unknown
                'Coefficients': coefs,
                'ModelObject': imported_data
            }
            st.sidebar.info("Raw model imported. Metrics unknown.")
            
        else:
            st.sidebar.error("Uploaded file does not contain a recognized model format.")
            
        # Add to session state if valid
        if new_result:
            # Avoid duplicates by name
            existing_names = [r['Model'] for r in st.session_state['trained_results']]
            if new_result['Model'] not in existing_names:
                st.session_state['trained_results'].append(new_result)
                st.sidebar.success(f"Added '{new_result['Model']}' to Models tab.")
            else:
                st.sidebar.warning(f"Model '{new_result['Model']}' already exists.")
            
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

if uploaded_file is not None:
    try:
        df = utils.load_and_process_data(uploaded_file)
        st.sidebar.success("File uploaded and processed successfully!")
    except Exception as e:
        st.sidebar.error(f"Error processing file: {e}")
        st.stop()
else:
    st.info("Please upload a CSV file to proceed. The file should have columns like 'name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats'.")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["General", "Custom", "Models", "Evaluation"])

# General Tab
with tab1:
    st.header("General Data Overview")
    
    st.subheader("First 5 Rows")
    st.dataframe(df.head())
    
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Selling Price")
        fig_hist = px.histogram(df, x="selling_price", nbins=50, title="Selling Price Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        st.subheader("Correlation Heatmap (Numeric)")
        numeric_df = df.select_dtypes(include=['float64', 'int64', 'int32'])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("No numeric columns found for correlation.")

# Custom Tab
with tab2:
    st.header("Custom Visualization")
    
    col_x, col_y, col_color, col_type = st.columns(4)
    
    columns = df.columns.tolist()
    
    with col_x:
        x_axis = st.selectbox("X-Axis", columns, index=columns.index('km_driven') if 'km_driven' in columns else 0)
    with col_y:
        y_axis = st.selectbox("Y-Axis", columns, index=columns.index('selling_price') if 'selling_price' in columns else 0)
    with col_color:
        color_opts = ["None"] + columns
        color_by = st.selectbox("Color By", color_opts, index=0)
    with col_type:
        plot_type = st.selectbox("Plot Type", ["Scatter", "Histogram", "Box", "Bar"])
        
    fig = None
    if plot_type == "Scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis, color=None if color_by == "None" else color_by, title=f"{plot_type} Plot of {y_axis} vs {x_axis}")
    elif plot_type == "Histogram":
        fig = px.histogram(df, x=x_axis, y=y_axis, color=None if color_by == "None" else color_by, title=f"{plot_type} of {x_axis}")
    elif plot_type == "Box":
        fig = px.box(df, x=x_axis, y=y_axis, color=None if color_by == "None" else color_by, title=f"{plot_type} Plot of {y_axis} by {x_axis}")
    elif plot_type == "Bar":
        # Aggregate for bar chart to avoid mess
        if color_by != "None":
             df_agg = df.groupby([x_axis, color_by])[y_axis].mean().reset_index()
             fig = px.bar(df_agg, x=x_axis, y=y_axis, color=color_by, title=f"Mean {y_axis} by {x_axis}")
        else:
             df_agg = df.groupby(x_axis)[y_axis].mean().reset_index()
             fig = px.bar(df_agg, x=x_axis, y=y_axis, title=f"Mean {y_axis} by {x_axis}")

    if fig:
        st.plotly_chart(fig, use_container_width=True)

# Models Tab
with tab3:
    st.header("Model Training & Evaluation")
    
    st.write("Train linear models (Linear Regression, Lasso, Ridge, ElasticNet) to predict `selling_price`.")
    st.write("Note: `name` column is dropped to avoid high dimensionality. `torque`, `mileage`, `engine`, `max_power` are parsed. `age` feature is engineered.")
    
    if st.button("Train Models"):
        with st.spinner("Training models... This might take a moment."):
            try:
                X, y = utils.prepare_data_for_model(df)
                if y is None:
                    st.error("Target column 'selling_price' not found in dataset.")
                else:
                    results, scaler = utils.train_models(X, y)
                    st.session_state['trained_results'] = results
                    st.session_state['scaler'] = scaler
                    st.success("Models trained successfully!")
                                
            except Exception as e:
                st.error(f"Error during training: {e}")

    if 'trained_results' in st.session_state:
        results = st.session_state['trained_results']
        
        # Metrics Table
        metrics_df = pd.DataFrame(results)[['Model', 'R2', 'RMSE']]
        st.subheader("Model Metrics")
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['R2'], color='lightgreen').highlight_min(axis=0, subset=['RMSE'], color='lightgreen'))
        
        # Coefficients
        st.subheader("Feature Importance (Coefficients)")
        
        model_names = [r['Model'] for r in results]
        selected_model = st.selectbox("Select Model to View Coefficients", model_names)
        
        for res in results:
            if res['Model'] == selected_model:
                coefs = res['Coefficients']
                
                # Top 30 absolute coefficients
                coefs_abs = coefs.abs().sort_values(ascending=False)
                top_30 = coefs_abs.head(30)
                
                # Sum of others
                others_sum = coefs_abs.iloc[30:].sum() if len(coefs_abs) > 30 else 0
                
                # Prepare data for plot
                plot_data = pd.DataFrame({'Feature': top_30.index, 'Abs Coefficient': top_30.values})
                if others_sum > 0:
                    new_row = pd.DataFrame({'Feature': ['Sum of Others'], 'Abs Coefficient': [others_sum]})
                    plot_data = pd.concat([plot_data, new_row], ignore_index=True)
                
                fig_coef = px.bar(plot_data, x='Feature', y='Abs Coefficient', title=f"Top 30 Coefficients for {selected_model}")
                st.plotly_chart(fig_coef, use_container_width=True)
                
                # Show raw coefficients if needed
                with st.expander("View All Coefficients"):
                    coefs = coefs.reindex(coefs_abs.index)
                    st.write(coefs)

        # Export Section
        st.markdown("---")
        st.header("Export Model")
        model_to_export_name = st.selectbox("Select Model to Export", model_names, key="export_model_select")
        
        if model_to_export_name:
            model_to_export = next(r for r in results if r['Model'] == model_to_export_name)
            
            # Create a dictionary to pickle
            export_data = {
                'model': model_to_export['ModelObject'],
                'scaler': st.session_state.get('scaler'),
                'metrics': {k:v for k,v in model_to_export.items() if k in ['R2', 'RMSE']},
                'coefficients': model_to_export['Coefficients']
            }
            
            buffer = io.BytesIO()
            pickle.dump(export_data, buffer)
            buffer.seek(0)
            
            st.download_button(
                label=f"Download {model_to_export_name} Model",
                data=buffer,
                file_name=f"{model_to_export_name.replace(' ', '_').lower()}_model.pickle",
                mime="application/octet-stream"
            )

# Evaluation Tab
with tab4:
    st.header("Model Evaluation & Prediction")
    
    if 'trained_results' not in st.session_state or not st.session_state['trained_results']:
        st.warning("No models available. Please train models in the 'Models' tab or upload a model.")
    else:
        results = st.session_state['trained_results']
        model_names = [r['Model'] for r in results]
        
        selected_eval_model_name = st.selectbox("Select Model for Evaluation", model_names, key="eval_model_select")
        selected_eval_model = next(r for r in results if r['Model'] == selected_eval_model_name)
        model = selected_eval_model['ModelObject']
        
        st.markdown("---")
        
        # --- Manual Prediction ---
        st.subheader("Single Observation Prediction")
        
        # Input Form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            # Defaults or ranges could be dynamic if df is available, but we use safe defaults
            with col1:
                year = st.number_input("Year", min_value=1980, max_value=2025, value=2015)
                km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
                seats = st.number_input("Seats", min_value=2, max_value=14, value=5)
                owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

            with col2:
                fuel = st.selectbox("Fuel", ['Diesel', 'Petrol', 'CNG', 'LPG'])
                seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
                transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
                
            with col3:
                mileage = st.text_input("Mileage (e.g., 23.4 kmpl)", "20.0 kmpl")
                engine = st.text_input("Engine (e.g., 1248 CC)", "1200 CC")
                max_power = st.text_input("Max Power (e.g., 74 bhp)", "80 bhp")
                torque = st.text_input("Torque (e.g., 190Nm@ 2000rpm)", "100Nm")

            submit_val = st.form_submit_button("Predict Price")
            
            if submit_val:
                # Create a DataFrame for the single observation
                data = {
                    'year': [year],
                    'km_driven': [km_driven],
                    'fuel': [fuel],
                    'seller_type': [seller_type],
                    'transmission': [transmission],
                    'owner': [owner],
                    'mileage': [mileage],
                    'engine': [engine],
                    'max_power': [max_power],
                    'torque': [torque],
                    'seats': [seats]
                }
                
                single_df = pd.DataFrame(data)
                
                # Preprocess
                # We need to apply the same steps as utils.load_and_process_data but for a single row
                # and then utils.prepare_data_for_model
                
                # 1. Clean numeric strings
                single_df['mileage_num'] = single_df['mileage'].apply(lambda x: utils.extract_unsigned_numbers_from_string(x)[0] if utils.extract_unsigned_numbers_from_string(x) else 0)
                single_df['engine_num'] = single_df['engine'].apply(lambda x: utils.extract_unsigned_numbers_from_string(x)[0] if utils.extract_unsigned_numbers_from_string(x) else 0)
                single_df['max_power_num'] = single_df['max_power'].apply(lambda x: utils.extract_unsigned_numbers_from_string(x)[0] if utils.extract_unsigned_numbers_from_string(x) else 0)
                
                torque_data = single_df.apply(utils.convert_torque_row, axis=1)
                single_df['torque_nm'] = torque_data.apply(lambda x: x[0] if not pd.isna(x[0]) else 0) # Handle NaN
                single_df['torque_rpm_max'] = torque_data.apply(lambda x: x[3] if not pd.isna(x[3]) else 0)
                
                # 2. Prepare for model (One-Hot Encoding & Feature Engineering)
                # We need to ensure columns match the trained model.
                # This is tricky because pd.get_dummies on a single row might miss categories.
                # We need to align with the model's coefficients or expected features.
                
                # Feature Engineering: Age
                single_df['age'] = 2026 - single_df['year']
                single_df = single_df.drop(columns=['year'])
                
                # Drop raw columns
                drop_cols = ['name', 'torque', 'mileage', 'engine', 'max_power']
                single_df = single_df.drop(columns=[c for c in drop_cols if c in single_df.columns])
                
                # One-Hot Encoding
                categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
                single_df = pd.get_dummies(single_df, columns=categorical_cols, drop_first=True)
                
                # ALIGNMENT
                # Get expected features from coefficients
                expected_features = selected_eval_model['Coefficients'].index.tolist()
                
                # Add missing columns with 0
                for col in expected_features:
                    if col not in single_df.columns:
                        single_df[col] = 0
                        
                # Drop extra columns (if any, though unlikely with this form unless new categories appear)
                single_df = single_df[expected_features]
                
                # Scale
                if 'scaler' in st.session_state and st.session_state['scaler'] is not None:
                    scaler = st.session_state['scaler']
                    try:
                        X_single_scaled = scaler.transform(single_df)
                        
                        # Predict
                        prediction = model.predict(X_single_scaled)
                        st.success(f"Predicted Selling Price: {prediction[0]:,.2f}")
                    except Exception as e:
                        st.error(f"Error during scaling/prediction: {e}")
                else:
                    st.error("Scaler not found. Cannot predict.")

        st.markdown("---")
        
        # --- Batch Evaluation ---
        st.subheader("Batch Evaluation on Uploaded Data")
        
        if uploaded_file is not None:
            # We assume df is already loaded in the main script scope
            # Prepare data
            try:
                X_batch, y_batch = utils.prepare_data_for_model(df)
                
                # Check alignment
                expected_features = selected_eval_model['Coefficients'].index.tolist()
                
                # Check for missing columns
                missing_cols = [col for col in expected_features if col not in X_batch.columns]
                extra_cols = [col for col in X_batch.columns if col not in expected_features]
                
                if not missing_cols:
                    # Align columns (order matters for scaler/model)
                    X_batch = X_batch[expected_features]
                    
                    if 'scaler' in st.session_state and st.session_state['scaler'] is not None:
                        scaler = st.session_state['scaler']
                        X_batch_scaled = scaler.transform(X_batch)
                        
                        y_pred_batch = model.predict(X_batch_scaled)
                        
                        if y_batch is not None:
                            r2 = utils.r2_score(y_batch, y_pred_batch)
                            rmse = utils.np.sqrt(utils.mean_squared_error(y_batch, y_pred_batch))
                            
                            st.write(f"**R2 Score:** {r2:.4f}")
                            st.write(f"**RMSE:** {rmse:,.2f}")
                            
                            # Optional: Plot actual vs predicted
                            fig_eval = px.scatter(x=y_batch, y=y_pred_batch, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
                            fig_eval.add_shape(type="line", line=dict(dash='dash'), x0=y_batch.min(), y0=y_batch.max(), x1=y_batch.min(), y1=y_batch.max())
                            st.plotly_chart(fig_eval, use_container_width=True)
                        else:
                            st.info("Target column 'selling_price' not found in uploaded data. Cannot calculate metrics.")
                            st.write("Predictions:", y_pred_batch[:5])
                    else:
                        st.error("Scaler not found.")
                else:
                    st.error("Columns do not match the model.")
                    with st.expander("See Mismatched Columns"):
                        st.write("Missing Columns (needed by model):")
                        st.write(missing_cols)
                        st.write("Extra Columns (in dataset but not in model):")
                        st.write(extra_cols)
                        
            except Exception as e:
                st.error(f"Error during batch evaluation: {e}")
        else:
            st.info("Upload a CSV file in the sidebar to see batch metrics.")
