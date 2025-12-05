import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer

def extract_unsigned_numbers_from_string(s):
    if pd.isna(s):
        return []
    return [float(x) for x in re.findall(r'\d+\.?\d*', str(s))]

def convert_str_torque_to_numbers(torque_str, nm_avg=None, kgm_avg=None):
    if pd.isna(torque_str):
        return np.nan, np.nan, np.nan, np.nan
    
    torque_str = str(torque_str).lower()
    numbers = extract_unsigned_numbers_from_string(torque_str)
    
    torque_value_nm = np.nan
    torque_value_kgm = np.nan
    rpm_min = np.nan
    rpm_max = np.nan
    
    if not numbers:
        return np.nan, np.nan, np.nan, np.nan

    # Extract torque value
    torque_value = numbers[0]
    
    # Handle units
    if 'nm' in torque_str:
        torque_value_nm = torque_value
        torque_value_kgm = torque_value * 0.10197
    elif 'kgm' in torque_str:
        torque_value_nm = torque_value * 9.80665
        torque_value_kgm = torque_value
    else:
        # Inference based on averages if unit is missing (simplified from notebook)
        if nm_avg and kgm_avg:
            dist_nm = abs(torque_value - nm_avg)
            dist_kgm = abs(torque_value - kgm_avg)
            if dist_nm < dist_kgm:
                torque_value_nm = torque_value
                torque_value_kgm = torque_value * 0.10197
            else:
                torque_value_nm = torque_value * 9.80665
                torque_value_kgm = torque_value
        else:
             # Default to Nm if no averages provided (or handle as NaN)
             torque_value_nm = torque_value
             torque_value_kgm = torque_value * 0.10197

    # Extract RPM
    if len(numbers) > 1:
        if len(numbers) == 2:
            rpm_max = numbers[1]
            rpm_min = rpm_max # Assume constant if only one RPM given
        elif len(numbers) >= 3:
            rpm_min = numbers[1]
            rpm_max = numbers[2]
            
    return torque_value_nm, torque_value_kgm, rpm_min, rpm_max

def convert_torque_row(row):
    # We need averages for inference, but for simplicity in this utility 
    # we might skip the complex inference or calculate it beforehand if needed.
    # Here we'll use a simplified version or pass None for averages if not available globally yet.
    # In the notebook, averages were calculated from the dataset. 
    # Let's try to be robust without them for single-row processing or calculate them in the main processing function.
    return convert_str_torque_to_numbers(row['torque'])

def clean_numeric_column(series, unit_str):
    """Removes unit string and converts to float."""
    return series.astype(str).str.replace(unit_str, '', regex=False).str.replace(',', '', regex=False).astype(float)

def load_and_process_data(file):
    df = pd.read_csv(file)
    
    # Drop duplicates
    df = df.drop_duplicates(keep='first')
    
    # Clean Mileage
    # Some mileage might be in km/kg, others in kmpl. The notebook treated them numerically.
    # We'll extract the first number.
    df['mileage_num'] = df['mileage'].apply(lambda x: extract_unsigned_numbers_from_string(x)[0] if extract_unsigned_numbers_from_string(x) else np.nan)
    
    # Clean Engine
    df['engine_num'] = df['engine'].apply(lambda x: extract_unsigned_numbers_from_string(x)[0] if extract_unsigned_numbers_from_string(x) else np.nan)
    
    # Clean Max Power
    df['max_power_num'] = df['max_power'].apply(lambda x: extract_unsigned_numbers_from_string(x)[0] if extract_unsigned_numbers_from_string(x) else np.nan)
    
    # Clean Torque
    # Calculate averages for inference if needed (simplified approach: just use median of knowns if we were strictly following notebook, 
    # but here we'll let the helper function handle it per row)
    # To do it properly like the notebook, we'd need a first pass to get averages of explicit Nm/kgm.
    # For this app, we'll use the per-row logic which defaults to Nm if ambiguous and no avgs passed.
    
    torque_data = df.apply(convert_torque_row, axis=1)
    df['torque_nm'] = torque_data.apply(lambda x: x[0])
    df['torque_rpm_max'] = torque_data.apply(lambda x: x[3])
    
    # Fill NaNs (Simple median imputation for numeric)
    numeric_cols = ['year', 'selling_price', 'km_driven', 'seats', 'mileage_num', 'engine_num', 'max_power_num', 'torque_nm', 'torque_rpm_max']
    # Ensure columns exist (selling_price might not be in test set if we were doing a competition, but here we assume it is for EDA/Training)
    available_numeric = [c for c in numeric_cols if c in df.columns]
    
    imputer = SimpleImputer(strategy='median')
    df[available_numeric] = imputer.fit_transform(df[available_numeric])
    
    # Convert seats to int
    if 'seats' in df.columns:
        df['seats'] = df['seats'].astype(int)
        
    return df

def prepare_data_for_model(df, target_col='selling_price'):
    # Drop 'name' and original string columns
    drop_cols = ['name', 'torque', 'mileage', 'engine', 'max_power']
    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # One-Hot Encoding for categorical variables
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    # Ensure they exist
    categorical_cols = [c for c in categorical_cols if c in df_model.columns]
    
    df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
    
    # Feature Engineering: Age
    if 'year' in df_model.columns:
        df_model['age'] = 2026 - df_model['year']
        df_model = df_model.drop(columns=['year'])
        
    # Separate X and y
    if target_col in df_model.columns:
        X = df_model.drop(columns=[target_col])
        y = df_model[target_col]
        return X, y
    else:
        return df_model, None

def train_models(X, y):
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    results = []
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_scaled, y)
    y_pred_lr = lr.predict(X_scaled)
    results.append({
        'Model': 'Linear Regression',
        'R2': r2_score(y, y_pred_lr),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred_lr)),
        'Coefficients': pd.Series(lr.coef_, index=X.columns),
        'ModelObject': lr
    })
    
    # Lasso
    lasso = Lasso(alpha=0.1) # Default alpha or tuned
    lasso.fit(X_scaled, y)
    y_pred_lasso = lasso.predict(X_scaled)
    results.append({
        'Model': 'Lasso',
        'R2': r2_score(y, y_pred_lasso),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred_lasso)),
        'Coefficients': pd.Series(lasso.coef_, index=X.columns),
        'ModelObject': lasso
    })
    
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    y_pred_ridge = ridge.predict(X_scaled)
    results.append({
        'Model': 'Ridge',
        'R2': r2_score(y, y_pred_ridge),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred_ridge)),
        'Coefficients': pd.Series(ridge.coef_, index=X.columns),
        'ModelObject': ridge
    })
    
    # ElasticNet
    en = ElasticNet(alpha=0.1, l1_ratio=0.5)
    en.fit(X_scaled, y)
    y_pred_en = en.predict(X_scaled)
    results.append({
        'Model': 'ElasticNet',
        'R2': r2_score(y, y_pred_en),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred_en)),
        'Coefficients': pd.Series(en.coef_, index=X.columns),
        'ModelObject': en
    })
    
    # Return scaler as well so we can use it for inference later
    return results, scaler
