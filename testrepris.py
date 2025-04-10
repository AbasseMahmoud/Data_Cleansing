from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import re

app = Flask(__name__)

def is_age_column(col_name):
    """Détecte si une colonne est une colonne d'âge"""
    age_keywords = ['age', 'âge', 'years', 'year', 'an', 'ans']
    return any(keyword in col_name.lower() for keyword in age_keywords)

def clean_age_values(age_series):
    """Nettoyage spécifique pour les valeurs d'âge"""
    cleaned_ages = []
    for value in age_series:
        if pd.isna(value):
            cleaned_ages.append(np.nan)
            continue
            
        # Convertir en chaîne si ce n'est pas déjà le cas
        str_value = str(value).strip()
        
        # Extraire les chiffres seulement
        digits = re.sub(r'[^\d]', '', str_value)
        
        if digits:  # Si on a trouvé des chiffres
            age = int(digits)
            # Appliquer des limites raisonnables (0-120 ans)
            cleaned_age = max(0, min(120, age))
            cleaned_ages.append(cleaned_age)
        else:
            cleaned_ages.append(np.nan)
            
    return pd.Series(cleaned_ages, index=age_series.index)

def clean_dataframe(df):
    """Nettoyage générique pour n'importe quel dataframe"""
    cleaned_df = df.copy()
    for col in cleaned_df.columns:
        # Traitement spécial pour les colonnes d'âge
        if is_age_column(col):
            cleaned_df[col] = clean_age_values(cleaned_df[col])
            continue
            
        try:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='raise')
            continue
        except:
            pass
        
        if cleaned_df[col].dtype == object:
            cleaned_df[col] = cleaned_df[col].replace(
                ['n/a', 'NA', 'na', 'NaN', 'NULL', 'nan', 'null', '--', '?', ''], np.nan)
            unique_vals = cleaned_df[col].dropna().unique()
            if all(str(x).upper() in ['Y', 'N', 'OUI', 'NON', 'YES', 'NO'] for x in unique_vals if pd.notnull(x)):
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: str(x).upper() if pd.notnull(x) else np.nan)
    return cleaned_df

def handle_missing_values(df):
    """Détection et traitement des valeurs manquantes"""
    missing_before = df.isnull().sum().to_dict()
    for col in df.columns:
        if is_age_column(col):
            # Pour les âges, on utilise la médiane avec un minimum de 0
            median_val = max(0, df[col].median())
            df[col] = df[col].fillna(median_val)
        elif pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else ''
            df[col] = df[col].fillna(mode_val)
    missing_after = df.isnull().sum().to_dict()
    return df, {'before': missing_before, 'after': missing_after}

def handle_outliers(df):
    """Traitement des valeurs aberrantes"""
    report = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if is_age_column(col):
            # Traitement spécial pour l'âge
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = max(0, q1 - 1.5 * iqr)  # Minimum 0 pour l'âge
                upper_bound = min(120, q3 + 1.5 * iqr)  # Maximum 120 pour l'âge
                
                # Appliquer les corrections
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                # Générer le rapport
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    report[col] = {
                        'count': int(outliers_count),
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
        else:
            # Traitement standard pour les autres colonnes numériques
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    report[col] = {
                        'count': int(outliers_count),
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
    return df, report

def handle_duplicates(df, subset=None):
    """
    Gestion améliorée des doublons
    :param subset: liste de colonnes à considérer pour la détection des doublons
    """
    dup_mask = df.duplicated(subset=subset, keep='first')
    dup_count = dup_mask.sum()
    
    if dup_count > 0:
        df = df[~dup_mask]
        return df, {
            'count': dup_count,
            'message': f"{dup_count} doublons supprimés",
            'columns_used': subset if subset else "toutes les colonnes"
        }
    return df, {'message': "Aucun doublon détecté"}

@app.route('/upload/')
def upload():
    return render_template('upload.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/traitement', methods=['POST'])
def traitement():
    if 'file' not in request.files:
        return render_template('error.html', message="Aucun fichier sélectionné")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', message="Aucun fichier sélectionné")
    
    # Lecture du fichier
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return render_template('error.html', message="Seuls les fichiers CSV et JSON sont acceptés")
    except Exception as e:
        return render_template('error.html', message=f"Erreur de lecture du fichier : {str(e)}")
    
    original_shape = df.shape
    
    # Traitement des données
    df = clean_dataframe(df)
    df, duplicates_report = handle_duplicates(df)
    df, missing_report = handle_missing_values(df)
    df, outliers_report = handle_outliers(df)
    
    # Vérification finale des colonnes d'âge
    age_columns = [col for col in df.columns if is_age_column(col)]
    for col in age_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].clip(lower=0, upper=120)  # Force 0-120 ans
    
    # Sauvegarde du fichier traité
    output_file = 'static/fichier_traiter_data.csv'
    df.to_csv(output_file, index=False)
    
    final_shape = df.shape
    sample_data = df.head(10).to_dict('records')
    column_types = {col: str(df[col].dtype) for col in df.columns}
    
    return render_template('results.html',
                         original_shape=original_shape,
                         final_shape=final_shape,
                         missing_report=missing_report,
                         outliers_report=outliers_report,
                         duplicates_report=duplicates_report,
                         sample_data=sample_data,
                         column_types=column_types)

@app.route('/download')
def download():
    file_path = 'static/fichier_traiter_data.csv'
    return send_file(file_path, as_attachment=True, mimetype='text/csv', download_name='fichier_traiter_data.csv')

if __name__ == '__main__':
    app.run(debug=True)