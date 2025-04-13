from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import re

app = Flask(__name__)

def is_age_column(col_name):
    """Détecte si une colonne est une colonne d'âge"""
    age_keywords = ['age', 'âge', 'years', 'year', 'an', 'ans']
    return any(keyword in col_name.lower() for keyword in age_keywords)

def is_id_column(col_name):
    """Détecte si une colonne est une colonne d'identifiant"""
    id_keywords = ['id', 'identifiant', 'matricule', 'numero', 'code']
    return any(keyword in col_name.lower() for keyword in id_keywords)

def clean_age_values(age_series):
    """Nettoyage spécifique pour les valeurs d'âge avec conversion en entiers"""
    cleaned_ages = []
    for value in age_series:
        if pd.isna(value):
            cleaned_ages.append(np.nan)
            continue
            
        str_value = str(value).strip()
        digits = re.sub(r'[^\d.]', '', str_value)
        
        if digits:
            try:
                age = float(digits)
                cleaned_age = int(max(0, min(120, age)))  # Limite à 120 ans
                cleaned_ages.append(cleaned_age)
            except:
                cleaned_ages.append(np.nan)
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
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').astype('Int64')
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
            median_val = int(max(0, min(120, np.nanmedian(df[col].dropna()))))
            df[col] = df[col].fillna(median_val).astype('Int64')
        elif pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else ''
            df[col] = df[col].fillna(mode_val)
    missing_after = df.isnull().sum().to_dict()
    return df, {'before': missing_before, 'after': missing_after}

def handle_outliers(df):
    """Traitement des valeurs aberrantes avec méthode IQR"""
    report = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if is_age_column(col):
            # Traitement spécial pour l'âge
            lower_bound = 0
            upper_bound = 120
            
            # Compter les outliers avant correction
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers_count > 0:
                # Corriger les valeurs
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound).astype('Int64')
                report[col] = {
                    'count': int(outliers_count),
                    'valeurs_inférieures': lower_bound,
                    'valeurs_supérieures': upper_bound
                }
        else:
            # Traitement standard pour les autres colonnes numériques
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Compter les outliers avant correction
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                if outliers_count > 0:
                    # Corriger les valeurs
                    df[col] = np.where(df[col] < lower_bound, lower_bound, 
                                     np.where(df[col] > upper_bound, upper_bound, df[col]))
                    report[col] = {
                        'count': int(outliers_count),
                        'valeurs_inférieures': lower_bound,
                        'valeurs_supérieures': upper_bound
                    }
    return df, report

def handle_duplicates(df):
    """Gestion des doublons basée sur l'identifiant"""
    id_columns = [col for col in df.columns if is_id_column(col)]
    dup_report = {
        'count': 0,
        'message': "Aucun doublon détecté",
        'columns_used': "Aucune colonne d'identifiant trouvée"
    }
    
    if id_columns:
        id_col = id_columns[0]
        dup_report['columns_used'] = id_col
        
        dup_mask = df.duplicated(subset=[id_col], keep='first')
        dup_count = dup_mask.sum()
        
        if dup_count > 0:
            df = df[~dup_mask]
            dup_report['count'] = dup_count
            dup_report['message'] = f"{dup_count} doublons supprimés (basés sur {id_col})"
    else:
        dup_mask = df.duplicated(keep='first')
        dup_count = dup_mask.sum()
        
        if dup_count > 0:
            df = df[~dup_mask]
            dup_report['count'] = dup_count
            dup_report['message'] = f"{dup_count} doublons supprimés (toutes colonnes)"
            dup_report['columns_used'] = "Toutes les colonnes"
    
    return df, dup_report

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/televerser/')
def televerser():
    return render_template('televerser.html')

@app.route('/about/')
def about():
    return render_template('about.html')

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
            df[col] = df[col].clip(lower=0, upper=120).astype('Int64')
    
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