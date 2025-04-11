from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import re
import os

app = Flask(__name__)

def is_age_column(col_name):
    """Détecte si une colonne est une colonne d'âge"""
    age_keywords = ['age', 'âge', 'years', 'year', 'an', 'ans']
    return any(keyword in col_name.lower() for keyword in age_keywords)

def is_id_column(col_name):
    """Détecte si une colonne est une colonne d'identifiant"""
    id_keywords = ['id', 'identifiant', 'matricule', 'numero']
    return any(keyword in col_name.lower() for keyword in id_keywords)

def clean_age_values(age_series):
    """Nettoyage spécifique pour les valeurs d'âge avec conversion en entiers"""
    cleaned_ages = []
    for value in age_series:
        if pd.isna(value):
            cleaned_ages.append(np.nan)
            continue
            
        # Convertir en chaîne si ce n'est pas déjà le cas
        str_value = str(value).strip()
        
        # Extraire les chiffres et point décimal
        digits = re.sub(r'[^\d.]', '', str_value)
        
        if digits:  # Si on a trouvé des chiffres
            try:
                # Convertir en float d'abord pour gérer les décimaux
                age = float(digits)
                # Corriger les valeurs négatives (mise à 0) et convertir en entier
                cleaned_age = int(max(0, age))  # Conversion en entier avec arrondi vers le bas
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
            # Conversion finale en Int64 (supportant les NaN)
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
            # Pour les âges, on utilise la médiane arrondie avec un minimum de 0
            median_val = int(max(0, np.nanmedian(df[col].dropna())))
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
    """Traitement des valeurs aberrantes avec conversion en entiers pour les âges"""
    report = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if is_age_column(col):
            # Traitement spécial pour l'âge - seulement corriger les valeurs négatives
            lower_bound = 0  # Minimum 0 pour l'âge
            upper_bound = df[col].max()  # Pas de maximum imposé
            
            # Appliquer les corrections et convertir en entier
            df[col] = df[col].clip(lower=lower_bound).astype('Int64')
            
            # Générer le rapport
            outliers_count = (df[col] < lower_bound).sum()  # Seulement compter les valeurs négatives
            if outliers_count > 0:
                report[col] = {
                    'count': int(outliers_count),
                    'lower_bound': lower_bound,
                    'upper_bound': "Aucune limite supérieure"
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

def handle_duplicates(df):
    """
    Gestion améliorée des doublons basée sur l'identifiant
    """
    # Trouver la colonne d'identifiant
    id_columns = [col for col in df.columns if is_id_column(col)]
    
    if not id_columns:
        # Si aucune colonne d'identifiant n'est trouvée, utiliser toutes les colonnes
        dup_mask = df.duplicated(keep='first')
        message = "Aucune colonne d'identifiant détectée - vérification sur toutes les colonnes"
    else:
        # Utiliser la première colonne d'identifiant trouvée
        id_col = id_columns[0]
        dup_mask = df.duplicated(subset=[id_col], keep='first')
        message = f"Doublons détectés sur la colonne d'identifiant: {id_col}"
    
    dup_count = dup_mask.sum()
    
    if dup_count > 0:
        df = df[~dup_mask]
        return df, {
            'count': dup_count,
            'message': f"{dup_count} doublons supprimés - {message}",
            'columns_used': id_columns[0] if id_columns else "toutes les colonnes"
        }
    return df, {
        'message': f"Aucun doublon détecté - {message}",
        'columns_used': id_columns[0] if id_columns else "toutes les colonnes"
    }

@app.route('/televerser/')
def televerser():
    return render_template('televerser.html')

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
        # elif file.filename.endswith('.xml'):
        #     df = pd.read_xml(file, parser='lxml')
        else:
            return render_template('error.html', message="Seuls les fichiers CSV et JSON sont acceptés")
    except Exception as e:
        return render_template('error.html', message=f"Erreur de lecture du fichier : {str(e)}")
    
    original_shape = df.shape
    
    # Traitement des données
    df = clean_dataframe(df)
    df, duplicates_report = handle_duplicates(df)  # Modification ici
    df, missing_report = handle_missing_values(df)
    df, outliers_report = handle_outliers(df)
    
    # Vérification finale des colonnes d'âge
    age_columns = [col for col in df.columns if is_age_column(col)]
    for col in age_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Ne corriger que les valeurs négatives et convertir en entier
            df[col] = df[col].clip(lower=0).astype('Int64')
    
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
    port = int(os.environ.get('PORT', 5000))  # Render définit automatiquement la variable d'env PORT
    app.run(host='0.0.0.0', port=port)