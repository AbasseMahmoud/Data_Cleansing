from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np


app = Flask(__name__)

def clean_dataframe(df):
    """Nettoyage générique pour n'importe quel dataframe"""
    cleaned_df = df.copy()
    
    # Conversion des colonnes numériques
    for col in cleaned_df.columns:
        # Essayer de convertir en numérique d'abord
        try:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='raise')
            continue  # Si la conversion réussit, passer à la colonne suivante
        except:
            pass
        
        # Traitement des valeurs manquantes pour les colonnes non numériques
        if cleaned_df[col].dtype == object:
            # Remplacer les variantes de NA/NaN
            cleaned_df[col] = cleaned_df[col].replace(
                ['n/a', 'NA', 'na', 'NaN', 'NULL', 'null', '--', '?', ''], np.nan)
            
            # Pour les colonnes catégorielles avec Y/N ou Oui/Non
            unique_vals = cleaned_df[col].dropna().unique()
            if all(str(x).upper() in ['Y', 'N', 'OUI', 'NON', 'YES', 'NO'] for x in unique_vals if pd.notnull(x)):
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: str(x).upper() if pd.notnull(x) else np.nan)
    
    return cleaned_df

def handle_missing_values(df):
    """Détection et traitement des valeurs manquantes pour n'importe quel dataframe"""
    missing_before = df.isnull().sum().to_dict()
    
    # Traitement différencié selon le type de colonne
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Pour les numériques: remplacer par la médiane
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            # Pour les catégorielles: remplacer par le mode
            if len(df[col].mode()) > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
            else:
                # Si pas de mode (toutes valeurs NA), remplacer par une string vide
                df[col] = df[col].fillna('')
    
    missing_after = df.isnull().sum().to_dict()
    
    report = {
        'before': missing_before,
        'after': missing_after,
        'message': "Valeurs manquantes traitées (médiane pour numérique, mode pour catégoriel)"
    }
    return df, report

def handle_outliers(df):
    """Détection et traitement des valeurs aberrantes pour les colonnes numériques"""
    report = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Calcul des limites avec IQR
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        if iqr > 0:  # Éviter les divisions par zéro pour colonnes constantes
            valeurs_inférieures = q1 - 1.5 * iqr
            valeurs_supérieures = q3 + 1.5 * iqr
            
            # Détection des outliers
            outliers = df[(df[col] < valeurs_inférieures) | (df[col] > valeurs_supérieures)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                # Winsorization
                df[col] = np.where(df[col] < valeurs_inférieures, valeurs_inférieures,
                                  np.where(df[col] > valeurs_supérieures, valeurs_supérieures, df[col]))
                
                report[col] = {
                    'count': outlier_count,
                    'valeurs_inférieures': round(valeurs_inférieures, 2),
                    'valeurs_supérieures': round(valeurs_supérieures, 2),
                    'method': 'Winsorization'
                }
    
    if not report:
        report = {"message": "Aucune valeur aberrante détectée dans les colonnes numériques"}
    return df, report

def handle_duplicates(df):
    """Détection et traitement des doublons"""
    duplicates = df[df.duplicated()]
    dup_count = len(duplicates)
    
    if dup_count > 0:
        df = df.drop_duplicates()
        report = {
            'count': dup_count,
            'message': f"{dup_count} doublons complets supprimés"
        }
        return df, report
    
    return df, {"message": "Aucun doublon complet détecté"}


@app.route('/televerser/')
def televerser():
    return render_template('televerser.html')

# Une fonction pour rediriger l'utilisateur vers la page about
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
    
    if not file.filename.endswith('.csv'):
        return render_template('error.html', message="Seuls les fichiers CSV sont acceptés")
    
    try:
        # Lecture du fichier
        df = pd.read_csv(file)
        original_shape = df.shape
        
        # Nettoyage et traitement
        df = clean_dataframe(df)
        df, missing_report = handle_missing_values(df)
        df, outliers_report = handle_outliers(df)
        df, duplicates_report = handle_duplicates(df)
        
        # Sauvegarde pour téléchargement
        fichier_traiter_csv = df.to_csv(index=False)
        with open('static/fichier_traiter_data.csv', 'w') as f:
            f.write(fichier_traiter_csv)
        
        # Préparation des résultats
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
    
    except Exception as e:
        return render_template('error.html', message=f"Erreur de traitement : {str(e)}")

@app.route('/download')
def download():
    return send_file('static/fichier_traiter_data.csv', 
                    as_attachment=True, 
                    mimetype='text/csv',
                    download_name='fichier_traiter_data.csv')

if __name__ == '__main__':
    app.run(debug=True)