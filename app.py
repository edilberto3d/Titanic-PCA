from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar los artefactos
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('datos_scaler.pkl')
pca = joblib.load('pca.pkl')
model = joblib.load('modelo_final_randomFores.pkl')

# Columnas categóricas y final
categorical_cols = ['Sex', 'Ticket']
features = ['Sex', 'Ticket', 'Age', 'Fare', 'Pclass', 'SibSp', 'Title', 'FamilySize']

@app.route('/')
def home():
    return render_template('formulario.html')  # Asegúrate de tener este archivo

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recoger datos del formulario
        input_data = {feature: request.form[feature] for feature in features}

        # Convertir a DataFrame
        df = pd.DataFrame([input_data])

        # Convertir numéricos
        num_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'FamilySize']
        for col in num_cols:
            df[col] = df[col].astype(float)

        # Codificar categóricas
        df[categorical_cols] = encoder.transform(df[categorical_cols])

        # Escalar
        X_scaled = scaler.transform(df[features])

        # PCA
        X_pca = pca.transform(X_scaled)

        # Predicción
        prediction = model.predict(X_pca)[0]

        return jsonify({'prediction': int(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
