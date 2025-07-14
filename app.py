import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

encoder = joblib.load('ordinal_encoder.pkl')
scaler = joblib.load('robust_scaler.pkl')
pca = joblib.load('pca_model.pkl')
model = joblib.load('random_forest_model.pkl')

categorical_cols = ['Sex', 'Title', 'Ticket']
features = ['Sex', 'Ticket', 'Age', 'Fare', 'Pclass', 'SibSp', 'Title', 'FamilySize']

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {}
        for feature in features:
            value = request.form.get(feature)
            if value is None:
                return jsonify({'error': f'Missing {feature}'}), 400
            data[feature] = value

        df = pd.DataFrame([data])
        
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

        # Predecir
        pred = model.predict(X_pca)

        return jsonify({'prediction': int(pred[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
