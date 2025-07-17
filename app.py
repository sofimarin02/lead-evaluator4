from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

#Se carga todo el modelo
model = pickle.load(open("model/lead_model.pkl", "rb"))
columnas = model.feature_names_in_

#  dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/bank-additional-full.csv"
df_original = pd.read_csv(url, sep=';')

# Campos visibles para el formulario
campos_visibles = [
    'age','job','marital','education','default','housing','loan','poutcome',
    'contact','month','day_of_week','campaign','pdays','previous',
    'duration','emp.var.rate','cons.price.idx','cons.conf.idx',
    'euribor3m','nr.employed'
]

# Rangos para validación numérica
rangos_numericos = {
    'age':           (18, 100),
    'campaign':      (1, 50),
    'pdays':         (-1, 999),
    'previous':      (0, 50),
    'duration':      (0, 5000),
    'emp.var.rate':  (-5, 5),
    'cons.price.idx':(90, 120),
    'cons.conf.idx': (-50, 50),
    'euribor3m':     (0, 10),
    'nr.employed':   (3000, 6000)
}

# Valores categóricos extraídos del CSV
valores_categoricos = {
    col: sorted(df_original[col].unique())
    for col in campos_visibles
    if df_original[col].dtype == 'object'
}
# Incluir manualmente las subcategorías de education si no aparecen todas
valores_categoricos['education'] = [
    'unknown','basic.4y','high.school','basic.6y','basic.9y',
    'professional.course','university.degree'
]

@app.route("/", methods=["GET"])
def formulario():
    return render_template(
        "api_test.html",
        campos=campos_visibles,
        valores_categoricos=valores_categoricos,
        rangos_numericos=rangos_numericos
    )

@app.route("/predict", methods=["POST"])
def predict():
    datos_formulario = request.get_json(force=True)

    # Validar campos
    faltantes = [c for c in campos_visibles if datos_formulario.get(c) is None]
    if faltantes:
        return jsonify({
            "error": f"Faltan campos: {', '.join(faltantes)}"
        }), 400

    # Construir dict de entrada
    datos_usuario = {}
    for campo in campos_visibles:
        valor = datos_formulario[campo]
        if campo in rangos_numericos:
            v = float(valor)
            mn, mx = rangos_numericos[campo]
            datos_usuario[campo] = max(min(v, mx), mn)
        else:
            datos_usuario[campo] = valor

    # Crear DataFrame y codificar
    df_usuario = pd.DataFrame([datos_usuario])
    df_encoded = pd.get_dummies(df_usuario) \
                   .reindex(columns=columnas, fill_value=0)

    # Predicción de probabilidad
    proba = model.predict_proba(df_encoded)[0][1]

    # Umbral óptimo obtenido en training
    best_threshold = 0.368
    pred = 1 if proba > best_threshold else 0

    resultado = "Likely lead" if pred == 1 else "Not a likely lead"
    return jsonify({"result": resultado})

if __name__ == "__main__":
    # Cloud Run inyecta la variable PORT, si no existe, caerá a 8080
    port = int(os.environ.get("PORT", 8080))
    # 0.0.0.0 para escuchar en todas las interfaces
    app.run(host="0.0.0.0", port=port, debug=True)

